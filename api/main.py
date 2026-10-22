"""
api/main.py - FastAPI inference service
AI-Powered Customer Support Intelligence Platform (GUVI x HCL capstone)

Endpoints
    GET  /health      model-loading status
    POST /predict     ticket type + priority + resolution-time estimate
    GET  /model-info  measured test-set performance of every model
    GET  /docs        interactive Swagger UI

Each step mirrors how its model was trained:
    ticket type      fine-tuned DistilBERT on raw "subject. description"
    priority         LightGBM on TF-IDF + 8 tabular features
    resolution time  XGBoost on 5 tabular features (incl. predicted priority)
                     + TF-IDF of the cleaned description
"""

# torch must be imported before numpy/pandas on Windows (c10.dll / OpenMP
# clash hit in notebooks 07-08). A missing install is reported via /health.
try:
    import torch
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )
except ImportError:
    torch = None

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix, hstack
from textblob import TextBlob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.preprocessing import clean_text, get_text_stats  # noqa: E402

logger = logging.getLogger("uvicorn.error")

# ── Configuration ────────────────────────────────────────────────────────
API_VERSION = "1.0.0"
MODELS_DIR = os.path.join(BASE, "models")
DEVICE = os.getenv("API_DEVICE", "cpu")  # cpu by default = same as deployment
MAX_TOKENS = 128

# DistilBERT's config.json stores only LABEL_0..LABEL_4, so the label order is
# fixed here: the sorted order used in notebooks/08_DistilBERT_FineTuning.
TICKET_TYPE_CLASSES = [
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request",
    "Technical issue",
]

# Tenure feature, as built in notebooks/03_Preprocessing (reference date =
# latest Date of Purchase in the data; training range 0-729 days, median 364).
TENURE_REFERENCE_DATE = date(2021, 12, 30)
TENURE_MIN, TENURE_MAX, TENURE_MEDIAN = 0, 729, 364

COMPONENTS = [
    "label_encoders",
    "priority_tfidf",
    "priority_bundle",
    "regression_tfidf",
    "regression_model",
    "ticket_type_model",
]

QUALITY_NOTE = (
    "On this dataset every model scores at about chance level on held-out "
    "test data (see /model-info). Treat outputs as a working demonstration "
    "of the pipeline, not as decisions."
)

# Measured numbers from the project notebooks / scripts (test set).
# Update these if any model is retrained.
MODEL_INFO = {
    "ticket_type": {
        "model": "DistilBERT (distilbert-base-uncased, fine-tuned)",
        "source": "notebooks/08_DistilBERT_FineTuning.ipynb",
        "test_accuracy": 0.2116,
        "test_f1_macro": 0.2109,
        "test_roc_auc": 0.5042,
        "majority_class_baseline_accuracy": 0.2069,
        "random_guess_accuracy": 0.2,
    },
    "priority": {
        "model": "LightGBM (Optuna-tuned)",
        "source": "src/train_priority_model.py",
        "validation_accuracy": 0.2772,
        "test_accuracy": 0.2486,
        "test_f1_macro": 0.2485,
        "test_f1_weighted": 0.2492,
        "test_roc_auc": 0.4923,
        "majority_class_baseline_accuracy": 0.2651,
        "note": "Tuned on the validation set, so validation is optimistic.",
    },
    "resolution_time": {
        "model": "XGBoost (Optuna-tuned)",
        "source": "notebooks/09_Regression_Clustering.ipynb",
        "test_rmse_hours": 5.8875,
        "test_mae_hours": 4.8560,
        "test_r2": -0.0203,
        "mean_baseline_rmse_hours": 5.8751,
        "note": (
            "Target is |Time to Resolution - First Response Time| in hours. "
            "48.6% of closed tickets have a resolution timestamp BEFORE the "
            "first response, so the timestamps look randomly generated."
        ),
    },
    "dataset": {
        "rows": 8469,
        "ticket_types": TICKET_TYPE_CLASSES,
        "note": "Labels carry almost no learnable signal from the features.",
    },
}

ARTIFACTS: dict = {}
ERRORS: Dict[str, str] = {}


# ── Loading ──────────────────────────────────────────────────────────────
def _models_path(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def load_ticket_type_model():
    """Load the fine-tuned DistilBERT tokenizer and model."""
    if torch is None:
        raise RuntimeError("torch / transformers are not installed")
    path = _models_path("distilbert_ticket_type")
    tokenizer = DistilBertTokenizerFast.from_pretrained(path)
    model = DistilBertForSequenceClassification.from_pretrained(path)
    if model.config.num_labels != len(TICKET_TYPE_CLASSES):
        raise ValueError(
            f"model has {model.config.num_labels} labels, "
            f"API expects {len(TICKET_TYPE_CLASSES)}"
        )
    model.to(DEVICE).eval()
    return tokenizer, model


def _check_shapes() -> None:
    """Fail loudly if a model and its vectorizer disagree on feature count."""
    checks = [
        ("priority_bundle", "priority_tfidf", "tabular", 8),
        ("regression_model", "regression_tfidf", "fixed", 5),
    ]
    for model_key, tfidf_key, kind, fixed in checks:
        if model_key not in ARTIFACTS or tfidf_key not in ARTIFACTS:
            continue
        model = ARTIFACTS[model_key]
        if kind == "tabular":
            fixed = len(model["tabular_features"])
            model = model["model"]
        expected = len(ARTIFACTS[tfidf_key].vocabulary_) + fixed
        if model.n_features_in_ != expected:
            ERRORS[model_key] = (
                f"feature mismatch: model expects {model.n_features_in_}, "
                f"pipeline builds {expected}"
            )
            ARTIFACTS.pop(model_key)


def load_artifacts() -> None:
    """Load every model once at startup; record failures instead of crashing."""
    ARTIFACTS.clear()
    ERRORS.clear()
    loaders = {
        "label_encoders": lambda: joblib.load(_models_path("label_encoders.joblib")),
        "priority_tfidf": lambda: joblib.load(
            _models_path("tfidf_vectorizer_ensemble.joblib")
        ),
        "priority_bundle": lambda: joblib.load(
            _models_path("priority_lightgbm.joblib")
        ),
        "regression_tfidf": lambda: joblib.load(
            _models_path("tfidf_vectorizer_leakage_free.joblib")
        ),
        "regression_model": lambda: joblib.load(
            _models_path("resolution_time_xgboost.joblib")
        ),
        "ticket_type_model": load_ticket_type_model,
    }
    for name, loader in loaders.items():
        try:
            ARTIFACTS[name] = loader()
        except Exception as exc:  # report every failure through /health
            ERRORS[name] = f"{type(exc).__name__}: {exc}"
    _check_shapes()


def models_ready() -> bool:
    """True only when every component loaded and passed its shape check."""
    return all(name in ARTIFACTS for name in COMPONENTS)


# ── Schemas ──────────────────────────────────────────────────────────────
class TicketRequest(BaseModel):
    """One support ticket. Categorical fields must match the training data."""

    ticket_subject: str = Field(..., min_length=1, max_length=500)
    ticket_description: str = Field(..., min_length=1, max_length=5000)
    customer_age: int = Field(..., ge=18, le=100, description="Age in years")
    ticket_channel: str = Field(..., description="Chat, Email, Phone, Social media")
    product_purchased: str = Field(..., description="One of the 42 training products")
    customer_gender: str = Field(..., description="Female, Male or Other")
    date_of_purchase: Optional[date] = Field(
        default=None,
        description="YYYY-MM-DD. Optional; if missing the training median "
        "tenure (364 days) is used and a note is returned.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticket_subject": "Cannot log in after the update",
                    "ticket_description": (
                        "Since the latest update I keep getting an error "
                        "when I try to sign in to my account."
                    ),
                    "customer_age": 35,
                    "ticket_channel": "Email",
                    "product_purchased": "iPhone",
                    "customer_gender": "Male",
                    "date_of_purchase": "2021-06-15",
                }
            ]
        }
    }


class LabelPrediction(BaseModel):
    label: str
    confidence: float = Field(..., description="Probability of the chosen label")
    probabilities: Dict[str, float]
    method: str


class PredictionResponse(BaseModel):
    ticket_type: LabelPrediction
    priority: LabelPrediction
    estimated_resolution_hours: float
    resolution_method: str
    latency_ms: float
    notes: List[str]
    quality_note: str
    api_version: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    components: Dict[str, bool]
    errors: Dict[str, str]
    device: str
    api_version: str


# ── Feature helpers ──────────────────────────────────────────────────────
def canonical(value: str, valid: List[str], field: str) -> str:
    """Match a value to the training spelling (case-insensitive) or raise 422."""
    lookup = {v.lower(): v for v in valid}
    key = value.strip().lower()
    if key not in lookup:
        raise HTTPException(
            status_code=422,
            detail={
                "field": field,
                "message": f"Unknown value '{value}' for {field}.",
                "valid_values": valid,
            },
        )
    return lookup[key]


def encode(column: str, value: str) -> int:
    """Integer code of a category, using the encoders saved in training."""
    return int(ARTIFACTS["label_encoders"][column].transform([value])[0])


def compute_tenure(purchase_date: Optional[date], notes: List[str]) -> int:
    """Days between purchase and the training reference date (as in nb 03)."""
    if purchase_date is None:
        notes.append(
            f"date_of_purchase not given: used training median tenure "
            f"({TENURE_MEDIAN} days)."
        )
        return TENURE_MEDIAN
    days = (TENURE_REFERENCE_DATE - purchase_date).days
    clipped = min(max(days, TENURE_MIN), TENURE_MAX)
    if clipped != days:
        notes.append(
            f"date_of_purchase gives {days} days of tenure, outside the "
            f"training range {TENURE_MIN}-{TENURE_MAX}; used {clipped}."
        )
    return clipped


def predict_ticket_type(subject: str, description: str) -> np.ndarray:
    """DistilBERT class probabilities, in TICKET_TYPE_CLASSES order."""
    tokenizer, model = ARTIFACTS["ticket_type_model"]
    inputs = tokenizer(
        f"{subject}. {description}",  # same join as notebook 08
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True,
    )
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)[0].cpu().numpy()


def predict_priority(combined_clean: str, tabular: Dict[str, float]) -> np.ndarray:
    """LightGBM priority probabilities: TF-IDF columns first, then tabular."""
    bundle = ARTIFACTS["priority_bundle"]
    text_vec = ARTIFACTS["priority_tfidf"].transform([combined_clean])
    row = [tabular[name] for name in bundle["tabular_features"]]
    features = hstack([text_vec, csr_matrix([row])]).tocsr()
    return bundle["model"].predict_proba(features)[0]


def predict_resolution_hours(
    description_clean: str,
    age: int,
    channel_code: int,
    product_code: int,
    gender_code: int,
    priority_code: int,
) -> float:
    """XGBoost hours: 5 tabular columns first, then TF-IDF (as in nb 09)."""
    tabular = csr_matrix(
        [[age, channel_code, product_code, gender_code, priority_code]],
        dtype=float,
    )
    text_vec = ARTIFACTS["regression_tfidf"].transform([description_clean])
    features = hstack([tabular, text_vec]).tocsr()
    return max(0.0, float(ARTIFACTS["regression_model"].predict(features)[0]))


def _label_prediction(classes, probs, method: str) -> LabelPrediction:
    best = int(np.argmax(probs))
    return LabelPrediction(
        label=classes[best],
        confidence=round(float(probs[best]), 4),
        probabilities={c: round(float(p), 4) for c, p in zip(classes, probs)},
        method=method,
    )


def run_pipeline(request: TicketRequest) -> PredictionResponse:
    """Full prediction for one ticket (shared by /predict and the warm-up)."""
    start = time.perf_counter()
    notes: List[str] = []
    encoders = ARTIFACTS["label_encoders"]

    channel = canonical(
        request.ticket_channel,
        list(encoders["Ticket Channel"].classes_),
        "ticket_channel",
    )
    product = canonical(
        request.product_purchased,
        list(encoders["Product Purchased"].classes_),
        "product_purchased",
    )
    gender = canonical(
        request.customer_gender,
        list(encoders["Customer Gender"].classes_),
        "customer_gender",
    )
    channel_code = encode("Ticket Channel", channel)
    product_code = encode("Product Purchased", product)
    gender_code = encode("Customer Gender", gender)

    # Same text handling as training: clean subject and description
    # separately, then join (notebook 03).
    description_clean = clean_text(request.ticket_description)
    combined_clean = f"{clean_text(request.ticket_subject)} {description_clean}"
    stats = get_text_stats(request.ticket_description)
    tabular = {
        "Customer Age": request.customer_age,
        "Ticket Channel_Encoded": channel_code,
        "Product Purchased_Encoded": product_code,
        "Customer Gender_Encoded": gender_code,
        "Customer_Tenure_Days": compute_tenure(request.date_of_purchase, notes),
        "Description_Char_Count": stats["char_count"],
        "Description_Word_Count": stats["word_count"],
        "Sentiment_Polarity": TextBlob(request.ticket_description).sentiment.polarity,
    }

    type_probs = predict_ticket_type(request.ticket_subject, request.ticket_description)
    priority_probs = predict_priority(combined_clean, tabular)
    priority_label = ARTIFACTS["priority_bundle"]["classes"][
        int(np.argmax(priority_probs))
    ]
    hours = predict_resolution_hours(
        description_clean,
        request.customer_age,
        channel_code,
        product_code,
        gender_code,
        encode("Ticket Priority", priority_label),
    )

    return PredictionResponse(
        ticket_type=_label_prediction(
            TICKET_TYPE_CLASSES, type_probs, "DistilBERT (fine-tuned)"
        ),
        priority=_label_prediction(
            ARTIFACTS["priority_bundle"]["classes"], priority_probs, "LightGBM"
        ),
        estimated_resolution_hours=round(hours, 2),
        resolution_method="XGBoost regressor (uses the predicted priority)",
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        notes=notes,
        quality_note=QUALITY_NOTE,
        api_version=API_VERSION,
    )


WARM_UP_REQUEST = TicketRequest(
    ticket_subject="Warm up",
    ticket_description="Warm up request that runs once when the server starts.",
    customer_age=35,
    ticket_channel="Email",
    product_purchased="iPhone",
    customer_gender="Male",
)


# ── App ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup and run one warm-up prediction."""
    load_artifacts()
    if models_ready():
        try:
            run_pipeline(WARM_UP_REQUEST)
            logger.info("Models loaded and warmed up.")
        except Exception as exc:
            ERRORS["warm_up"] = f"{type(exc).__name__}: {exc}"
            logger.error("Warm-up failed: %s", exc)
    else:
        logger.error("Model loading problems: %s", ERRORS)
    yield


app = FastAPI(
    title="Customer Support AI Platform",
    description=(
        "Classifies a support ticket's **type** (DistilBERT), predicts its "
        "**priority** (LightGBM) and estimates **resolution time** (XGBoost). "
        "See `/model-info` for the measured, test-set quality of each model."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


@app.get("/", tags=["Root"])
def root():
    """Pointers to the useful endpoints."""
    return {
        "message": "AI-Powered Customer Support Intelligence Platform API",
        "api_version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "model_info": "/model-info",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Report which components loaded. 'degraded' means /predict will fail."""
    ready = models_ready()
    return HealthResponse(
        status="healthy" if ready and "warm_up" not in ERRORS else "degraded",
        models_loaded=ready,
        components={name: name in ARTIFACTS for name in COMPONENTS},
        errors=dict(ERRORS),
        device=DEVICE,
        api_version=API_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: TicketRequest):
    """Predict ticket type, priority and estimated resolution time."""
    if not models_ready():
        raise HTTPException(
            status_code=503,
            detail={"message": "Models are not loaded.", "errors": dict(ERRORS)},
        )
    return run_pipeline(request)


@app.get("/model-info", tags=["Health"])
def model_info():
    """Measured performance of each model on held-out test data."""
    return MODEL_INFO
