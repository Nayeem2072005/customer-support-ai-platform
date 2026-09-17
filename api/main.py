"""
api/main.py — FastAPI Inference Endpoint
AI-Powered Customer Support Intelligence Platform
GUVI × HCL Capstone

Endpoints:
    GET  /health       — Health check
    POST /predict      — Ticket type + priority + resolution time prediction
    GET  /docs         — Auto-generated Swagger UI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import os
from scipy.sparse import hstack, csr_matrix

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Customer Support AI Platform",
    description="""
## AI-Powered Customer Support Intelligence API

Automatically classifies, prioritizes, and estimates resolution time
for customer support tickets using machine learning.

### Endpoints
- **POST /predict** — Get ticket type, priority, and resolution time predictions
- **GET /health** — Check API health and model status
    """,
    version="1.0.0",
    contact={"name": "Nayeem Mohammed", "url": "https://github.com/Nayeem2072005"}
)

# ── Model paths ───────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")

# ── Load models at startup ────────────────────────────────────
tfidf  = None
le     = None
reg    = None

@app.on_event("startup")
def load_models():
    global tfidf, le, reg
    try:
        tfidf = joblib.load(os.path.join(MODELS, "tfidf_vectorizer_ensemble.joblib"))
        le    = joblib.load(os.path.join(MODELS, "label_encoders.joblib"))
        reg   = joblib.load(os.path.join(MODELS, "resolution_time_xgboost.joblib"))
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")

# ── Request / Response schemas ────────────────────────────────
class TicketRequest(BaseModel):
    ticket_subject: str = Field(
        ..., example="Cannot login to my account",
        description="Short subject line of the support ticket"
    )
    ticket_description: str = Field(
        ..., example="I have been trying to access my account for 2 days but keep getting password incorrect error.",
        description="Full description of the customer's issue"
    )
    customer_age: Optional[int] = Field(
        default=35, ge=18, le=100, example=35,
        description="Customer age in years"
    )
    ticket_channel: Optional[str] = Field(
        default="Email", example="Email",
        description="Channel: Email, Chat, Phone, Social Media"
    )
    product_purchased: Optional[str] = Field(
        default="GadgetX Pro", example="GadgetX Pro",
        description="Product the ticket is related to"
    )
    customer_gender: Optional[str] = Field(
        default="Male", example="Male",
        description="Customer gender: Male, Female, Other"
    )

class PredictionResponse(BaseModel):
    ticket_type: str
    ticket_type_method: str
    priority: str
    estimated_resolution_hours: float
    estimated_resolution_days: float
    confidence: str
    model_version: str = "1.0.0"

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    tfidf_loaded: bool
    regression_loaded: bool
    label_encoders_loaded: bool
    api_version: str = "1.0.0"

# ── Helper functions ──────────────────────────────────────────
def rule_based_ticket_type(text: str) -> str:
    """Rule-based ticket type classification."""
    t = text.lower()
    if any(w in t for w in ["billing","payment","charge","invoice","refund","money","overcharged"]):
        return "Billing Inquiry"
    elif any(w in t for w in ["cancel","cancellation","unsubscribe","terminate","subscription"]):
        return "Cancellation Request"
    elif any(w in t for w in ["error","crash","bug","not working","issue","broken","fail","glitch"]):
        return "Technical Issue"
    elif any(w in t for w in ["account","login","password","access","username","sign in","locked"]):
        return "Account Management"
    else:
        return "Product Inquiry"

def safe_encode(le_dict: dict, col: str, val: str) -> int:
    """Safely encode a categorical value."""
    if col in le_dict:
        le_obj = le_dict[col]
        if val in le_obj.classes_:
            return int(le_obj.transform([val])[0])
    return 0

def predict_resolution(text: str, age: int, channel: str,
                       product: str, gender: str, description: str) -> float:
    """Predict resolution time using XGBoost."""
    text_vec = tfidf.transform([text])
    ch_enc   = safe_encode(le, "Ticket Channel", channel)
    pr_enc   = safe_encode(le, "Product Purchased", product)
    gen_enc  = safe_encode(le, "Customer Gender", gender)
    tab      = csr_matrix([[age, ch_enc, pr_enc, gen_enc, 0, 0, len(description), 0]])
    X        = hstack([text_vec, tab]).toarray()

    # Fix shape mismatch
    expected = reg.n_features_in_
    if X.shape[1] < expected:
        X = np.hstack([X, np.zeros((1, expected - X.shape[1]))])
    elif X.shape[1] > expected:
        X = X[:, :expected]

    return float(abs(reg.predict(X)[0]))

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "AI-Powered Customer Support Intelligence Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API health and model loading status."""
    models_loaded = all([tfidf is not None, le is not None, reg is not None])
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        tfidf_loaded=tfidf is not None,
        regression_loaded=reg is not None,
        label_encoders_loaded=le is not None
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: TicketRequest):
    """
    Predict ticket type, priority level, and estimated resolution time.

    - **ticket_type**: Billing Inquiry / Technical Issue / Account Management / Product Inquiry / Cancellation Request
    - **priority**: Low / Medium / High / Critical (based on predicted resolution time)
    - **estimated_resolution_hours**: Predicted hours to resolve the ticket
    """
    combined_text = f"{request.ticket_subject} {request.ticket_description}".strip()

    if not combined_text:
        raise HTTPException(status_code=400, detail="Ticket subject and description cannot both be empty")

    # Ticket Type — rule-based NLP
    ticket_type = rule_based_ticket_type(combined_text)

    # Resolution Time + Priority — XGBoost
    if reg is not None and tfidf is not None:
        try:
            hours = predict_resolution(
                combined_text,
                request.customer_age,
                request.ticket_channel,
                request.product_purchased,
                request.customer_gender,
                request.ticket_description
            )
            method = "XGBoost Regressor"
        except Exception as e:
            hours  = 12.0
            method = f"fallback (error: {str(e)[:50]})"
    else:
        hours  = 12.0
        method = "fallback (models not loaded)"

    # Priority from hours
    if hours < 5:    priority = "Low"
    elif hours < 15: priority = "Medium"
    elif hours < 30: priority = "High"
    else:            priority = "Critical"

    return PredictionResponse(
        ticket_type=ticket_type,
        ticket_type_method="Rule-based NLP classifier",
        priority=priority,
        estimated_resolution_hours=round(hours, 2),
        estimated_resolution_days=round(hours / 24, 2),
        confidence="Medium"
    )

@app.get("/examples", tags=["Examples"])
def get_examples():
    """Get example requests to test the API."""
    return {
        "examples": [
            {
                "name": "Account Issue",
                "request": {
                    "ticket_subject": "Cannot login to my account",
                    "ticket_description": "I have been trying to access my account for 2 days but keep getting password incorrect error.",
                    "customer_age": 35,
                    "ticket_channel": "Email",
                    "product_purchased": "GadgetX Pro",
                    "customer_gender": "Male"
                }
            },
            {
                "name": "Billing Issue",
                "request": {
                    "ticket_subject": "Charged twice this month",
                    "ticket_description": "I noticed two charges of $49.99 on my credit card statement for the same subscription period.",
                    "customer_age": 42,
                    "ticket_channel": "Chat",
                    "product_purchased": "HomeSmartHub",
                    "customer_gender": "Female"
                }
            },
            {
                "name": "Technical Issue",
                "request": {
                    "ticket_subject": "App crashes on startup",
                    "ticket_description": "Every time I open the mobile app it crashes within 3 seconds. Tried reinstalling but the problem persists.",
                    "customer_age": 28,
                    "ticket_channel": "Phone",
                    "product_purchased": "FitTrack Band",
                    "customer_gender": "Male"
                }
            }
        ]
    }
