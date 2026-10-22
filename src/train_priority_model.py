"""
src/train_priority_model.py

Trains and saves the ticket-PRIORITY model (LightGBM) used by the API.

It reuses, on purpose, the same ingredients as notebooks/06_Ensemble_Models.ipynb:
  - the saved train/val/test split in data_processed/
  - the same fitted TF-IDF vectorizer (models/tfidf_vectorizer_ensemble.joblib)
  - the best hyperparameters found by Optuna in that notebook
So the model here matches the one that notebook logged to MLflow.

Run from the project root:
    python src/train_priority_model.py
"""
import os

import joblib
import lightgbm as lgb
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RANDOM_STATE = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data_processed")
MODELS_DIR = os.path.join(BASE, "models")

# Same 8 tabular columns, same order, as notebook 06 (order matters).
TABULAR_FEATURES = [
    "Customer Age", "Ticket Channel_Encoded", "Product Purchased_Encoded",
    "Customer Gender_Encoded", "Customer_Tenure_Days",
    "Description_Char_Count", "Description_Word_Count", "Sentiment_Polarity",
]

# Best Optuna parameters from notebook 06 (LightGBM, ticket priority).
BEST_PARAMS = {
    "n_estimators": 217,
    "max_depth": 6,
    "learning_rate": 0.12622788897753512,
    "num_leaves": 94,
}


def build_features(df, tfidf):
    """Stack TF-IDF text columns first, then the tabular columns."""
    text = tfidf.transform(df["Combined_Text_Clean"].fillna(""))
    tabular = csr_matrix(df[TABULAR_FEATURES].values)
    return hstack([text, tabular]).tocsr()


def report(model, X, y_true, split_name):
    """Print the project's standard classification metrics for one split."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    acc = accuracy_score(y_true, pred)
    f1_macro = f1_score(y_true, pred, average="macro")
    f1_weighted = f1_score(y_true, pred, average="weighted")
    auc = roc_auc_score(y_true, proba, multi_class="ovr")
    majority = y_true.value_counts(normalize=True).max()
    print(f"\n{split_name} ({len(y_true)} rows)")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1-macro:    {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Majority-class baseline accuracy: {majority:.4f}")


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer_ensemble.joblib"))

    classes = sorted(train["Ticket Priority"].unique())
    label_map = {name: i for i, name in enumerate(classes)}
    print(f"Priority classes (index order): {label_map}")

    X_train = build_features(train, tfidf)
    X_val = build_features(val, tfidf)
    X_test = build_features(test, tfidf)
    y_train = train["Ticket Priority"].map(label_map)
    y_val = val["Ticket Priority"].map(label_map)
    y_test = test["Ticket Priority"].map(label_map)
    print(f"Feature matrix: {X_train.shape} "
          f"({X_train.shape[1] - len(TABULAR_FEATURES)} text + "
          f"{len(TABULAR_FEATURES)} tabular)")

    model = lgb.LGBMClassifier(**BEST_PARAMS, random_state=RANDOM_STATE, verbose=-1)
    model.fit(X_train, y_train)

    # Validation = reproducibility check against notebook 06.
    # Test = first time this model is scored on the untouched test set.
    report(model, X_val, y_val, "VALIDATION (should be close to notebook 06)")
    report(model, X_test, y_test, "TEST (untouched until now)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "priority_lightgbm.joblib")
    bundle = {
        "model": model,
        "classes": classes,
        "tabular_features": TABULAR_FEATURES,
    }
    joblib.dump(bundle, out_path)
    print(f"\nSaved priority model to {out_path}")


if __name__ == "__main__":
    main()
