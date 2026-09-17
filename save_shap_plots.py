import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, pandas as pd, shap, numpy as np, os
from scipy.sparse import hstack, csr_matrix

print("Loading models...")
reg   = joblib.load('models/resolution_time_xgboost.joblib')
tfidf = joblib.load('models/tfidf_vectorizer_ensemble.joblib')

print("Loading data...")
df = pd.read_csv('data_processed/test.csv').head(100).fillna('')

# Build feature matrix
text_col = next((c for c in ['Combined_Text_Clean','Description_Clean','Ticket Description']
                 if c in df.columns), None)
text = tfidf.transform(df[text_col].astype(str)) if text_col else csr_matrix((len(df), tfidf.get_feature_names_out().shape[0]))

TABULAR = ['Customer Age','Ticket Channel_Encoded','Product Purchased_Encoded',
           'Customer Gender_Encoded','Customer_Tenure_Days',
           'Description_Char_Count','Description_Word_Count','Sentiment_Polarity']
avail = [c for c in TABULAR if c in df.columns]
print(f"Tabular features available: {avail}")

tab = csr_matrix(df[avail].fillna(0).values.astype(float))
X   = hstack([text, tab]).toarray()
print(f"Feature matrix shape: {X.shape}")

# Fix shape mismatch — pad or trim to match model's expected input
expected = reg.n_features_in_
actual   = X.shape[1]
print(f"Model expects {expected} features, got {actual}")
if actual < expected:
    pad = np.zeros((X.shape[0], expected - actual))
    X   = np.hstack([X, pad])
    print(f"Padded to {X.shape[1]} features")
elif actual > expected:
    X = X[:, :expected]
    print(f"Trimmed to {X.shape[1]} features")

print("Computing SHAP values (may take 30-60 seconds)...")
explainer   = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X)

os.makedirs('capstone_app/reports', exist_ok=True)
feat_names = list(tfidf.get_feature_names_out()) + avail
# Pad feature names to match X columns
while len(feat_names) < X.shape[1]:
    feat_names.append(f"feature_{len(feat_names)}")
feat_names = feat_names[:X.shape[1]]

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=feat_names,
                  max_display=15, show=False, plot_type='bar')
plt.title('SHAP Feature Importance — XGBoost Regressor (Resolution Time)')
plt.tight_layout()
plt.savefig('capstone_app/reports/shap_summary_plot.png', dpi=120, bbox_inches='tight')
plt.close()
print("✅ shap_summary_plot.png saved!")

# Waterfall plot for first sample
plt.figure(figsize=(10, 6))
explanation = shap.Explanation(values=shap_values[0],
                                base_values=explainer.expected_value,
                                data=X[0],
                                feature_names=feat_names)
shap.plots.waterfall(explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig('capstone_app/reports/shap_waterfall_plot.png', dpi=120, bbox_inches='tight')
plt.close()
print("✅ shap_waterfall_plot.png saved!")

print("Done! Refresh the SHAP page in your browser.")
