"""
Customer Support Intelligence Platform - Step 1b: Project Structure Setup
=============================================================================
Brief requirement (Day 1-2): "Set up Git repo and project folder structure
(src/, notebooks/, models/, mlruns/, api/). Configure MLflow tracking server
locally."

Run this ONCE, from inside your project folder (e.g. "Final Project"),
before any other work - the brief explicitly wants version control from
Day 1, not added later.
"""
import os
import subprocess

FOLDERS = ["src", "notebooks", "models", "mlruns", "api", "data_raw", "reports"]

for folder in FOLDERS:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}/")

# .gitignore - keep large/generated files out of version control
gitignore_content = """\
# Data (large, and shouldn't be duplicated in git - reference it, don't commit it)
data_raw/*.csv
*.zip

# MLflow tracking data (can get large fast with many runs)
mlruns/

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Environment
.env
venv/
"""
with open(".gitignore", "w") as f:
    f.write(gitignore_content)
print("Created: .gitignore")

# requirements.txt - pinned as the brief requires ("Pin all package versions")
requirements_content = """\
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
nltk==3.8.1
spacy==3.7.5
xgboost==2.1.0
lightgbm==4.5.0
torch==2.4.0
transformers==4.44.0
mlflow==2.15.1
optuna==3.6.1
shap==0.46.0
fastapi==0.112.0
uvicorn==0.30.5
matplotlib==3.9.1
seaborn==0.13.2
plotly==5.23.0
wordcloud==1.9.3
gdown==5.2.0
joblib==1.4.2
"""
with open("requirements.txt", "w") as f:
    f.write(requirements_content)
print("Created: requirements.txt (pinned versions, per brief's reproducibility requirement)")

# --- Git init and first commit ---
print("\nInitializing Git repository...")
subprocess.run(["git", "init"], check=False)
subprocess.run(["git", "add", ".gitignore", "requirements.txt"], check=False)
subprocess.run(["git", "add", "src", "notebooks", "models", "api", "reports"], check=False)
result = subprocess.run(["git", "commit", "-m", "chore: initial project structure setup"], check=False)

print("\n" + "=" * 60)
print("PROJECT STRUCTURE SETUP COMPLETE")
print("=" * 60)
print("""
Folder structure created:
  src/          - reusable preprocessing/model code (not just notebook cells)
  notebooks/    - Jupyter notebooks (EDA, modeling, etc.)
  models/       - saved trained models (joblib, HF format)
  mlruns/       - MLflow's local tracking data (auto-populated when we log runs)
  api/          - FastAPI app for deployment
  data_raw/     - the original CSV (never overwritten, per brief's data handling rule)
  reports/      - final project report, business insights

Git repo initialized with first commit. From now on, commit after every
meaningful milestone with descriptive messages (e.g. "feat: add EDA notebook"),
exactly as the brief specifies.
""")
