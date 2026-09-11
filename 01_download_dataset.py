"""
Customer Support Intelligence Platform - Step 1: Dataset Download & Inspection
=================================================================================
Downloads the dataset zip from the shared Google Drive link, extracts it,
and prints a full inspection (columns, shape, dtypes, missing values, sample
rows) so we know exactly what we're working with before building anything.
"""

import os
import zipfile
import pandas as pd

# --- Download ---
# gdown handles Google Drive's "large file" confirmation page automatically,
# which a plain requests/urllib download would get stuck on.
try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system("pip install gdown --quiet")
    import gdown

FILE_ID = "1Xr-7TlAunE4uYI07sxhTOXiIcs0wZboz"
ZIP_PATH = "Customer_Experience.zip"

if not os.path.exists(ZIP_PATH):
    print("Downloading dataset from Google Drive...")
    gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False)
else:
    print(f"{ZIP_PATH} already exists, skipping download.")

# --- Extract ---
EXTRACT_DIR = "data_raw"
os.makedirs(EXTRACT_DIR, exist_ok=True)

print(f"\nExtracting {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
    print("Files extracted:")
    for name in zip_ref.namelist():
        print(f"  {name}")

# --- Find the CSV inside (name may vary) ---
csv_files = [f for f in os.listdir(EXTRACT_DIR) if f.endswith(".csv")]
if not csv_files:
    # Check one level deeper in case it extracted into a subfolder
    for root, dirs, files in os.walk(EXTRACT_DIR):
        csv_files += [os.path.join(root, f) for f in files if f.endswith(".csv")]

if not csv_files:
    print("\nERROR: No CSV file found after extraction. Check the extracted contents manually.")
else:
    csv_path = csv_files[0] if os.path.isabs(csv_files[0]) or "/" in csv_files[0] else f"{EXTRACT_DIR}/{csv_files[0]}"
    print(f"\nFound CSV: {csv_path}")

    # --- Inspect ---
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col}")

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  None")

    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    if "Ticket Type" in df.columns:
        print(f"\nTicket Type distribution:")
        print(df["Ticket Type"].value_counts())

    if "Ticket Priority" in df.columns:
        print(f"\nTicket Priority distribution:")
        print(df["Ticket Priority"].value_counts())

    print(f"\nSaved for later use at: {csv_path}")
