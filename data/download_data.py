
import os
import shutil
from pathlib import Path
import kagglehub

def download_data(target_folder="../data", filename="creditcard.csv"):

    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)

    destination_file = target_path / filename

    if destination_file.exists():
        print(f"[INFO] File already exists at: {destination_file}. Skipping download.")
        return str(destination_file)

    print("[INFO] Downloading dataset from Kaggle using kagglehub...")
    try:
        downloaded_dir = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        downloaded_path = Path(downloaded_dir)

        csv_files = list(downloaded_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset.")

        shutil.copy(csv_files[0], destination_file)
        print(f"[SUCCESS] Dataset successfully saved to: {destination_file}")
        return str(destination_file)

    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print("[TIP] Make sure you have installed kagglehub: pip install kagglehub")
        return None

