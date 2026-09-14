"""Automated dataset downloader for MachineGuard.

Downloads and prepares:
  1. NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)
  2. AI4I 2020 Predictive Maintenance Dataset (UCI)

Idempotent: skips files that already exist.

Usage:
    python scripts/download_datasets.py
"""

import io
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# --- C-MAPSS ---
# Multiple download sources (NASA redirect is unreliable)
CMAPSS_URLS = [
    "https://ti.arc.nasa.gov/c/6/",
    "https://data.nasa.gov/download/ff5v-kuh6/application%2Fx-zip-compressed",
]
# Fallback: Kaggle dataset (requires kaggle CLI)
CMAPSS_KAGGLE = "behrad3d/nasa-cmaps"
CMAPSS_DIR = DATA_DIR / "cmapss"

# Expected files after extraction
CMAPSS_FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
]

# --- AI4I ---
AI4I_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
AI4I_DIR = DATA_DIR / "ai4i"
AI4I_FILE = "ai4i2020.csv"


def download_file(url: str, dest: Path, desc: str = "Downloading") -> Path:
    """Download a file with progress bar. Returns path to downloaded file."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=120, allow_redirects=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return dest


def _is_valid_zip(path: Path) -> bool:
    """Check if a file is a valid ZIP archive."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.testzip()
        return True
    except (zipfile.BadZipFile, Exception):
        return False


def download_cmapss() -> None:
    """Download and extract NASA C-MAPSS dataset."""
    # Check if already extracted
    if all((CMAPSS_DIR / f).exists() for f in CMAPSS_FILES[:3]):
        print("[OK] C-MAPSS FD001 already present, skipping download.")
        return

    CMAPSS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CMAPSS_DIR / "CMAPSSData.zip"

    downloaded = False

    # Try each URL
    for url in CMAPSS_URLS:
        print(f"Trying C-MAPSS download from: {url}")
        try:
            download_file(url, zip_path, desc="C-MAPSS")

            if _is_valid_zip(zip_path):
                print("Extracting C-MAPSS...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(CMAPSS_DIR)

                # Move files from any subdirectory to the cmapss root
                for f in CMAPSS_DIR.rglob("*.txt"):
                    if f.parent != CMAPSS_DIR:
                        target = CMAPSS_DIR / f.name
                        if not target.exists():
                            f.rename(target)

                zip_path.unlink(missing_ok=True)
                print("[OK] C-MAPSS dataset extracted successfully.")
                downloaded = True
                break
            else:
                print(f"  Downloaded file is not a valid ZIP. Trying next source...")
                zip_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"  Download failed: {e}")
            zip_path.unlink(missing_ok=True)

    if not downloaded:
        print("[FAIL] All direct download URLs failed.")
        print()
        print("  Please download manually:")
        print("  Option 1: Kaggle CLI")
        print(f"    pip install kaggle")
        print(f"    kaggle datasets download -d {CMAPSS_KAGGLE} -p {CMAPSS_DIR} --unzip")
        print()
        print("  Option 2: Manual download")
        print("    1. Go to https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
        print("    2. Download 'Turbofan Engine Degradation Simulation Data Set'")
        print(f"    3. Extract to: {CMAPSS_DIR}")
        print()
        _try_kaggle_cmapss()


def _try_kaggle_cmapss() -> None:
    """Attempt to download C-MAPSS via Kaggle CLI as fallback."""
    import subprocess

    try:
        print("Attempting Kaggle CLI fallback...")
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", CMAPSS_KAGGLE,
                "-p", str(CMAPSS_DIR),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("[OK] C-MAPSS downloaded via Kaggle CLI.")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[FAIL] Kaggle CLI not available or failed: {e}")
        print("  Please install: pip install kaggle")
        print("  And configure: https://www.kaggle.com/docs/api")


def download_ai4i() -> None:
    """Download and extract UCI AI4I 2020 dataset."""
    csv_path = AI4I_DIR / AI4I_FILE

    if csv_path.exists():
        print("[OK] AI4I dataset already present, skipping download.")
        return

    AI4I_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI AI4I 2020 Predictive Maintenance dataset...")
    print(f"  URL: {AI4I_URL}")

    try:
        zip_path = AI4I_DIR / "ai4i_dataset.zip"
        download_file(AI4I_URL, zip_path, desc="AI4I 2020")

        # Extract
        print("Extracting AI4I...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(AI4I_DIR)

        # Find the CSV (may be in a subdirectory)
        for f in AI4I_DIR.rglob("*.csv"):
            if f.parent != AI4I_DIR:
                target = AI4I_DIR / f.name
                if not target.exists():
                    f.rename(target)

        zip_path.unlink()
        print("[OK] AI4I dataset extracted successfully.")

    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
        print(f"\n  Manual download:")
        print(f"  1. Go to https://archive.ics.uci.edu/dataset/601")
        print(f"  2. Download the ZIP file")
        print(f"  3. Extract CSV to: {AI4I_DIR}")


def main() -> None:
    """Download all datasets."""
    print("=" * 60)
    print("MachineGuard -- Dataset Downloader")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}\n")

    download_cmapss()
    print()
    download_ai4i()

    print("\n" + "=" * 60)
    print("Dataset download complete.")
    print("=" * 60)

    # Verify
    print("\nVerification:")
    fd001_train = CMAPSS_DIR / "train_FD001.txt"
    if fd001_train.exists():
        lines = fd001_train.read_text().strip().split("\n")
        print(f"  C-MAPSS FD001 train: {len(lines)} rows [OK]")
    else:
        print("  C-MAPSS FD001 train: NOT FOUND [FAIL]")

    ai4i_csv = AI4I_DIR / AI4I_FILE
    if ai4i_csv.exists():
        lines = ai4i_csv.read_text().strip().split("\n")
        print(f"  AI4I 2020: {len(lines) - 1} rows (+ header) [OK]")
    else:
        print("  AI4I 2020: NOT FOUND [FAIL]")


if __name__ == "__main__":
    main()
