import os
import shutil
import time
import random
from pathlib import Path
import pymupdf
import re

import pandas as pd
import requests

# Constants
TEMP_DIR_PDF = Path("app/temp/pdf")
TEMP_DIR = Path("app/temp/")
EXTENSION = {
    "csv": ".csv",
    "xlsx": [".xlsx", ".xls"],
}

# Configure session
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})

def is_pdf_valid_but_repaired(filename: str) -> bool:
    try:
        doc = pymupdf.open(filename)
        doc.load_page(0)
        # Explicitly trigger MuPDF warnings if available
        logs = pymupdf.TOOLS.mupdf_warnings()
        doc.close()
    except Exception:
        return False  # File memang rusak dan gak bisa dibuka

    # Use regex to match repair/warning messages
    if re.search(r"(repairing PDF document|format error)", logs, re.IGNORECASE):
        return False  # File bisa dibuka, tapi sebenarnya rusak
    return True

def ensure_temp_dir():
    """Ensure the temporary directory exists."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR_PDF, exist_ok=True)
    

def download_pdf(id: str, url: str):
    """
    Download a PDF file from a URL and save it to TEMP_DIR.
    Retries up to 3 times if download fails.

    Args:
        id (str): The ID of the PDF file.
        url (str): The URL to download the PDF from.

    Yields:
        str: Status message.
    """
    ensure_temp_dir()
    filename = os.path.join(TEMP_DIR_PDF, f"{id}.pdf")
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            if os.path.exists(filename):
                if not is_pdf_valid_but_repaired(filename):
                    os.remove(filename)
                    continue  # Retry download

                else:
                    yield {
                        "status": "info",
                        "id": id,
                        "url": url,
                        "message": f"🟢 Using cached file for {id}.pdf",
                    }
                    return

            with session.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

            time.sleep(random.uniform(0.5, 2))  # Delay supaya gak disangka brute force
            yield {
                "status": "success",
                "id": id,
                "url": url,
                "message": f"✅ Downloaded {id}.pdf",
            }
            return
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            else:
                yield {
                    "status": "error",
                    "id": id,
                    "url": url,
                    "message": f"❌ Failed to download {id}.pdf after {max_retries} attempts: {str(e)}",
                }

def read_dataset(dataset_file: str | Path) -> pd.DataFrame:
    """
    Read a dataset file (CSV or Excel) into a DataFrame.

    Args:
        dataset_file (str | Path): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    dataset_file = str(dataset_file)
    if dataset_file.lower().endswith(EXTENSION["csv"]):
        # Try to auto-detect delimiter ("," or ";")
        with open(dataset_file, "r", encoding="utf-8") as f:
            sample = f.read(2048)
        comma_count = sample.count(",")
        semicolon_count = sample.count(";")
        delimiter = "," if comma_count >= semicolon_count else ";"
        return pd.read_csv(dataset_file, delimiter=delimiter)
    elif any(dataset_file.lower().endswith(ext) for ext in EXTENSION["xlsx"]):
        return pd.read_excel(dataset_file)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

def handle_pdf_download_from_dataset(dataset_file: str, id_col: str, url_col: str):
    """
    Download PDFs from a dataset CSV or Excel file.

    Args:
        dataset_file (str): Path to the dataset file.
        id_col (str): Column name for the ID.
        url_col (str): Column name for the URL.

    Yields:
        str: Status message for each download attempt.
    """
    ensure_temp_dir()
    df = read_dataset(dataset_file)
    for _, row in df.iterrows():
        id = str(row[id_col])
        url = row[url_col]
        for status in download_pdf(id, url):
            yield status
    

def clear_temp_dir(dir_name: str | Path):
    """Clear the temporary directory when all processes are done."""
    shutil.rmtree(dir_name, ignore_errors=True)