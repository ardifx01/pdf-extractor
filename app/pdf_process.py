import os
import shutil
import time
import random
from pathlib import Path
from typing import List
import pymupdf
import re

import pandas as pd
import requests

from urllib.parse import urlparse

from sympy import use

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

def ensure_temp_dir(dir_name: str | Path | List[str] | List[Path] = TEMP_DIR):
    """Ensure the temporary directory exists."""
    if isinstance(dir_name, list):
        for sub_dir in dir_name:
            os.makedirs(sub_dir, exist_ok=True)
    else:
        os.makedirs(dir_name, exist_ok=True)

def download_pdf(id: str = None, url: str = None, use_specific_id: bool = False):
    """
    Download a PDF file from a URL and save it to TEMP_DIR.
    Retries up to 3 times if download fails.

    Args:
        id (str, optional): The ID of the PDF file.
        url (str): The URL to download the PDF from.

    Yields:
        str: Status message.
    """
    ensure_temp_dir(TEMP_DIR_PDF)
    filename = ""
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            if use_specific_id and id:
                filename = TEMP_DIR_PDF / f"{id}.pdf"
            elif not use_specific_id and url:
                filename = TEMP_DIR_PDF / f"{urlparse(url).path.split('/')[-1]}"
            else:
                yield {
                    "status": "error",
                    "id": id,
                    "url": url,
                    "message": "❌ No ID or URL provided.",
                }
                return

            if os.path.exists(filename):
                if not is_pdf_valid_but_repaired(filename):
                    os.remove(filename)
                    continue  # Retry download
                else:
                    yield {
                        "status": "info",
                        "id": id,
                        "url": url,
                        "message": f"🟢 Using cached file for {filename.name}",
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
                "message": f"✅ Downloaded {filename.name}",
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
                    "message": f"❌ Failed to download {filename.name if filename else 'file' } after {max_retries} attempts: {str(e)}",
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

def handle_pdf_download_from_dataset(
    dataset_file: str,
    id_col: str,
    url_col: str,
    use_specific_id: bool = False,
):
    """
    Download PDFs from a dataset CSV or Excel file.

    Args:
        dataset_file (str): Path to the dataset file.
        id_col (str): Column name for the ID.
        url_col (str): Column name for the URL.
        use_specific_id (bool): Whether to use a specific ID for downloading.

    Yields:
        dict: Status message for each download attempt.
    """
    ensure_temp_dir(TEMP_DIR)
    df = read_dataset(dataset_file)
    for _, row in df.iterrows():
        pdf_id = None
        if use_specific_id:
            pdf_id = str(row[id_col])
        else:
            pdf_id = None
        url = row[url_col]
        yield from download_pdf(pdf_id, url, use_specific_id=use_specific_id)

def clear_temp_dir(dir_name: str | Path):
    """Clear the temporary directory when all processes are done."""
    shutil.rmtree(dir_name, ignore_errors=True)