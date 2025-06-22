"""This is a helper module for processing PDF such as 
logging process status and checking JSON file existence.
"""

import os
import json
from pathlib import Path
from typing import Any


def logging_process(status: str, message: str):
    """Logs the process status and message.

    Args:
        status (str): The status of the process.
        message (str): The message to log.

    Returns:
        dict: A dictionary containing the status and message.
    """
    return {
        "status": status,
        "message": message,
    }


def check_json_file_exists(file_path: Any | Path):
    """Checks if a JSON file exists and is have content.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        bool: True if the file exists and has content, False otherwise.
    """
    if file_path.exists():
        json_content = json.loads(file_path.read_text(encoding="utf-8"))
        total_pages = json_content.get("total_page", 0) or 0
        total_page_extracted = len(json_content.get("content", [])) or 0
        if total_pages == total_page_extracted:
            return True
    return False
