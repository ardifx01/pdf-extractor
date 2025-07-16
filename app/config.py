from pathlib import Path

OUTPUT_DIR = Path("app/results")
TEMP_DIR = Path("app/temp")
ARTIFACT_DIR = Path("app/models")
YOLO_DIR = Path("app/yolo")

# Dictionary to map file types to their respective directories
TEMP_DIR_MAP = {
    "data": ["csv", "xlsx", "json"],
    "image": ["jpg", "jpeg", "png", "webp"],
    "pdf": ["pdf"],
    "video": ["mp4", "avi", "mkv", "m4a", "mov"],
    "audio": ["mp3", "wav", "aac", "flac"],
}