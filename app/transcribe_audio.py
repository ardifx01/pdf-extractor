from pathlib import Path
import json
import whisper
import torch
import pydub
import gc

from export_results import ARTIFACT_PATH
from pdf_process import ensure_temp_dir

from helper import logging_process, check_json_file_exists

PARAM = {
    "language": "id",
    "condition_on_previous_text": False,
    "temperature": 0.0,
    "no_speech_threshold": 0.9,
    "logprob_threshold":-2.0,
    "verbose":True,
}
VIDEO_PATH = Path("app/temp/video")
WAV_PATH = Path("app/temp/wav")
OUTPUT_DIR = Path("app/results/transcribed_audio")

def group_segments_by_minute(segments, chunk_length=60):
    """Groups segments into chunks based on a fixed chunk length and returns in result_json format."""
    result_json = {
        "content": [],
        "total_segments": 0,
    }
    if not segments:
        return result_json
    max_end = max(segment["end"] for segment in segments)
    start = 0
    counter = 0
    while start < max_end:
        end = start + chunk_length
        texts = []
        for segment in segments:
            seg_start = segment["start"]
            seg_end = segment["end"]
            if seg_start < end and seg_end >= start:
                texts.append(segment["text"].strip())
        if texts:
            result_json["content"].append(
                {
                    "segment": counter,
                    "start_time": start,
                    "end_time": end,
                    "content": " ".join(texts),
                }
            )
            counter += 1
        start = end
    result_json["total_segments"] = len(result_json["content"])
    return result_json


def preprocess_audio(file_path, format="wav", sample_rate=16000, channels=1):
    """
    Preprocess audio file to ensure it is in the correct format.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        str: Path to the processed audio file.
    """
    # Ensure the temporary directories exist
    ensure_temp_dir(WAV_PATH)
    audio = pydub.AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(channels)
    processed_path = WAV_PATH / f"{Path(file_path).stem}.{format}"
    audio.export(processed_path, format=format)
    return str(processed_path)

def transcribe_audio(file_path, model_type="large", overwrite=True):
    """
    Transcribe audio file using Whisper model.

    Args:
        file_path (str): Path to the audio file.
        model_type (str): Type of Whisper model to use.
        overwrite (bool): Whether to overwrite existing results.

    Yields:
        str: Logging messages during the process.
    """
    file_path_obj = Path(file_path)
    output_file = OUTPUT_DIR / f"{file_path_obj.stem}.json"

    if not overwrite and check_json_file_exists(output_file):
        yield logging_process(
            "info", f"[SKIP] JSON result already exists for {file_path_obj.name}, skipping."
        )
        return

    if file_path_obj.suffix.lower() not in (".mp3", ".wav", ".m4a", ".mp4", ".mov", ".flac", ".aac", ".ogg"):
        yield logging_process("error", f"Unsupported file type for conversion: {file_path}")
        return

    yield logging_process("info", f"Processing audio file: {file_path}")
    processed_audio = preprocess_audio(file_path)

    yield logging_process("info", f"Transcribing audio file: {file_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_type, device=device, download_root=str(ARTIFACT_PATH))

    result = model.transcribe(str(processed_audio), **PARAM)
    segments = result.get("segments", [])
    minute_chunks = group_segments_by_minute(segments, chunk_length=60)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(minute_chunks, f, ensure_ascii=False, indent=4)

    yield logging_process(
        "success",
        f"Transcription completed for {file_path_obj.name}"
    )

    # Clean up resources to free memory
    del model, processed_audio, result, minute_chunks
    gc.collect()
