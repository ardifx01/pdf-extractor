import os
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO
import re
import gc
import pandas as pd
import time
from pathlib import Path
from glob import glob

from export_results import (
    OUTPUT_DIR,
)

from helper import logging_process, check_json_file_exists

FOLDER_OUTPUT_PYMU_TESSERACT = OUTPUT_DIR / "pymu_tesseract_finetuned"

def get_latest_yolo_model_path(yolo_dir="app/yolo"):
    """
    Get the latest YOLO model file from the specified directory.
    Args:
        yolo_dir (str): Directory where YOLO model files are stored.
    Returns:
        Path: Path to the latest YOLO model file.
    Raises:
        FileNotFoundError: If no YOLO model files are found in the specified directory.
    """
    yolo_files = sorted(
        glob(str(Path(yolo_dir) / "*.pt")),
        key=lambda f: os.path.getmtime(f),
        reverse=True,
    )
    if not yolo_files:
        raise FileNotFoundError(f"YOLO model file not found at {yolo_dir}/")
    return Path(yolo_files[0])

def page_to_image(page, dpi=300):
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples), zoom

def clean_text(text):
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = "\n".join([line.strip() for line in text.splitlines()])
    text = text.replace('\t', ' ')
    return text.strip()

def mask_image_with_yolo(image, model):

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model.predict(img_cv, verbose=False)

    target_labels = list(model.names.values())
    bounding_boxes = {label: [] for label in target_labels}
    label_masking = ["Non-Text"]  # Labels to mask
    masked_cv = img_cv.copy()

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box[:4])
            label_id = int(cls)
            label_name = model.names[label_id]
            bounding_boxes[label_name].append((x1, y1, x2, y2))

            if label_name in label_masking:
                cv2.rectangle(masked_cv, (x1, y1), (x2, y2), (255, 255, 255), -1)  # Draw rectangle with red border and thickness 2

    masked_pil = Image.fromarray(cv2.cvtColor(masked_cv, cv2.COLOR_BGR2RGB))

    return masked_pil, bounding_boxes


def extract_text_from_image(image):
    data = pytesseract.image_to_data(image, config="--oem 3 --psm 4", lang="eng+id", output_type=pytesseract.Output.DICT)
    confidences = [conf for conf in data['conf'] if conf != -1]
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    text = " ".join([
        data['text'][i]
        for i in range(len(data['text']))
        if data['conf'][i] != -1 and data['text'][i].strip() != ""
    ])

    return text, avg_confidence


def extract_pdf_single_page(doc, base_name, model_yolo, page_number):
    """
    Extract text and tables from a single PDF page using a combination of YOLO object detection and OCR.
    This function processes a PDF page by:
    1. Converting the page to an image
    2. Using YOLO to detect and mask non-text elements
    3. Applying redactions to exclude non-text objects from the PDF
    4. Identifying text regions and tables
    5. Extracting content from each region in order (top to bottom)
    Parameters
    ----------
    doc : fitz.Document
        The PyMuPDF document object containing the PDF
    base_name : str
        Base name of the PDF file (used for logging)
    model_yolo : object
        Loaded YOLO model for text/non-text detection
    page_number : int
        The page number to process (0-indexed)
    Returns
    -------
    tuple
        A tuple containing:
        - combined_content (str): The extracted text and table content
        - confidence (float): The OCR confidence score (average if multiple text regions)
    Notes
    -----
    The function sorts detected elements by their vertical position (y-coordinate)
    and processes them sequentially. Tables are extracted using PyMuPDF's table detection,
    while text is extracted using OCR after appropriate masking.
    """

    page = doc.load_page(page_number)
    
    img, zoom = page_to_image(page)
    draw = ImageDraw.Draw(img)

    # Masking gambar
    mask_image, bounding_boxes = mask_image_with_yolo(img, model_yolo)
    # mask_image.save("temp_masked_image.png")
    draw = ImageDraw.Draw(mask_image)
    

    # Masking PDF dari exclude object
    for label, bboxes in bounding_boxes.items():
        if label == "Non-Text":
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                rect = fitz.Rect(x1 // zoom, y1 // zoom, x2 // zoom, y2 // zoom)
                page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()

    # Menyimpan jumlah label yang lebih dari 1
    label_count = {}
    # Hanya menyimpan bounding box untuk label yang diinginkan
    text_bounding_box = {}
    for label, bboxes in bounding_boxes.items():
        if bboxes and label in ["Text"]:
            for bbox in bboxes:
                if label not in label_count:
                    label_count[label] = 1
                else:
                    label_count[label] += 1

                unique_label = f"{label}{label_count[label]}"
                if unique_label not in text_bounding_box:
                    text_bounding_box[unique_label] = []
                text_bounding_box[unique_label].append(bbox)

    # print(f"\n Bounding box teks:", text_bounding_box)
    
    # Menyimpan bounding box tabel dan objek tabel dalam satu dictionary
    table_bounding_box = {}
    tables = page.find_tables(strategy="lines_strict")
    tables = tables.tables

    # print(tables)
    if tables:
        for index, x in enumerate(tables, start=1):
            table_key = f"Table{index}" if len(tables) > 1 else "Table"
            if table_key not in table_bounding_box:
                table_bounding_box[table_key] = {"bounding_box": [], "objects": []}
            table_bounding_box[table_key]["bounding_box"].append((x.bbox[0], x.bbox[1], x.bbox[2], x.bbox[3]))
            table_bounding_box[table_key]["objects"].append(x)

    # print(f"\n Data tabel:", table_bounding_box)
    
    combined_data = {}

    # Tambahkan data tabel (dengan objek)
    if table_bounding_box:  # Periksa apakah table_bounding_box tidak kosong
        for label, info in table_bounding_box.items():
            if info["bounding_box"]:  # Periksa apakah bounding_box tidak kosong
                # Kalikan koordinat bounding box tabel dengan zoom
                scaled_bounding_boxes = [
                    (int(x1 * zoom), int(y1 * zoom), int(x2 * zoom), int(y2 * zoom))
                    for x1, y1, x2, y2 in info["bounding_box"]
                ]
                combined_data[label] = {
                    "bbox": scaled_bounding_boxes[0],  # Ambil bounding box pertama
                    "objects": info["objects"]  # bisa berupa list berisi 1 table object
                }

    # Tambahkan data teks (tanpa objek)
    if text_bounding_box:  # Periksa apakah text_bounding_box tidak kosong
        for label, bboxes in text_bounding_box.items():
            if bboxes:  # Periksa apakah bboxes tidak kosong
                combined_data[label] = {
                    "bbox": bboxes[0],  # kamu bilang setiap label hanya punya 1 bbox
                    "objects": []
                }

    
    # Urutkan berdasarkan y1 (bbox[1])
    sorted_combined = dict(
        sorted(combined_data.items(), key=lambda item: item[1]["bbox"][1])
    )

    # print(f"\n Gabungan bounding box:", sorted_combined)

    # Ekstraksi teks dari gambar yang sudah dimask
    # print(f"Jumlah label yang ditemukan: {len(sorted_combined)}")

    if sorted_combined:
        combined_content = ""
        confidences = []

        working_image1 = mask_image.copy()
        draw1 = ImageDraw.Draw(working_image1)

        for label, info in sorted_combined.items():
            bbox = info["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            draw1.rectangle([x1, y1, x2, y2], fill="white", outline="red")


        for idx, (label, info) in enumerate(sorted_combined.items()):
            bbox = info["bbox"]
            

            if label.startswith("Text"):
                # Draw bounding box for the current text label
        
                working_image2 = mask_image.copy()
                draw2 = ImageDraw.Draw(working_image2)

                # Masking seluruh objek kecuali footer atau header saat ini
                for other_label, other_info in sorted_combined.items():
                    if other_label != label:
                        other_bbox = other_info["bbox"]
                        if isinstance(other_bbox, (list, tuple)) and len(other_bbox) == 4:
                            x1, y1, x2, y2 = map(int, other_bbox)
                            draw2.rectangle([x1, y1, x2, y2], fill="white", outline="red")

                # Simpan gambar hasil masking untuk label saat ini
                # x1, y1, x2, y2 = map(int, bbox)
                # draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)

                # Display the image with bounding box
                # working_image.show()

                # Save the image with bounding box
                # working_image.save(f"output_text_bounding_box_{label}_{idx}.png")

                raw_text, confidence = extract_text_from_image(working_image2)
                confidences.append(confidence)
                combined_content += f"\n\n{raw_text}\n\n"

                del working_image2, draw2

            elif label.startswith("Table"):
                # Masking seluruh objek kecuali tabel saat ini
                for bbox, table in zip(info["bbox"], info["objects"]):
                    rows = table.extract()
                    if rows:  # Cek apakah ada data hasil ekstraksi
                        combined_content += f"\n\n{label}:\n\n"
                        for row in rows:
                            combined_content += f"{row}\n\n"
                    else:
                        # yield logging_process(
                        #     "warning",
                        #     f"⚠️ Table in File {base_name} on page {page_number + 1} {label} has no data to extract."
                        # )
                        continue
        
        # working_image1.save(f"temp_masked.png")
        raw_text, confidence = extract_text_from_image(working_image1)
        combined_content += f"\n\n{raw_text}\n\n"

        combined_content = clean_text(combined_content)
        avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

        del working_image1, draw1, table_bounding_box, text_bounding_box, combined_data
        gc.collect()

        return combined_content, avg_confidence

    else:
        combined_content = ""
        # Jika tidak ada tabel, hanya ambil teks dari gambar yang sudah dimask
        raw_text, confidence = extract_text_from_image(mask_image)
        combined_content += f"\n\n{raw_text}\n\n"

        combined_content = clean_text(combined_content)
        
        return combined_content, confidence


def process_pdf_pymu_tesseract(pdf_path, folder_output_path, overwrite=True):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = Path(folder_output_path) / f"{base_name}.json"
    # start_ram = psutil.Process().memory_info().rss / 1024**2
    model = YOLO(get_latest_yolo_model_path())

    if not overwrite and check_json_file_exists(output_path):
        yield logging_process(
            "info",
            f"[SKIP] JSON result already exists for {base_name}.pdf, skipping.",
        )
        return
    
    doc = fitz.open(pdf_path)
    output_data = {"content": [], "total_page": doc.page_count}
    os.makedirs(folder_output_path, exist_ok=True)
    total_times = 0

    for page_number in range(len(doc)):
        start_time = time.time()

        yield logging_process(
            "info",
            f"🚀 Starting process for file: {base_name}.pdf\n📄 Processing page {page_number + 1}/{len(doc)} pages"
        )
        content, confidence = extract_pdf_single_page(doc, base_name, model, page_number)

        duration = round(time.time() - start_time, 2)

        output_data["content"].append({
            "page": page_number + 1,
            "content": content,
            "confidence": confidence,
            "duration": duration,
        })

        total_times += duration

        # print(f"📄 Halaman {page_number + 1} | Confidence: {confidence}")
        # print(f"🕒 Durasi: {time.time() - start_time:.2f} detik | RAM: {start_ram:+.2f} MB")
        del content, confidence
        gc.collect()
    
        with open(output_path, "w+", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    output_data["total_time"] = round(total_times, 2)

    with open(output_path, "w+", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    yield logging_process("success", f"Finished processing PDF: {base_name}")
        
    # print(f"\nDurasi total 1 File:{base_name}.pdf {time.time() - start_time:.2f} detik")


# ==== 🛠 Contoh penggunaan ====
# model = YOLO("yolo/best-1.pt")
# folder_output_path = "result"
# list_pdf_path = r"list_pdf_testing/list_pdf_testing.csv"

