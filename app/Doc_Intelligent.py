from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import numpy as np
import re
import base64
import os
import requests
import fitz
from ultralytics import YOLO
from PIL import Image
import time
import json
import gc
from dotenv import load_dotenv
from pathlib import Path

from helper import logging_process, check_json_file_exists

load_dotenv()

PDF_PATH = Path("app/temp/pdf")
OUTPUT_DIR = Path("app/results/doc_intelligent")
TEMP_PDF_DIR = Path("app/temp/processed_pdf")
MODEL_YOLO = YOLO("app/yolo/best-1.pt")

def clean_text(text):
    """Remove extra newlines and strip whitespace from each line."""
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()

def extract_table_contents(table_dict):
    """Extract table contents from Azure Document Intelligence table dict."""
    table = next(iter(table_dict.values()))
    row_count, col_count = table.get("rowCount"), table.get("columnCount")
    table_contents = [["" for _ in range(col_count)] for _ in range(row_count)]
    for cell in table["cells"]:
        row, col, content = cell["rowIndex"], cell["columnIndex"], cell["content"]
        if "columnSpan" in cell:
            span = cell["columnSpan"]
            combined = f"[{content}]"
            for c in range(col, col + span):
                table_contents[row][c] = combined
        else:
            table_contents[row][col] = content
    return table_contents

def download_pdf_if_url(url_pdf, output_folder="temp"):
    """Download PDF if input is a URL, else return the local path."""
    if not url_pdf.startswith("http"):
        return url_pdf
    try:
        os.makedirs(output_folder, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url_pdf, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        filename = url_pdf.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[DOWNLOADED] PDF to local directory: {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] Failed to download PDF: {e}")
        return None

def mask_pdf_with_fitz(local_pdf_path, page_number, bounding_polygons, output_folder="temp", zoom=1.0):
    """Mask table regions in a PDF page using Fitz and save the result."""
    os.makedirs(output_folder, exist_ok=True)
    if not local_pdf_path:
        print("[ERROR] PDF masking cannot be processed.")
        return None
    filename = f"masked_table_page_{page_number}.pdf"
    output_path = os.path.join(output_folder, filename)
    original_doc = fitz.open(local_pdf_path)
    new_doc = fitz.open()
    new_doc.insert_pdf(
        original_doc,
        from_page=page_number - 1,
        to_page=page_number - 1,
        links=False,
        widgets=False,
    )
    new_page = new_doc[-1]
    for poly in bounding_polygons:
        if len(poly) < 8:
            print(f"[SKIP] Polygon coordinates anomaly: {poly}")
            continue
        x_coords, y_coords = poly[0::2], poly[1::2]
        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        dpi = 72
        rect = fitz.Rect(x1 * dpi, y1 * dpi, x2 * dpi, y2 * dpi)
        new_page.add_redact_annot(rect, fill=(1, 1, 1))
    new_page.apply_redactions()
    new_doc.save(output_path)
    new_doc.close()
    original_doc.close()
    print(f"[SAVED] Successfully masked table page and saved to: {output_path}")
    return output_path

def page_to_image(pdf_path, page_number=0, dpi=300):
    """Convert a PDF page to a PIL image."""
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples), zoom

def mask_image_with_yolo(image, page_number, model, output_folder="temp"):
    """Mask non-text objects in an image using YOLO and save as PDF."""
    import cv2  # Local import to avoid issues if cv2 is not installed globally
    os.makedirs(output_folder, exist_ok=True)
    filename = f"masked_page_yolo{page_number}.pdf"
    output_path = os.path.join(output_folder, filename)
    pil_image = image[0] if isinstance(image, tuple) else image
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = model.predict(img_cv, verbose=False)
    label_masking = ["Non-Text"]
    masked_cv = img_cv.copy()
    for r in results:
        if not getattr(r.boxes, "xyxy", None) is not None or len(r.boxes.xyxy) == 0:
            print("[WARNING] No exclude objects detected")
            continue
        try:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box[:4])
                label_id = int(cls)
                label_name = model.names[label_id]
                if label_name in label_masking:
                    cv2.rectangle(masked_cv, (x1, y1), (x2, y2), (255, 255, 255), -1)
        except Exception as e:
            print(f"[ERROR] Error while masking exclude object: {e}")
            continue
    masked_pil = Image.fromarray(cv2.cvtColor(masked_cv, cv2.COLOR_BGR2RGB))
    masked_pil.save(output_path, "PDF", resolution=300.0)
    print(f"[SAVED] Successfully masked YOLO page and saved to: {output_path}")
    return masked_pil, output_path

def get_doc_Intelligent_Result(file, endpoint_azure, key_azure, model="prebuilt-layout"):
    """Get Azure Document Intelligence result for a local file or URL."""
    client = DocumentIntelligenceClient(
        endpoint=endpoint_azure, credential=AzureKeyCredential(key_azure)
    )
    try:
        if isinstance(file, str) and file.startswith("http"):
            request = AnalyzeDocumentRequest(url_source=file)
            poller = client.begin_analyze_document(model, request)
        else:
            with open(file, "rb") as f:
                base64_encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
            content = {"base64Source": base64_encoded_pdf}
            poller = client.begin_analyze_document(model, analyze_request=content)
    except Exception as e:
        print(f"[ERROR] Failed to create poller: {e}")
        return None
    return poller.result()

def extract_text_from_azure_result(result):
    """Extract text from Azure Document Intelligence result."""
    if hasattr(result, "pages") and result.pages:
        return "\n".join(
            line.content for page in result.pages for line in page.lines
        )
    return str(result)

def process_page_with_table(page_number, detected_tables, local_pdf_path, model_yolo, endpoint_azure, key_azure, temp_dir):
    """Process a page containing tables."""
    table_content = [
        table["content"] for table in detected_tables if table["page"] == page_number
    ]
    table_polygon = [
        table["boundingPolygon"] for table in detected_tables if table["page"] == page_number
    ]
    page_text = []
    for tbl in table_content:
        table_str = "\n".join(["\t".join(map(str, row)) for row in tbl])
        page_text.append("\n[Teks di dalam tabel:]\n" + table_str)
    mask_table_pdf_path = mask_pdf_with_fitz(
        local_pdf_path, page_number, table_polygon, output_folder=temp_dir
    )
    mask_table_to_img = page_to_image(mask_table_pdf_path)
    _, output_path_exclude_object = mask_image_with_yolo(
        mask_table_to_img, page_number, model_yolo, output_folder=temp_dir
    )
    text_without_table = get_doc_Intelligent_Result(
        str(output_path_exclude_object), endpoint_azure, key_azure
    )
    ocr_text = extract_text_from_azure_result(text_without_table)
    page_text.append("\n[Teks di luar tabel:]\n" + clean_text(ocr_text))
    return page_text

def process_page_without_table(page, local_pdf_path, model_yolo, endpoint_azure, key_azure, temp_dir):
    """Process a page without tables."""
    page_text = []
    mask_table_to_img = page_to_image(local_pdf_path, page_number=page.page_number - 1)
    _, output_path_exclude_object = mask_image_with_yolo(
        mask_table_to_img, page.page_number, model_yolo, output_folder=temp_dir
    )
    text_without_table = get_doc_Intelligent_Result(
        str(output_path_exclude_object), endpoint_azure, key_azure
    )
    for p in text_without_table.pages:
        for line in p.lines:
            page_text.append(line.content)
    return page_text

def compute_avg_confidence(page):
    """Compute average confidence for a page."""
    all_confidences = [word.confidence for word in page.words]
    return round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0

def transcribe_pdf_with_azureDocIntelligent(
    file: str, model_yolo=MODEL_YOLO, output_dir=OUTPUT_DIR, temp_dir=TEMP_PDF_DIR, overwrite=True
):
    """Transcribe PDF using Azure Document Intelligence and YOLO, yielding progress logs."""
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    file_name = Path(file).stem
    output_folder = output_dir 
    os.makedirs(output_folder, exist_ok=True)
    output_json_path = Path(output_folder) / f"{file_name}.json"
    print(check_json_file_exists(output_json_path))

    if not overwrite and check_json_file_exists(output_json_path):
        print("ini di JSON")
        yield logging_process(
            "info", f"[SKIP] JSON result already exists for {file_name}, skipping."
        )
        return

    try:
        result_json = {
            "content": [],
            "total_page": 0,
            "total_time": 0,
        }
        result = get_doc_Intelligent_Result(file, endpoint, key)
        page_count = len(result.pages)
        print(f"\n\n📘 Starting to read file: {file_name} | Total pages: {page_count} -----")
        detected_tables = []
        if getattr(result, "tables", None):
            for i, table in enumerate(result.tables, 1):
                table_key = f"table_{i}"
                bounding_region = table.get("boundingRegions", [{}])[0]
                page_number = bounding_region.get("pageNumber")
                polygon = bounding_region.get("polygon", [])
                table_dict = {table_key: table}
                content = extract_table_contents(table_dict)
                detected_tables.append(
                    {
                        "page": page_number,
                        "table_id": table_key,
                        "boundingPolygon": polygon,
                        "content": content,
                    }
                )
        result_json["total_pages"] = page_count
        pages_with_tables = {table["page"] for table in detected_tables}
        pages_str = ", ".join(str(page) for page in sorted(pages_with_tables))
        print(f"File contains tables on pages {pages_str}\n")

        local_pdf_path = download_pdf_if_url(file) if isinstance(file, str) and file.startswith("http") else file

        total_page_time = 0
        for page in result.pages:
            print(f"\n---- Processing Page #{page.page_number} ----")
            page_number = page.page_number
            page_start_time = time.time()
            if page_number in pages_with_tables:
                print("\n Page contains a table, using Table conversion algorithm")
                page_text = process_page_with_table(
                    page_number, detected_tables, local_pdf_path, model_yolo, endpoint, key, temp_dir
                )
            else:
                print("\n Page does not contain a table, using Non-Table conversion algorithm")
                page_text = process_page_without_table(
                    page, local_pdf_path, model_yolo, endpoint, key, temp_dir
                )
            full_text = clean_text("\n".join(page_text))
            avg_confidence = compute_avg_confidence(page)
            print(f"📄 Page {page_number} - Average confidence score: {avg_confidence}")

            elapsed_time = time.time() - page_start_time
            total_page_time += elapsed_time
            result_json["content"].append(
                {
                    "page": page_number,
                    "content": full_text,
                    "avg_confident": avg_confidence,
                    "duration": elapsed_time,
                }
            )

            yield logging_process(
                "info",
                f"Processed page {page_number}/{page_count} of {file_name} in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
            )

            del page_text, full_text, avg_confidence
            gc.collect()

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)

        result_json["total_time"] = total_page_time

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        total_minutes, total_seconds = divmod(total_page_time, 60)
        yield logging_process(
            "info",
            f"Total time for all pages: {int(total_minutes)} minutes {total_seconds:.2f} seconds",
        )
        print(
            f"✅ All transcription results for file {file_name} have been saved to: {output_json_path}"
        )

    except Exception as e:
        yield logging_process(
            "error",
            f"An error occurred during transcription: {str(e)}"
        )
        print(f"[ERROR] An error occurred during transcription: {str(e)}")

