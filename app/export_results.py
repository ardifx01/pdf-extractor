from pathlib import Path
import gc
import shutil
import json
import os
import pymupdf
from pymupdf import Page
from ultralytics import YOLO
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    EasyOcrOptions,
    RapidOcrOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from huggingface_hub import snapshot_download

# --- Constants ---
OUTPUT_DIR = Path("app/results")
MODEL_PATH_YOLO = Path("./app/yolo/best-1.pt")
PDF_PATH = Path("app/pdf")
TEMP_IMAGE_DIR = Path("app/temp/image")

# --- Setup ---
model = YOLO(MODEL_PATH_YOLO)

def logging_process(status: str, message: str):
    return {
        "status": status,
        "message": message,
    }

# --- PDF Utilities ---
def yolo_to_pdf_rectangles(boxes, zoom):
    return [
        pymupdf.Rect(
            box[0] // zoom,  # x0
            box[1] // zoom,  # y0
            box[2] // zoom,  # x1
            box[3] // zoom,  # y1
        )
        for box in boxes
    ]


def draw_bounding_boxes(page: Page, rectangles: list[pymupdf.Rect]):
    for rect in rectangles:
        page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()
    return page


def extract_text_from_pdf_page(
    src_path, result_path, create_markdown, number_thread, force_full_page_ocr=False
):
    """Extract text from a PDF page using OCR if necessary.
    
    Args:
        - src_path (str): Path to the source PDF file.
        - result_path (str): Path to save the result.
        - create_markdown (bool): Whether to create a markdown file.
        - number_thread (int): Number of threads to use for OCR.
        - force_full_page_ocr (bool): Whether to force full page OCR. Default is False.
    
    Returns:
        - text (str): Extracted text from the PDF page.
        - doc_conversion_secs (float): Time taken for document conversion.
    """

    accelerator_options = AcceleratorOptions(
        num_threads=number_thread, device=AcceleratorDevice.AUTO
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    settings.debug.profile_pipeline_timings = True

    # pipeline_options.ocr_options = TesseractCliOcrOptions(
    #     lang=["eng", "id"],
    #     force_full_page_ocr=force_full_page_ocr,
    #     tesseract_cmd="tesseract",
    # )

    pipeline_options.ocr_options = EasyOcrOptions(
        lang=["en", "id"],
        force_full_page_ocr=force_full_page_ocr,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )
    conv_result = converter.convert(src_path)
    doc_conversion_secs = conv_result.timings["pipeline_total"].times
    text = conv_result.document.export_to_markdown()

    if len(text) == 0 and force_full_page_ocr is False:
        # If the text is empty, it might be a scanned PDF, so we run OCR again with force_full_page_ocr=True
        return None, doc_conversion_secs

    if create_markdown:
        md_filename = f"{result_path}.md"
        with open(md_filename, "w+", encoding="utf-8") as md_file:
            md_file.write(text)

    return text, doc_conversion_secs


def is_scanned_pdf(pdf_path: str) -> bool:
    with pymupdf.open(pdf_path) as pdf:
        first_page = pdf[0]
        text = first_page.get_text()
        if text.strip():
            return False
        # Check if the first page has any images
        return True if first_page.get_images(full=True) else False


def process_pdf(
    pdf_file: str,
    idx: int = 1,
    create_markdown=False,
    separate_result_dir=False,
    overwrite=True,
    number_thread: int = 4,
):
    OUTPUT_DIR.mkdir(exist_ok=True)
    knowledge_id = Path(pdf_file).stem
    pdf_path = pdf_file

    if separate_result_dir or create_markdown:
        result_dir = OUTPUT_DIR / knowledge_id
        result_dir.mkdir(exist_ok=True)
        json_result_path = result_dir / f"{knowledge_id}.json"
    else:
        result_dir = OUTPUT_DIR
        json_result_path = result_dir / f"{knowledge_id}.json"

    # Read json result if exists
    if json_result_path.exists():
        json_content = json.loads(json_result_path.read_text(encoding="utf-8"))
        total_pages = json_content.get("total_page", 0) or 0
        total_page_extracted = len(json_content.get("content", [])) or 0

        if not overwrite and total_pages == total_page_extracted:
            yield logging_process(
                "info", f"[SKIP] JSON result already exists for {knowledge_id}, skipping."
            )
            return

    try:
        with pymupdf.open(pdf_path) as pdf:
            result_json = {"content": [], "total_page": pdf.page_count}
            temp_image_dir = TEMP_IMAGE_DIR / knowledge_id
            temp_image_dir.mkdir(parents=True, exist_ok=True)
            total = pdf.page_count

            for i, page in enumerate(pdf.pages()):
                page_index = i + 1
                zoom = 3
                mat = pymupdf.Matrix(zoom, zoom)
                page_image = page.get_pixmap(matrix=mat)
                image_path = temp_image_dir / f"{knowledge_id}-page-{page_index}.png"
                page_image.save(str(image_path))

                # YOLO inference
                results = model.predict(str(image_path), verbose=False, conf=0.5)

                result_dict = {
                    "cls": results[0].boxes.cls.cpu().numpy(),
                    "box": results[0].boxes.xyxy.cpu().numpy(),
                }

                # Filter boxes based on class values
                boxes = []
                for i, cls_value in enumerate(result_dict["cls"]):
                    if cls_value == 0:
                        boxes.append(result_dict["box"][i])

                rectangles = yolo_to_pdf_rectangles(boxes, zoom) if boxes else []
                if rectangles:
                    page = draw_bounding_boxes(page, rectangles)

                page_pdf_path = result_dir / f"{knowledge_id}-page-{page_index}.pdf"
                with pymupdf.open() as temp_pdf:
                    temp_pdf.insert_pdf(
                        pdf,
                        from_page=page.number,
                        to_page=page.number,
                        links=False,
                        widgets=False,
                    )
                    temp_pdf.save(str(page_pdf_path), garbage=4, deflate=True)

                # Checking if the PDF is scanned and needs OCR
                markdown_text, time_spent = extract_text_from_pdf_page(
                    page_pdf_path,
                    result_dir / f"{knowledge_id}-page-{page_index}",
                    create_markdown,
                    number_thread,
                )

                if markdown_text is None:
                    yield logging_process(
                        "info",
                        f"Page {page_index}/{pdf.page_count} of {knowledge_id} is empty, running OCR again."
                    )
                    # If the text is empty, it might be a scanned PDF, so we run OCR again with force_full_page_ocr=True
                    markdown_text, time_spent = extract_text_from_pdf_page(
                        page_pdf_path,
                        result_dir / f"{knowledge_id}-page-{page_index}",
                        create_markdown,
                        number_thread,
                        force_full_page_ocr=True,
                    )

                result_json["content"].append(
                    {
                        "page": page_index,
                        "content": markdown_text,
                    }
                )

                yield logging_process(
                    "info",
                    f"Processed page {page_index}/{pdf.page_count} of {knowledge_id} in {time_spent[0]:.2f} seconds."
                )

                del (
                    mat,
                    page_image,
                    results,
                    boxes,
                    rectangles,
                    result_dict,
                    page_pdf_path,
                    markdown_text,
                    time_spent,
                )
                gc.collect()

                with open(json_result_path, "w+", encoding="utf-8") as json_file:
                    json.dump(result_json, json_file, ensure_ascii=False, indent=2)

        # Remove temp PDF files
        for f in result_dir.glob("*.pdf"):
            f.unlink()
        shutil.rmtree(temp_image_dir, ignore_errors=True)

        yield logging_process(
            "info",
            f"Finished processing PDF: {knowledge_id}"
        )

    except Exception as e:
        yield logging_process(
            "error",
            f"Failed to process PDF {idx + 1}/{total}: {e}"
        )
