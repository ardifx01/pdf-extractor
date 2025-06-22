from pathlib import Path
import gc
import shutil
import json
import time
import os
import pymupdf
from pymupdf import Page
from ultralytics import YOLO
import math
import logging

from docling_core.types.doc import PictureItem, TextItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    EasyOcrOptions,
    TesseractCliOcrOptions,
)

from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models

from helper import logging_process, check_json_file_exists

import warnings
from glob import glob
warnings.filterwarnings("ignore")
_log = logging.getLogger(__name__)

# --- Constants ---
OUTPUT_DIR = Path("app/results")
PDF_PATH = Path("app/pdf")
TEMP_IMAGE_DIR = Path("app/temp/image")
ARTIFACT_PATH = Path("app/models")

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

# --- PDF Utilities ---
def yolo_to_pdf_rectangles(boxes, zoom):
    """
    Converts YOLO-format bounding boxes to PyMuPDF rectangle objects, scaling coordinates by the given zoom factor.

    Args:
        boxes (list of list or tuple): A list of bounding boxes, where each box is represented as [x0, y0, x1, y1].
        zoom (float): The zoom factor to scale down the bounding box coordinates.

    Returns:
        List of PyMuPDF Rect objects
    """
    return [
        pymupdf.Rect(
            box[0] // zoom,  # x0
            box[1] // zoom,  # y0
            box[2] // zoom,  # x1
            box[3] // zoom,  # y1
        )
        for box in boxes
    ]

def extract_unique_texts(document):
    seen = set()
    texts = []

    for item, _ in document.iterate_items():
        if isinstance(item, TextItem):
            if item.text not in seen:
                texts.append(item.text)
                seen.add(item.text)

        elif isinstance(item, PictureItem):
            for sub_item, _ in document.iterate_items(
                root=item, traverse_pictures=True
            ):
                if isinstance(sub_item, TextItem) and sub_item.text not in seen:
                    texts.append(sub_item.text)
                    seen.add(sub_item.text)

    return texts

def draw_bounding_boxes(page: Page, rectangles: list[pymupdf.Rect]):
    """
    Draws white bounding boxes on the given PDF page by adding redaction annotations to the specified rectangles and applying the redactions.

    Args:
        page (Page): The PDF page object to draw bounding boxes on.
        rectangles (list[pymupdf.Rect]): A list of rectangle objects specifying the areas to be covered with bounding boxes.

    Returns:
        Page: The modified PDF page with the bounding boxes applied.
    """
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

    logging.basicConfig(level=logging.INFO)
    # Check if the models are already downloaded
    if not os.path.exists(ARTIFACT_PATH):
        download_models(output_dir=ARTIFACT_PATH, progress=True)

    accelerator_options = AcceleratorOptions(
        num_threads=number_thread, device=AcceleratorDevice.AUTO
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.artifacts_path = ARTIFACT_PATH
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_picture_images = True
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
        },
    )
    conv_result = converter.convert(src_path)
    doc_conversion_secs = round(conv_result.timings["pipeline_total"].times[0], 2)
    text = conv_result.document.export_to_markdown(escape_underscores=False)
    
    confidence_data = conv_result.confidence.model_dump()

    if len(text.strip()) == 0 and force_full_page_ocr is False:
        # If the text is empty, it might be a scanned PDF, so we run OCR again with force_full_page_ocr=True
        return None, doc_conversion_secs, confidence_data

    if create_markdown:
        md_filename = f"{result_path}.md"
        with open(md_filename, "w+", encoding="utf-8") as md_file:
            md_file.write(text)

    return text, doc_conversion_secs, confidence_data

def process_pdf(
    pdf_file: str,
    idx: int = 1,
    create_markdown=False,
    overwrite=True,
    exclude_object=True,
    number_thread: int = 4,
    output_dir: str | Path = None,
):
    """
    Process a PDF file, extracting text and optionally creating markdown files.
    Args:
        pdf_file (str): Path to the PDF file to process.
        idx (int): Index of the PDF file in the processing queue.
        create_markdown (bool): Whether to create markdown files from the extracted text.
        overwrite (bool): Whether to overwrite existing JSON results.
        exclude_object (bool): Whether to exclude objects detected by YOLO.
        number_thread (int): Number of threads to use for OCR.
        output_dir (str | Path): Directory to save the output results.
    Yields:
        dict: Status messages indicating the progress of the processing.
    """
    MODEL_YOLO = get_latest_yolo_model_path()
    model = YOLO(MODEL_YOLO)
    base_name = Path(pdf_file).stem
    pdf_path = pdf_file

    if create_markdown:
        result_dir = output_dir / base_name
        result_dir.mkdir(exist_ok=True)
        json_result_path = result_dir / f"{base_name}.json"
    else:
        result_dir = output_dir
        json_result_path = result_dir / f"{base_name}.json"

    if not overwrite and check_json_file_exists(json_result_path):
        yield logging_process(
            "info", f"[SKIP] JSON result already exists for {base_name}, skipping."
        )
        return

    try:
        with pymupdf.open(pdf_path) as pdf:
            result_json = {"content": [], "total_page": pdf.page_count}
            temp_image_dir = TEMP_IMAGE_DIR / base_name
            temp_image_dir.mkdir(parents=True, exist_ok=True)
            total = pdf.page_count
            total_times = 0

            for i, page in enumerate(pdf.pages()):
                page_index = i + 1
                zoom = 3
                mat = pymupdf.Matrix(zoom, zoom)
                page_image = page.get_pixmap(matrix=mat)
                image_path = temp_image_dir / f"{base_name}-page-{page_index}.png"
                page_image.save(str(image_path))
                
                if exclude_object:
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
                    
                    del results, result_dict, boxes, rectangles
                    gc.collect()

                page_pdf_path = result_dir / f"{base_name}-page-{page_index}.pdf"
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
                markdown_text, time_spent, confidence_data = extract_text_from_pdf_page(
                    page_pdf_path,
                    result_dir / f"{base_name}-page-{page_index}",
                    create_markdown,
                    number_thread,
                )
                with pymupdf.open() as temp_pdf:
                    temp_pdf.insert_pdf(
                        pdf,
                        from_page=page.number,
                        to_page=page.number,
                        links=False,
                        widgets=False,
                    )
                    temp_pdf.save(str(page_pdf_path), garbage=4, deflate=True)

                if markdown_text is None:
                    yield logging_process(
                        "info",
                        f"Page {page_index}/{pdf.page_count} of {base_name} is empty, running OCR again."
                    )
                    # If the text is empty, it might be a scanned PDF, so we run OCR again with force_full_page_ocr=True
                    markdown_text, time_spent, confidence_data = extract_text_from_pdf_page(
                        page_pdf_path,
                        result_dir / f"{base_name}-page-{page_index}",
                        create_markdown,
                        number_thread,
                        force_full_page_ocr=True,
                    )

                temp_content = {
                    "page": page_index,
                    "content": markdown_text,
                    "duration": time_spent,
                }

                confidence_data["pages"][0] = {
                    k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in confidence_data["pages"][0].items()
                }
                

                temp_content.update(confidence_data["pages"][0])

                result_json["content"].append(
                    temp_content
                )

                total_times += time_spent

                yield logging_process(
                    "info",
                    f"Processed page {page_index}/{pdf.page_count} of {base_name} in {time.strftime('%H:%M:%S', time.gmtime(time_spent))}"
                )

                del (
                    mat,
                    page_image,
                    page_pdf_path,
                    markdown_text,
                    time_spent,
                )
                gc.collect()

                with open(json_result_path, "w+", encoding="utf-8") as json_file:
                    json.dump(result_json, json_file, ensure_ascii=False, indent=2)
        
            # Save the total time taken for processing the PDF
            result_json["total_time"] = round(total_times, 2)

            with open(json_result_path, "w+", encoding="utf-8") as json_file:
                json.dump(result_json, json_file, ensure_ascii=False, indent=2, allow_nan=False)

        # Remove temp PDF files
        for f in result_dir.glob("*.pdf"):
            f.unlink()
        shutil.rmtree(temp_image_dir, ignore_errors=True)

        yield logging_process(
            "success",
            f"Finished processing PDF: {base_name}"
        )

    except Exception as e:
        yield logging_process(
            "error",
            f"Failed to process PDF {idx + 1}/{total}: {e}"
        )
