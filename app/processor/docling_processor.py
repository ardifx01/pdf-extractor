from pathlib import Path
import time
from typing import Any, Generator, Optional, Union
import logging
import math
import json

import pymupdf
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,  # faster than EasyOCR
    EasyOcrOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models

from app.config import OUTPUT_DIR, ARTIFACT_DIR
from app.processor.yolo_processor import YoloProcessor
from app.utils.helper import (
    ensure_temp_dir,
    log_process,
    check_json_file_exists,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoclingProcessor:
    def __init__(
        self,
        input_path: Union[str, Path],
        create_markdown: bool = False,
        overwrite: bool = False,
        exclude_object=True,
        number_threads: Optional[int] = None,
        ocr_engine: str = "easyocr",
    ):
        self.input_path = Path(input_path)
        self.create_markdown = create_markdown
        self.overwrite = overwrite
        self.exclude_object = exclude_object
        self.number_threads = number_threads
        self.ocr_engine = ocr_engine
        self.output_dir = OUTPUT_DIR / "docling"

    def process_pdf(self) -> Generator[dict[str, str], Any, None]:
        """Process a PDF file and extract text"""
        ensure_temp_dir(self.output_dir)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file {self.input_path} does not exist.")

        model_exclude_object = YoloProcessor()
        base_name = self.input_path.stem
        page_index = 0
        total_pages = 0

        list_models = [f for f in ARTIFACT_DIR.rglob("*") if f.is_file()]
        if not list_models:
            logger.warning(
                "No models found in ARTIFACT_DIR. Downloading default models."
            )
            download_models(progress=True, output_dir=ARTIFACT_DIR)

        if self.create_markdown:
            result_path = self.output_dir / base_name
            result_path.mkdir(parents=True, exist_ok=True)
            json_result_path = result_path / f"{base_name}.json"
        else:
            json_result_path = self.output_dir / f"{base_name}.json"

        if check_json_file_exists(json_result_path) and not self.overwrite:
            logger.info(
                f"JSON result already exists at {json_result_path}. Skipping processing."
            )
            yield log_process(
                "skip",
                f"JSON result already exists for {base_name}. Skipping processing.",
            )
            return

        try:
            with pymupdf.open(self.input_path) as doc:
                result_json = {"content": [], "total_pages": doc.page_count}
                total_pages = doc.page_count
                total_times = 0.0

                for i, page in enumerate(doc.pages()):
                    page_index = i + 1
                    logger.info(
                        f"Processing page {page_index}/{total_pages} of {self.input_path.name}"
                    )

                    if self.exclude_object:
                        page = model_exclude_object.exclude_object(
                            page=page,
                            class_names=0,
                        )

                    page_pdf_path = (
                        self.output_dir / f"{base_name}_page_{page_index}.pdf"
                    )
                    with pymupdf.open() as temp_pdf:
                        temp_pdf.insert_pdf(
                            doc,
                            from_page=page.number, 
                            to_page=page.number,
                            links=False,
                            widgets=False,
                        )
                        temp_pdf.save(page_pdf_path, garbage=4, deflate=True)

                    result_text, conversion_time, confidence = self._extract_text(file_path=page_pdf_path, ocr_engine=self.ocr_engine)

                    if result_text is None:
                        logger.warning(
                            f"No text extracted from page {page_index} of {self.input_path.name}. Retrying with OCR."
                        )
                        yield log_process(
                            "info",
                            f"No text extracted from page {page_index} of {self.input_path.name}. Retrying with OCR.",
                        )
                        result_text, conversion_time, confidence = self._extract_text(
                            file_path=page_pdf_path,
                            ocr_engine=self.ocr_engine,
                            force_ocr=True
                        )

                    total_times += conversion_time

                    temp_json = {
                        "page": page_index,
                        "content": result_text,
                        "duration": conversion_time,
                    }
                    confidence["pages"][0] = {
                        k: (None if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in confidence["pages"][0].items()
                    }
                    temp_json.update(confidence["pages"][0])
                    result_json["content"].append(temp_json)

                    yield log_process(
                        "info",
                        f"Processed page {page_index}/{doc.page_count} of {base_name} in {time.strftime('%H:%M:%S', time.gmtime(conversion_time))}.",
                    )

                    with open(json_result_path, "w+", encoding="utf-8") as json_file:
                        json.dump(
                            result_json,
                            json_file,
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        )

                result_json["total_time"] = round(total_times, 2)

                with open(json_result_path, "w+", encoding="utf-8") as json_file:
                    json.dump(
                        result_json,
                        json_file,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )

            # Clean up temporary files
            for f in self.output_dir.glob("*.pdf"):
                if f.is_file():
                    f.unlink(missing_ok=True)

            yield log_process(
                "success", f"Processed {base_name} with {total_pages} pages."
            )

        except Exception as e:
            logger.error(f"Error processing {self.input_path.name}: {e}")
            yield log_process(
                "error", f"Failed to process {self.input_path.name}: {str(e)}"
            )
            raise e

    def _extract_text(
        self, file_path: Path, ocr_engine: str, force_ocr: bool = False,
    ):
        """Extract text from the PDF document."""
        accelerator_options = AcceleratorOptions(
            num_threads=4 if self.number_threads is None else self.number_threads,
            device=AcceleratorDevice.AUTO,
        )
        pipeline_options = PdfPipelineOptions()
        pipeline_options.artifacts_path = ARTIFACT_DIR
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.images_scale = 2.0

        settings.debug.profile_pipeline_timings = True

        match ocr_engine.lower():
            case "easyocr":
                pipeline_options.ocr_options = EasyOcrOptions(
                    lang=["en", "id"],
                    force_full_page_ocr=force_ocr,
                    download_enabled=True,
                )
            case "tesseract":
                pipeline_options.ocr_options = TesseractCliOcrOptions(
                    lang=["eng", "ind"],
                    force_full_page_ocr=force_ocr,
                    tesseract_cmd="tesseract",
                )

        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            },
        )

        converter_result = converter.convert(file_path)
        text = converter_result.document.export_to_markdown(escape_underscores=False)
        conversion_time = round(converter_result.timings["pipeline_total"].times[0], 2)

        confidence = converter_result.confidence.model_dump()

        if len(text.strip()) == 0 and not force_ocr:
            logger.warning(
                f"No text extracted from {file_path.name}. Retrying with OCR."
            )
            return None, conversion_time, confidence

        if self.create_markdown:
            markdown_path = self.output_dir / self.input_path.stem / f"{file_path.stem}.md"
            with open(markdown_path, "w+", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Markdown file created at {markdown_path}")

        return text, conversion_time, confidence


if __name__ == "__main__":
    processor = DoclingProcessor("sample-layout.pdf", overwrite=True, create_markdown=True)
    for result in processor.process_pdf():
        print(result)