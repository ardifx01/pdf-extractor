import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import os
import shutil
import torch
import pymupdf
import json
import time
import re
from pathlib import Path
from glob import glob
import uuid
import pyperclip
from datetime import datetime, timedelta
import atexit

from pdf_process import (
    handle_pdf_download_from_dataset,
    read_dataset,
    ensure_temp_dir,
    clear_temp_dir,
    download_pdf,
    TEMP_DIR,
    TEMP_DIR_PDF,
)
from export_results import (
    process_pdf,
    OUTPUT_DIR,
)

from Pymu_Tesseract_Finetuned import (
    process_pdf_pymu_tesseract,
    FOLDER_OUTPUT_PYMU_TESSERACT,
)

from Doc_Intelligent import (
    transcribe_pdf_with_azureDocIntelligent
)

from transcribe_audio import (
    VIDEO_PATH,
    transcribe_audio,
)

# Constants
EXTENSION = {
    "csv": [".csv"],
    "xlsx": [".xlsx"],
    "pdf": [".pdf"],
    "wav": [".wav"],
}

# Fix torch path handling
torch.classes.__path__ = []


DATA_TEMP = Path("app/temp/data")
os.makedirs(DATA_TEMP, exist_ok=True)


# Initialize session state variables
def init_session_state():
    if "export_ready" not in st.session_state:
        st.session_state["export_ready"] = False
    if "zip_path" not in st.session_state:
        st.session_state["zip_path"] = None
    if "show_confirm_dialog" not in st.session_state:
        st.session_state["show_confirm_dialog"] = False
    if "process_file_clicked" not in st.session_state:
        st.session_state["process_file_clicked"] = False
    if "cancel_processing" not in st.session_state:
        st.session_state["cancel_processing"] = False
    if "data_temp" not in st.session_state:
        st.session_state["data_temp"] = DATA_TEMP
    if "already_exported" not in st.session_state:
        st.session_state["already_exported"] = False
    if "error_archive" not in st.session_state:
        st.session_state["error_archive"] = False
    if "method_option" not in st.session_state:
        st.session_state["method_option"] = None
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = str(uuid.uuid4())
    if "temp_file_path" not in st.session_state:
        st.session_state["temp_file_path"] = None
    if "already_copied" not in st.session_state:
        st.session_state["already_copied"] = False
    if "selected_file" not in st.session_state:
        st.session_state["selected_file"] = None
    if "uploaded_files_meta" not in st.session_state:
        st.session_state["uploaded_files_meta"] = {}


# Configure page
def setup_page():
    st.set_page_config(page_title="PDF Processing Dashboard", layout="wide")
    st.title("File Conversion & Audio Transcription")


# Utility functions
def zip_for_download(progress_callback=None):
    target_folder = OUTPUT_DIR
    base_name = "exported_results"

    # Show progress
    if progress_callback:
        progress_callback(0.2)  # 20% before zipping

    zip_path = shutil.make_archive(
        base_name=base_name,
        format="zip",
        root_dir=target_folder,
    )

    if progress_callback:
        progress_callback(0.7)  # 70% after zip creation

    return zip_path


def prepare_export():
    progress_bar = st.sidebar.progress(0, text="Preparing export...")

    def update_progress(val):
        progress_bar.progress(val, text="Preparing export...")

    try:
        st.session_state["zip_path"] = zip_for_download(
            progress_callback=update_progress
        )
    except Exception:
        st.session_state["error_archive"] = True
        return

    time.sleep(0.5)  # Small delay simulation
    update_progress(0.9)
    clear_temp_dir(OUTPUT_DIR)
    clear_temp_dir(TEMP_DIR)

    update_progress(1.0)
    time.sleep(0.5)

    st.session_state["export_ready"] = True
    progress_bar.empty()


def has_extracted_data(output_dir: str | Path, export_to_markdown: bool):
    if not os.path.exists(output_dir):
        return False

    if st.session_state["method_option"] == "Docling":
        output_dir = os.path.join(output_dir, "docling_results")
    elif st.session_state["method_option"] == "PyMuPDF + Tesseract":
        output_dir = FOLDER_OUTPUT_PYMU_TESSERACT
    elif st.session_state["method_option"] == "Azure Doc Intelligence":
        output_dir = os.path.join(output_dir, "doc_intelligent")
    elif st.session_state["method_option"] == "Whisper AI":
        output_dir = OUTPUT_DIR / "transcribed_audio"

    if export_to_markdown:
        # Use glob for efficient file matching
        extracted_files = glob(
            pathname="**/*.json", root_dir=output_dir, recursive=True
        )
    # elif st.session_state["method_option"] == "PyMuPDF + Tesseract" or st.session_state["method_option"] == "Azure Doc Intelligence":
    #     extracted_files = glob(
    #         pathname="**/*.json", root_dir=output_dir, recursive=True
    #     )
    else:
        extracted_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    return False if len(extracted_files) > 0 else True


def process_file_clicked():
    st.session_state["process_file_clicked"] = True


def cancel_processing():
    st.session_state["cancel_processing"] = True
    st.session_state["process_file_clicked"] = False


def toast_upload_success():
    st.toast(
        "File uploaded successfully!",
        icon=":material/done_outline:",
    )


# UI Components
def render_sidebar():
    st.sidebar.title("Options")
    st.sidebar.write("Select options to process PDF files. Default export JSON")

    # Processing options
    number_thread = st.sidebar.number_input(
        "Number of Threads",
        min_value=1,
        max_value=64,
        value=4,
        step=1,
        key="number_thread",
    )

    export_to_markdown = st.sidebar.checkbox("Export to Markdown", value=False)
    overwrite = st.sidebar.toggle(
        "Overwrite existing files",
        value=False,
        help="Overwrite existing files if they exist.",
        key="overwrite",
    )

    # Selector tab for Dataset Upload or Download from URL
    tab1, tab2 = st.sidebar.tabs(["Upload Dataset", "Download PDF from URL"])

    with tab1:
        # Dataset handling
        dataset_files = st.file_uploader(
            "Upload Dataset (CSV/Excel/PDF/Video)",
            type=["csv", "xlsx", "pdf", "mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True,
            key=st.session_state["file_uploader_key"],
            on_change=toast_upload_success,
        )

    with tab2:
        url_pdf_file = st.text_input(
            "Download PDF from URL",
            placeholder="Enter PDF URL here",
            key="pdf_url_input",
            help="Enter a valid URL to download a PDF file.",
        )

        if url_pdf_file:
            results = download_pdf(url=url_pdf_file)

            for result in results:
                if result.get("status") == "success":
                    st.toast(
                        result.get("message", "PDF downloaded successfully."),
                        icon=":material/done_outline:",
                    )

    df = None
    column_list = []
    id_col = None
    url_col = None

    if dataset_files:
        for dataset_file in dataset_files:
            if re.search(r"\.(csv|xlsx)$", dataset_file.name, re.IGNORECASE):
                temp_file_path = DATA_TEMP / dataset_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(dataset_file.getbuffer())

                df = read_dataset(temp_file_path)
                column_list = df.columns.tolist()

                st.session_state["uploaded_files_meta"][str(dataset_file.name)] = {
                    "extracted_at": datetime.now().isoformat(),
                }

            elif dataset_file.name.endswith(".pdf"):
                # Save directly to TEMP_DIR_PDF
                ensure_temp_dir(TEMP_DIR_PDF)
                temp_file_path = TEMP_DIR_PDF / dataset_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(dataset_file.getbuffer())
            
            elif dataset_file.name.endswith((".mp4", ".avi", ".mov", ".mkv")):
                # Save directly to VIDEO_PATH
                ensure_temp_dir(VIDEO_PATH)
                temp_file_path = VIDEO_PATH / dataset_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(dataset_file.getbuffer())
        st.session_state["file_uploader_key"] = str(uuid.uuid4())
        st.rerun()

    # Combined sidebar for existing dataset and PDF files
    existing_files = list(DATA_TEMP.glob("*.csv")) + list(DATA_TEMP.glob("*.xlsx"))
    existing_pdfs = list(TEMP_DIR_PDF.glob("*.pdf"))

    if existing_files or existing_pdfs:
        if existing_files:
            st.sidebar.info("Existing dataset files found in temp directory")
            selected_file = st.sidebar.selectbox(
                "Select a dataset file to preview",
                options=[file.name for file in existing_files],
                help="Select a file from the existing dataset files in DATA_TEMP.",
                key="existing_dataset_file",
            )
            if selected_file:
                selected_file_path = DATA_TEMP / selected_file
                df = read_dataset(selected_file_path)
                column_list = df.columns.tolist()
                st.session_state["temp_file_path"] = selected_file_path
        else:
            st.sidebar.warning(
                "No dataset file uploaded or found. Please upload a CSV or Excel file."
            )
            df = None
            column_list = []

        if existing_pdfs:
            st.sidebar.info("Existing PDF files found in temp directory")
        else:
            st.sidebar.warning(
                "No PDF files uploaded or found. Please upload or download PDFs first."
            )
    else:
        st.sidebar.warning(
            "No dataset or PDF files found. Please upload a CSV/Excel dataset and/or PDF files."
        )
        df = None
        column_list = []

    use_specific_id = False
    # Column selection
    if df is not None and column_list:
        url_col = st.sidebar.selectbox("Select URL Column", options=column_list)
        with st.sidebar.expander(f"Sample {url_col}", expanded=False):
            st.write(f"Sample: {df.loc[0, url_col]}")

        use_specific_id = st.sidebar.checkbox(
            "Use Specific ID Column",
            value=False,
            help="Select a specific ID column to use for processing.",
        )
        if use_specific_id:
            id_col = st.sidebar.selectbox("Select ID Column", options=column_list)
            with st.sidebar.expander(f"Sample {id_col}", expanded=False):
                st.write(f"Sample: {df.loc[0, id_col]}")
    else:
        id_col = None
        url_col = None

    # Action buttons
    download_col, clear_temp_col = st.sidebar.columns(2, vertical_alignment="bottom")

    with download_col:
        download_button = st.button(
            "Download",
            type="primary",
            key="download_pdfs",
            icon=":material/download:",
            help="Download PDFs from the dataset.",
        )

    with clear_temp_col:
        clear_temp_button = st.button(
            "Temp Files",
            icon=":material/delete:",
            help="Clear all temporary files, including PDFs and results.",
        )
        if clear_temp_button:
            st.session_state["file_uploader_key"] = str(uuid.uuid4())
            clear_temp_dir(TEMP_DIR)
            clear_temp_dir(OUTPUT_DIR)
            st.toast("Temporary files cleared successfully.")
            st.rerun()

    # Export button
    ensure_temp_dir(OUTPUT_DIR / "docling_results")
    ensure_temp_dir(VIDEO_PATH)
    export_disabled = has_extracted_data(OUTPUT_DIR, export_to_markdown)
    export_btn = st.sidebar.button(
        label="Export",
        disabled=export_disabled,
        help="No extracted files yet!"
        if export_disabled
        else "Export all extracted results as ZIP",
        icon=":material/publish:",
        use_container_width=True,
        key="export_btn",
    )

    if export_btn and not export_disabled:
        prepare_export()
        st.session_state["show_confirm_dialog"] = True

    return (
        df,
        id_col,
        url_col,
        download_button,
        clear_temp_button,
        export_to_markdown,
        number_thread,
        overwrite,
        use_specific_id,
    )


@st.dialog("Warning: Export will delete all results ⚠ ")
def confirmation_delete():
    st.warning(
        "You are about to export and download all processed results as a ZIP file. "
        "This action will also delete all existing results from the output directory. "
        "Are you sure you want to continue?"
    )
    st.markdown(
        """
        - **Download Exported Results**: Downloads all processed files as a ZIP archive.
        - **Cancel**: Aborts the export and leaves your results untouched.
        - After export, all result files in the output directory will be deleted to free up space.
        """
    )
    if st.session_state["error_archive"]:
        st.error("Error during export preparation")
        st.info("Maybe you haven't started extraction yet?")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state["export_ready"] and st.session_state["zip_path"]:
            with open(st.session_state["zip_path"], "rb") as f:
                zip_bytes = f.read()

            if st.download_button(
                label="Download Exported Results",
                data=zip_bytes,
                file_name="exported_results.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_zip",
                type="primary",
                icon=":material/download:",
                help="After downloading, all exported results will be deleted.",
            ):
                st.session_state["show_confirm_dialog"] = False
                st.session_state["already_exported"] = True
                os.remove(st.session_state["zip_path"])
                del st.session_state["zip_path"]
                st.rerun()
    with col2:
        if st.button("Cancel"):
            st.session_state["show_confirm_dialog"] = False
            st.session_state["cancelled_export"] = True
            st.rerun()

def handle_download_pdfs(file_path, df, id_col, url_col, use_specific_id):
    if df is not None and id_col != url_col:
        total_pdf_files = df[url_col].nunique()
        total_processing = 0
        total_success = 0
        failed_files = []

        with st.status("Downloading PDFs...", expanded=True) as status:
            results = handle_pdf_download_from_dataset(
                file_path, id_col, url_col, use_specific_id
            )
            download_slot = st.empty()

            for result in results:
                total_processing += 1
                status.update(
                    label=f"Downloading ({total_processing}/{total_pdf_files}) PDFs..."
                )

                if result.get("status") == "success":
                    total_success += 1
                    download_slot.success(result.get("message", "Download succeeded."))
                elif result.get("status") == "info":
                    total_success += 1
                    download_slot.info(result.get("message", "Download skipped."))
                else:
                    failed_files.append(result)
                    download_slot.error(result.get("message", "Download failed."))

            # Retry failed files
            if len(failed_files) > 0:
                st.info(f"Retrying {len(failed_files)} failed downloads...")
                retry_results = []

                for fail in failed_files:
                    row_id = fail.get("id")
                    url = fail.get("url")

                    if row_id is not None and url is not None:
                        retry_df = df[(df[id_col] == row_id) & (df[url_col] == url)]

                        if not retry_df.empty:
                            retry_result = handle_pdf_download_from_dataset(
                                retry_df, id_col, url_col
                            )

                            for r in retry_result:
                                if r.get("status") in ["success", "info"]:
                                    total_success += 1
                                    status_type = (
                                        "success"
                                        if r.get("status") == "success"
                                        else "info"
                                    )
                                    getattr(download_slot, status_type)(
                                        f"Retry: {r.get('message', 'Download succeeded.' if status_type == 'success' else 'Download skipped.')}"
                                    )
                                else:
                                    download_slot.error(
                                        f"Retry: {r.get('message', 'Download failed.')}"
                                    )
                                retry_results.append(r)

            status.update(
                label=f"PDF download completed. Total success {total_success}/{total_pdf_files}",
                expanded=False,
            )
    else:
        st.sidebar.error("Please upload a dataset and select ID and URL columns.")


def handle_pdf_processing(export_to_markdown, number_thread, overwrite):
    ensure_temp_dir(TEMP_DIR_PDF)
    pdf_files = os.listdir(TEMP_DIR_PDF)
    audio_files = os.listdir(VIDEO_PATH)
    
    # Combine PDF and audio files for processing
    files = pdf_files + audio_files

    if not files:
        st.warning(
            "No PDF or audio files found in the temporary directories. Please upload or download files first."
        )
        return None

    process_all_files, process_single_file, exclude_object, method_options = st.columns(
        4, gap="small", vertical_alignment="bottom"
    )
    method = ["Docling", "PyMuPDF + Tesseract", "Azure Doc Intelligence", "Whisper AI"]

    method_option_select = method_options.selectbox(
        "Select Processing Method",
        options=method,
        index=1,
        key="method_option",
    )

    ensure_temp_dir(
        [
            OUTPUT_DIR / "docling_results",
            FOLDER_OUTPUT_PYMU_TESSERACT,
            OUTPUT_DIR / "doc_intelligent",
            OUTPUT_DIR / "transcribed_audio",
        ]
    )

    process_file_btn = process_all_files.button(
        "Process File",
        key="process_file",
        disabled=False if files else True,
        on_click=process_file_clicked,
    )

    extract_current_file = process_single_file.toggle(
        "Process Selected File",
        key="process_current_file",
        disabled=False if files else True,
        help="Process only the currently selected file.",
    )

    exclude_object_value = exclude_object.toggle(
        "Object Detection",
        value=True,
        help="Use object detection during processing.",
    )

    if process_file_btn:
        st.session_state["cancel_processing"] = (
            False  # Uncommented to enable processing
        )

        # Stop button
        stop_button_disabled = st.session_state["cancel_processing"]

        if st.session_state["process_file_clicked"]:
            st.session_state["cancel_processing"] = False
            st.session_state["process_file_clicked"] = False

            stop_button = st.button(
                "Stop", on_click=cancel_processing, disabled=stop_button_disabled
            )

        total_files = len(files)
        file_status = st.empty()
        total_success = 0
        total_failed = 0
        total_skipped = 0

        if extract_current_file:
            files = [st.session_state["selected_file"]]
            total_files = 1

        with st.status(
            f"Processing Files to {'Markdown and JSON' if export_to_markdown else 'JSON'} files...",
            expanded=True,
        ) as status:
            for idx, file_name in enumerate(files, 1):
                if st.session_state["cancel_processing"]:
                    status.warning("Processing canceled by user.")
                    break

                file_status.info(f"Processing: {file_name}")
                page_processing_slot_status = st.empty()

                if method_option_select == "Docling":
                    output_dir = OUTPUT_DIR / "docling_results"

                    for log in process_pdf(
                        os.path.join(TEMP_DIR_PDF, file_name),
                        create_markdown=export_to_markdown,
                        overwrite=overwrite,
                        exclude_object=exclude_object_value,
                        number_thread=number_thread,
                        output_dir=output_dir,
                    ):
                        if log.get("status") == "info":
                            msg = log.get("message", "SKIP")
                            if "[SKIP]" in msg:
                                total_success += 1
                                total_skipped += 1
                            page_processing_slot_status.info(
                                log.get("message", "Processing skipped.")
                            )
                        elif log.get("status") == "success":
                            total_success += 1
                            file_status.success(
                                log.get("message", "Processing succeeded.")
                            )
                        elif log.get("status") == "error":
                            total_failed += 1
                            st.write(log.get("message", "Processing failed."))
                        elif log.get("status") == "ocr_active":
                            page_processing_slot_status.info(
                                log.get("message", "OCR is active.")
                            )
                        else:
                            st.write(log.get("message", "Processing failed."))

                    status.update(
                        label=f"Processing: {idx}/{total_files} Files | Success {total_success} | Skipped {total_skipped} | Failed {total_failed}"
                    )

                    st.session_state["process_file_clicked"] = False
                    page_processing_slot_status.empty()

                if method_option_select == "PyMuPDF + Tesseract":
                    for log in process_pdf_pymu_tesseract(
                        os.path.join(TEMP_DIR_PDF, file_name),
                        folder_output_path=FOLDER_OUTPUT_PYMU_TESSERACT,
                        overwrite=overwrite,
                    ):
                        if log.get("status") == "info":
                            msg = log.get("message", "SKIP")
                            if "[SKIP]" in msg:
                                total_success += 1
                                total_skipped += 1
                            page_processing_slot_status.info(
                                log.get("message", "Processing skipped.")
                            )
                        elif log.get("status") == "success":
                            total_success += 1
                            file_status.success(
                                log.get("message", "Processing succeeded.")
                            )
                        elif log.get("status") == "error":
                            total_failed += 1
                            st.write(log.get("message", "Processing failed."))
                        elif log.get("status") == "ocr_active":
                            page_processing_slot_status.info(
                                log.get("message", "OCR is active.")
                            )
                        else:
                            st.write(log.get("message", "Processing failed."))

                    status.update(
                        label=f"Processing: {idx}/{total_files} Files | Success {total_success} | Skipped {total_skipped} | Failed {total_failed}"
                    )

                    st.session_state["process_file_clicked"] = False
                    page_processing_slot_status.empty()

                # Need Attention: Penyesuaian untuk Azure Doc Intelligence
                if method_option_select == "Azure Doc Intelligence":
                    for log in transcribe_pdf_with_azureDocIntelligent(
                        os.path.join(TEMP_DIR_PDF, file_name),
                        overwrite=overwrite,
                    ):
                        if log.get("status") == "info":
                            msg = log.get("message", "SKIP")
                            if "[SKIP]" in msg:
                                total_success += 1
                                total_skipped += 1
                            page_processing_slot_status.info(
                                log.get("message", "Processing skipped.")
                            )
                        elif log.get("status") == "success":
                            total_success += 1
                            file_status.success(
                                log.get("message", "Processing succeeded.")
                            )
                        elif log.get("status") == "error":
                            total_failed += 1
                            st.write(log.get("message", "Processing failed."))
                        else:
                            st.write(log.get("message", "Processing failed."))

                    status.update(
                        label=f"Processing: {idx}/{total_files} Files | Success {total_success} | Skipped {total_skipped} | Failed {total_failed}"
                    )

                    st.session_state["process_file_clicked"] = False
                    page_processing_slot_status.empty()

                if method_option_select == "Whisper AI":
                    for log in transcribe_audio(
                        os.path.join(VIDEO_PATH, file_name),
                        overwrite=overwrite,
                    ):
                        if log.get("status") == "info":
                            msg = log.get("message", "SKIP")
                            if "[SKIP]" in msg:
                                total_success += 1
                                total_skipped += 1
                            page_processing_slot_status.info(
                                log.get("message", "Processing skipped.")
                            )
                        elif log.get("status") == "success":
                            total_success += 1
                            file_status.success(
                                log.get("message", "Processing succeeded.")
                            )
                        elif log.get("status") == "error":
                            total_failed += 1
                            st.write(log.get("message", "Processing failed."))
                        else:
                            st.write(log.get("message", "Processing failed."))

                    status.update(
                        label=f"Processing: {idx}/{total_files} Videos | Success {total_success} | Skipped {total_skipped} | Failed {total_failed}"
                    )

                    st.session_state["process_file_clicked"] = False
                    page_processing_slot_status.empty()

                st.session_state["uploaded_files_meta"][str(file_name)] = {
                    "extracted_at": datetime.now().isoformat(),
                }

            file_status.empty()

            if not st.session_state["cancel_processing"]:
                status.success(
                    f"Files converted to {'Markdown and JSON' if export_to_markdown else 'JSON'} files. Total Success: {total_success}, Skipped {total_skipped}, Failed: {total_failed}"
                )
                st.rerun()

    return files


def clean_old_files(max_age_minutes=30):
    """Clean up old files and provide warnings before deletion."""
    now = datetime.now()
    expired_files = []

    for file_path_str, meta in st.session_state["uploaded_files_meta"].items():
        file_path = Path(file_path_str)
        if "extracted_at" not in meta:
            continue

        uploaded_time = datetime.fromisoformat(meta["extracted_at"])
        age = now - uploaded_time
        remaining = timedelta(minutes=max_age_minutes) - age
        remaining_minutes = int(remaining.total_seconds() // 60)

        # Add warning flags if they don't exist
        if "warned_at_10" not in meta:
            meta["warned_at_10"] = False
        if "warned_at_5" not in meta:
            meta["warned_at_5"] = False

        # Show warnings at specific thresholds
        if remaining_minutes <= 10 and not meta["warned_at_10"]:
            st.toast(f"⚠️ {file_path.name} will be deleted in 10 minutes!", icon="⏳")
            meta["warned_at_10"] = True

        elif remaining_minutes <= 5 and not meta["warned_at_5"]:
            st.toast(f"🚨 {file_path.name} only 5 minutes until deletion!", icon="⏳")
            meta["warned_at_5"] = True

        # Delete expired files
        if age > timedelta(minutes=max_age_minutes):
            try:
                if os.path.exists(os.path.join(TEMP_DIR_PDF, file_path)):
                    os.remove(os.path.join(TEMP_DIR_PDF, file_path))
                    st.toast(
                        f"🧹 {file_path.name} has been automatically deleted", icon="✅"
                    )
                    expired_files.append(file_path_str)

                if os.path.exists(os.path.join(DATA_TEMP, file_path)):
                    os.remove(os.path.join(DATA_TEMP, file_path))
                    st.toast(
                        f"🧹 {file_path.name} has been automatically deleted", icon="✅"
                    )
                    expired_files.append(file_path_str)
            except Exception as e:
                st.warning(f"Failed to delete {file_path}: {e}")

    # Remove deleted files from tracking
    for expired in expired_files:
        if expired in st.session_state["uploaded_files_meta"]:
            del st.session_state["uploaded_files_meta"][expired]


def render_preview_file(files, export_to_markdown):
    if not files or len(files) == 0:
        st.info("No files available for preview.")
        return

    def update_selected_file(file):
        if file:
            st.session_state["selected_file"] = file

    query_file = st.selectbox(
        "Select File to preview",
        options=files,
        index=files.index(st.session_state["selected_file"])
        if st.session_state["selected_file"] in files
        else 0,
        placeholder="Select a file",
        help="Select a file to preview.",
        on_change=update_selected_file,
        args=(st.session_state["selected_file"],),
    )

    # Store the selected file in session state
    st.session_state["selected_file"] = query_file

    if not query_file:
        st.info("Please select a file to preview.")
        return

    query_file = Path(query_file)
    file_ext = query_file.suffix.lower()

    # Determine file type and set up preview logic
    if file_ext == ".pdf":
        file_path = TEMP_DIR_PDF / query_file
        try:
            doc = pymupdf.open(file_path)
        except Exception:
            st.error("Failed to open the PDF document.")
            return
        if doc.page_count == 0:
            st.error("The PDF document is empty.")
            return
        max_page = doc.page_count
        is_pdf = True
        is_video = False
    elif file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
        file_path = VIDEO_PATH / query_file
        doc = None
        # For videos, you might want to use a video player component
        max_page = 1
        is_pdf = False
        is_video = True
    else:
        st.error("Unsupported file type for preview.")
        return

    # Page/video selector
    page_number = st.number_input(
        "Select Page Number" if is_pdf else "Select Video",
        min_value=1,
        max_value=max_page,
        value=1,
        step=1,
        key="page_number",
    )
    st.write(f"{'Page' if is_pdf else 'Video'} {page_number} of {max_page}")

    preview, result = st.columns(2, border=True)

    with preview:
        if is_pdf:
            st.write("Selected PDF:")
            pdf_viewer(
                file_path,
                width=900,
                height=700,
                pages_to_render=[page_number],
                key=f"pdf_viewer_{query_file.stem}_{page_number}",
            )
        elif is_video:
            st.write("Selected Video:")
            st.video(str(file_path))

    with result:
        st.write("Result:")

        file_name = query_file.stem

        # Determine output/result path
        if st.session_state["method_option"] == "Docling":
            base_path = os.path.join(OUTPUT_DIR, "docling_results")
        elif st.session_state["method_option"] == "PyMuPDF + Tesseract":
            base_path = FOLDER_OUTPUT_PYMU_TESSERACT
        elif st.session_state["method_option"] == "Azure Doc Intelligence":
            base_path = os.path.join(OUTPUT_DIR, "doc_intelligent")
        elif st.session_state["method_option"] == "Whisper AI":
            base_path = os.path.join(OUTPUT_DIR, "transcribed_audio")
        else:
            base_path = OUTPUT_DIR

        # Result file path logic
        if export_to_markdown and st.session_state["method_option"] == "Docling":
            result_path = os.path.join(base_path, file_name, file_name + ".json")
        else:
            result_path = os.path.join(base_path, file_name + ".json")

        if is_pdf and st.session_state["method_option"] == "Whisper AI":
            st.error(
                "Whisper AI is not applicable for PDF files. Please select another method."
            )
            return
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                json_result = json.load(f)
                total_duration = json_result.get("total_time", 0)
                content = json_result.get("content", [])
                st.json(json_result, expanded=False)

                # For PDF: show per-page markdown; for video: show transcript
                if is_pdf and 0 <= page_number - 1 < len(content):
                    selected_page = content[page_number - 1]["content"]
                    dur_per_page = content[page_number - 1].get("duration", 0)
                    parse_score = content[page_number - 1].get("parse_score", 0)
                    layout_score = content[page_number - 1].get("layout_score", 0)
                    table_score = content[page_number - 1].get("table_score", 0) or 0
                    ocr_score = content[page_number - 1].get("ocr_score", 0) or 0

                    raw_md_button = st.button(
                        "Copy Raw Markdown",
                        key=f"raw_md_{file_name}_{page_number}",
                        help="Click to copy raw markdown content.",
                    )
                    if raw_md_button:
                        pyperclip.copy(selected_page)
                        st.session_state["already_copied"] = True
                        st.rerun()

                    with st.expander("Processing Details", expanded=False):
                        st.markdown(
                            f"""
                            - **Total Duration**: {total_duration:.2f} seconds
                            - **Time for Page {page_number}**: {dur_per_page:.2f} seconds
                            - **Parse Score**: {parse_score:.4f}
                            - **Layout Score**: {layout_score:.4f}
                            - **Table Score**: {table_score:.4f}
                            - **OCR Score**: {ocr_score:.4f}
                            """
                        )
                    st.markdown("### Conversion")
                    with st.container(key="markdown_result", height=400):
                        st.write(selected_page, unsafe_allow_html=True)
                elif is_video:
                    # For video, show transcript or content
                    if content:
                        st.markdown("### Conversion")
                        with st.container(key="video_transcript", height=400):
                            if isinstance(content, str):
                                st.markdown(content)
                            else:
                                for idx, c in enumerate(content):
                                    st.text_area(
                                        f"Transcript Segment {idx+1}",
                                        c.get("content", ""),
                                        height=100,
                                        key=f"video_transcript_{query_file.stem}_{idx}"
                                    )
                    else:
                        st.info("No transcript found for this video.")
                else:
                    st.info("No markdown content for this page.")
        else:
            st.info("No result JSON found for this file.")


def main():
    # Initialize
    init_session_state()
    setup_page()

    # Sidebar and settings
    (
        df,
        id_col,
        url_col,
        download_button,
        clear_temp_button,
        export_to_markdown,
        number_thread,
        overwrite,
        use_specific_id,
    ) = render_sidebar()

    # Handle export confirmation dialog
    if st.session_state.get("show_confirm_dialog", False):
        confirmation_delete()

    if st.session_state.get("cancelled_export"):
        st.toast("Export cancelled!", icon="❌")
        st.session_state["cancelled_export"] = False

    if st.session_state.get("already_exported"):
        st.toast("Export completed!", icon="✅")
        st.session_state["already_exported"] = False

    if st.session_state.get("already_copied"):
        st.toast("Markdown copied to clipboard!", icon="✅")
        st.session_state["already_copied"] = False

    # Handle downloads
    if download_button:
        try:
            handle_download_pdfs(
                st.session_state["temp_file_path"], df, id_col, url_col, use_specific_id
            )
        except Exception as e:
            st.error(f"Error during downloading PDFs: {e}")

    # Handle temp clearing
    if clear_temp_button:
        clear_temp_dir(TEMP_DIR)
        if os.path.exists("exported_results.zip"):
            os.remove("exported_results.zip")
        st.sidebar.success("Temporary files cleared.")
        st.rerun()
    # Clean old files
    clean_old_files(max_age_minutes=30)

    # PDF processing
    files = handle_pdf_processing(export_to_markdown, number_thread, overwrite)

    # PDF Preview
    if files:
        render_preview_file(files, export_to_markdown)


main()

# Cleanup on exit
atexit.register(clear_temp_dir, TEMP_DIR)
atexit.register(clear_temp_dir, OUTPUT_DIR)
atexit.register(clear_temp_dir, TEMP_DIR_PDF)
atexit.register(clear_temp_dir, DATA_TEMP)
