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

from pdf_process import (
    handle_pdf_download_from_dataset,
    read_dataset,
    ensure_temp_dir,
    clear_temp_dir,
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

# Constants
EXTENSION = {
    "csv": [".csv"],
    "xlsx": [".xlsx"],
    "pdf": [".pdf"],
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
    if "process_pdf_clicked" not in st.session_state:
        st.session_state["process_pdf_clicked"] = False
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


# Configure page
def setup_page():
    st.set_page_config(page_title="PDF Processing Dashboard", layout="wide")
    st.title("PDF Processing Dashboard")


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
        st.session_state["zip_path"] = zip_for_download(progress_callback=update_progress)
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

    if export_to_markdown:
        # Use glob for efficient file matching
        extracted_files = glob(pathname="**/*.json", root_dir=output_dir, recursive=True)
    elif st.session_state["method_option"] == "PyMuPDF + Tesseract":
        extracted_files = glob(pathname="*/*.json", root_dir=output_dir, recursive=True)
    else:
        extracted_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    return False if len(extracted_files) > 0 else True


def process_pdf_click():
    st.session_state["process_pdf_clicked"] = True


def cancel_processing():
    st.session_state["cancel_processing"] = True
    st.session_state["process_pdf_clicked"] = False

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
    separate_result_dir = st.sidebar.checkbox(
        "Create separate result directory",
        value=export_to_markdown,
        help="Create a separate folder for each PDF file. True if exporting to Markdown.",
    )
    overwrite = st.sidebar.toggle(
        "Overwrite existing files",
        value=False,
        help="Overwrite existing files if they exist.",
        key="overwrite",
    )

    # Dataset handling
    dataset_files = st.sidebar.file_uploader(
        "Upload Dataset (CSV/Excel/PDF)",
        type=["csv", "xlsx", "pdf"],
        accept_multiple_files=True,
        key=st.session_state["file_uploader_key"],
        on_change=toast_upload_success,
    )

    df = None
    column_list = []

    if dataset_files:
        for dataset_file in dataset_files:

            if re.search(r"\.(csv|xlsx)$", dataset_file.name, re.IGNORECASE):
                temp_file_path = DATA_TEMP / dataset_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(dataset_file.getbuffer())

                df = read_dataset(temp_file_path)
                column_list = df.columns.tolist()

            elif dataset_file.name.endswith(".pdf"):
                # Save directly to TEMP_DIR_PDF
                ensure_temp_dir()
                temp_file_path = TEMP_DIR_PDF / dataset_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(dataset_file.getbuffer())
        st.session_state["file_uploader_key"] = str(uuid.uuid4())
        st.rerun()

    # Check if there are existing CSV/Excel files in DATA_TEMP
    existing_files = list(DATA_TEMP.glob("*.csv")) + list(DATA_TEMP.glob("*.xlsx"))
    if existing_files:
        st.sidebar.info("Existing dataset files found in")
        selected_file = st.sidebar.selectbox(
        "Select a file to preview",
        options=[file.name for file in existing_files],
        help="Select a file from the existing dataset files in DATA_TEMP.",
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
        temp_file_path = None
        df = None
        column_list = []

    # Column selection
    if df is not None and column_list:
        id_col = st.sidebar.selectbox("Select ID Column", options=column_list)
        with st.sidebar.expander(f"Sample {id_col}", expanded=False):
            st.write(f"Sample: {df.loc[0, id_col]}")
        url_col = st.sidebar.selectbox("Select URL Column", options=column_list)
        with st.sidebar.expander(f"Sample {url_col}", expanded=False):
            st.write(f"Sample: {df.loc[0, url_col]}")
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
        separate_result_dir,
        overwrite,
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


def handle_download_pdfs(file_path, df, id_col, url_col):
    if df is not None and id_col != url_col:
        total_pdf_files = df[url_col].nunique()
        total_processing = 0
        total_success = 0
        failed_files = []

        with st.status("Downloading PDFs...", expanded=True) as status:
            results = handle_pdf_download_from_dataset(file_path, id_col, url_col)
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


def handle_pdf_processing(
    export_to_markdown, separate_result_dir, number_thread, overwrite
):
    ensure_temp_dir()
    pdf_files = os.listdir(TEMP_DIR_PDF)

    if not pdf_files:
        st.warning(
            "No PDF files found in the temporary PDF directory. Please upload or download PDFs first."
        )
        return None

    process_btn, method_options = st.columns(
        2, gap="small", vertical_alignment="bottom"
    )

    method_option_select = method_options.selectbox(
        "Select Processing Method",
        options=["Docling", "PyMuPDF + Tesseract"],
        index=0,
        key="method_option",
    )

    process_pdf_btn = process_btn.button(
        "Process PDF",
        key="process_pdf",
        disabled=False if pdf_files else True,
        on_click=process_pdf_click,
    )

    if process_pdf_btn:
        st.session_state["cancel_processing"] = False  # Uncommented to enable processing

        # Stop button
        stop_button_disabled = st.session_state["cancel_processing"]

        if st.session_state["process_pdf_clicked"]:
            st.session_state["cancel_processing"] = False
            st.session_state["process_pdf_clicked"] = False

            stop_button = st.button(
                "Stop", on_click=cancel_processing, disabled=stop_button_disabled
            )

        total_files = len(pdf_files)
        pdf_status = st.empty()
        total_success = 0
        total_failed = 0
        total_skipped = 0

        with st.status(
            f"Processing PDFs to {'Markdown and JSON' if export_to_markdown else 'JSON'} files...",
            expanded=True,
        ) as status:
            for idx, pdf_filename in enumerate(pdf_files, 1):
                if st.session_state["cancel_processing"]:
                    status.warning("Processing canceled by user.")
                    break

                pdf_status.info(f"Processing: {pdf_filename.split('.')[0]}")
                page_processing_slot_status = st.empty()

                if method_option_select == "Docling":
                    output_dir = OUTPUT_DIR / "docling_results"

                    output_dir.mkdir(parents=True, exist_ok=True)

                    for log in process_pdf(
                        os.path.join(TEMP_DIR_PDF, pdf_filename),
                        create_markdown=export_to_markdown,
                        overwrite=overwrite,
                        separate_result_dir=separate_result_dir,
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
                            pdf_status.success(
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
                        label=f"Processing: {idx}/{total_files} PDFs | Success {total_success} | Skipped {total_skipped} | Failed {total_failed}"
                    )

                    st.session_state["process_pdf_clicked"] = False
                    page_processing_slot_status.empty()

                if method_option_select == "PyMuPDF + Tesseract":
                    FOLDER_OUTPUT_PYMU_TESSERACT.mkdir(parents=True, exist_ok=True)
                    for log in process_pdf_pymu_tesseract(
                        os.path.join(TEMP_DIR_PDF, pdf_filename),
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
                            pdf_status.success(
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
                        label=f"Processing: {idx}/{total_files} PDFs | Success {total_success} | Skipped {total_skipped} | Failed {total_failed}"
                    )

                    st.session_state["process_pdf_clicked"] = False
                    page_processing_slot_status.empty()

            pdf_status.empty()

            if not st.session_state["cancel_processing"]:
                status.success(
                    f"PDFs converted to {'Markdown and JSON' if export_to_markdown else 'JSON'} files. Total Success: {total_success}, Skipped {total_skipped}, Failed: {total_failed}"
                )

    return pdf_files


def render_pdf_preview(pdf_files, export_to_markdown):
    if not pdf_files or len(pdf_files) == 0:
        st.info("No PDFs available for preview.")
        return

    query_pdf = st.selectbox(
        "Select PDF to preview",
        options=pdf_files,
        index=0,
        key="pdf_select",
        placeholder="Select a PDF file",
    )

    pdf_path = os.path.join(TEMP_DIR_PDF, query_pdf)
    doc = pymupdf.open(pdf_path)

    if doc is None:
        st.error("Failed to open the PDF document.")
        return

    if doc.page_count == 0:
        st.error("The PDF document is empty.")
        return

    page_number = st.number_input(
        "Select Page Number",
        min_value=1,
        max_value=doc.page_count,
        value=1,
        step=1,
        key="page_number",
    )

    pdf, result = st.columns(2, border=True)

    with pdf:
        st.write("Selected PDF:")
        pdf_viewer(
            pdf_path,
            width=900,
            height=700,
            pages_to_render=[page_number],
            key=f"pdf_viewer_{query_pdf.split('.')[0]}_{page_number}",
        )

    with result:
        st.write("Markdown Result:")

        @st.dialog("Markdown Result")
        def raw_markdown(raw_markdown):
            st.code(raw_markdown, language="markdown")
            if st.button("Close"):
                st.session_state["show_raw_markdown"] = False
                st.session_state["raw_markdown_content"] = None
                st.session_state["raw_markdown_pdf_id"] = None
                st.session_state["raw_markdown_page_number"] = None
                st.rerun()

        pdf_id = query_pdf.split(".")[0]

        if st.session_state["method_option"] == "Docling":
            base_path = os.path.join(OUTPUT_DIR, "docling_results")
        elif st.session_state["method_option"] == "PyMuPDF + Tesseract":
            base_path = FOLDER_OUTPUT_PYMU_TESSERACT
        else:
            base_path = OUTPUT_DIR

        if export_to_markdown and st.session_state["method_option"] == "Docling":
            result_path = os.path.join(base_path, pdf_id, pdf_id + ".json")
        else:
            result_path = os.path.join(base_path, pdf_id + ".json")

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                json_result = json.load(f)
                content = json_result.get("content", [])

                if 0 <= page_number - 1 < len(content):
                    selected_page = content[page_number - 1]["content"]
                    raw_md_button = st.button(
                        "Show Raw Markdown",
                        key=f"raw_md_{pdf_id}_{page_number}",
                        help="Click to view raw markdown content.",
                    )
                    if raw_md_button:
                        st.session_state["show_raw_markdown"] = True
                        st.session_state["raw_markdown_content"] = selected_page
                        st.session_state["raw_markdown_pdf_id"] = pdf_id
                        st.session_state["raw_markdown_page_number"] = page_number
                        raw_markdown(selected_page)

                    with st.container(key="markdown_result", height=600):
                        st.markdown(selected_page, unsafe_allow_html=True)
                else:
                    st.info("No markdown content for this page.")
        else:
            st.info("No result JSON found for this PDF.")


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
        separate_result_dir,
        overwrite,
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

    # Handle downloads
    if download_button:
        try:
            handle_download_pdfs(st.session_state["temp_file_path"], df, id_col, url_col)
        except Exception as e:
            st.error("Error during downloading PDFs: Please check your dataset")

    # Handle temp clearing
    if clear_temp_button:
        clear_temp_dir(TEMP_DIR)
        if os.path.exists("exported_results.zip"):
            os.remove("exported_results.zip")
        st.sidebar.success("Temporary files cleared.")
        st.rerun()

    # PDF processing
    pdf_files = handle_pdf_processing(
        export_to_markdown, separate_result_dir, number_thread, overwrite
    )

    # PDF Preview
    if pdf_files:
        render_pdf_preview(pdf_files, export_to_markdown)


main()
