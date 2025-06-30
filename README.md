# PDF Extractor using PyMuPDF + Tesseract and Docling

PDF Extractor is a Python-based tool that extracts text from PDF files using PyMuPDF and Tesseract OCR. It is designed to handle both text-based and image-based PDFs, making it versatile for various use cases.

## How to Use

We test two methods to run this project, we can concluded:

- Docker is the easiest way to run this project
- Clone the repository is the fastest way to extract text from PDF
- If you want to modify the code, you can clone the repository and run it locally

### Using Docker

1. **Pull the Docker image**  
    Run the following command to pull the latest Docker image:

    ```bash
    docker pull dzuladj/pdf-extractor:latest
    ```

2. **Run the Docker container**  
    Use this command to start the Docker container:

    ```bash
    docker run -p 8501:8501 dzuladj/pdf-extractor:latest
    ```

3. **Access the Web Interface**  
    Open your web browser and go to `http://localhost:8501` to access the web interface.

### Clone the Repository

1. **Clone the repository**  
    Clone this repository to your local machine:

    ```bash
    git clone https://github.com/Dzoel31/pdf-extractor.git
    ```

2. **Navigate to the project directory**

    ```bash
    cd pdf-extractor
    ```

3. **Create a virtual environment**  
    Create a virtual environment to manage dependencies:

    ```bash
    python -m venv venv
    ```

4. **Activate the virtual environment**

    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

5. **Install dependencies**
    Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

6. **Install Tesseract OCR**
    - **Windows**: Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

    - **macOS**: Use Homebrew to install Tesseract:

        ```bash
        brew install tesseract
        ```
    - **Linux**: Use the package manager to install Tesseract:

        ```bash
        sudo apt-get install tesseract-ocr
        ```

7. **Run the application**
    Start the Streamlit application:

    ```bash
    streamlit run app\dashboard.py
    ```
8. **Access the Web Interface**
