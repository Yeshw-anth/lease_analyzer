# Lease Analyzer

This project is a sophisticated lease document analysis tool designed to automatically extract key information from lease agreements and answer user questions in a conversational manner. It uses a hybrid approach that combines deterministic methods (like regular expressions) with advanced AI-powered Retrieval-Augmented Generation (RAG) to ensure high accuracy and robustness.

## Features

- **Hybrid Extraction Pipeline**: The system first attempts to extract information using a high-confidence deterministic approach. If that fails, it automatically falls back to a powerful RAG model to find the information, ensuring the best of both worlds.
- **Multi-Query AI Search**: For critical fields like "Tenant" and "Lease Start Date," the system automatically tries multiple variations of a query (e.g., "Lease Commencement Date," "start date of the lease term") to overcome mismatches in terminology.
- **Interactive RAG Chat**: After the initial analysis, you can ask direct questions about the lease document through a chat interface and receive answers with source references (page and section number).
- **Multiple LLM Providers**: The system is designed to be flexible and supports several language model providers:
  - **OpenAI** (e.g., GPT-3.5-turbo, GPT-4)
  - **Google Gemini**
  - **Ollama** (for running local models like Llama 3)
- **Excel Template Integration**: Users can upload an Excel template with the fields they want to extract. The system will populate this template with the extracted data and make it available for download.
- **Data Normalization**: Automatically normalizes extracted data, such as dates and currency amounts, into a consistent format.
- **Extraction Audit**: Provides a summary of how each piece of information was extracted (e.g., deterministic success, RAG fallback), giving you insight into the system's performance.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Yeshw-anth/lease_doc_analyzer
    cd lease_doc_analyzer
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created if it doesn't exist. It would include libraries like `streamlit`, `pandas`, `google-generativeai`, `openai`, `ollama`, `sentence-transformers`, `faiss-cpu`, etc.)*

## Usage

1.  **Run the Streamlit application:**
    ```bash
    python -m streamlit run lease_doc_analyzer/app.py
    ```

2.  **Open your web browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Configure the LLM Provider** in the sidebar:
    -   Choose between "openai," "gemini," and "ollama."
    -   If using OpenAI or Gemini, enter the corresponding API key.

4.  **Upload Files**:
    -   Upload the lease agreement in PDF format.
    -   Upload an Excel file containing the fields you wish to extract in the first column.

5.  **Analyze and Interact**:
    -   Click the "Analyze Lease" button to start the extraction process.
    -   Once complete, you can download the results as an Excel file.
    -   Use the chat interface at the bottom of the page to ask further questions about the document.

## Project Structure

```
lease_analyzer/
├── __init__.py
├── app.py                  # Main Streamlit application UI and workflow
├── deterministic_extractor.py # Functions for rule-based/regex extraction
├── llm_provider.py         # Abstraction for different LLM providers (OpenAI, Gemini, Ollama)
├── models.py               # Data models (e.g., LLMResponse)
├── normalizer.py           # Functions for data normalization (dates, currency)
├── pdf_parser.py           # Logic for parsing and chunking PDF files
└── qa_system.py            # Core logic for the QA system, including RAG and multi-query search
```