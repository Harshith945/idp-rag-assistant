LIVE DEMO : http://98.93.47.244:8501/

# AWS IDP RAG Document Intelligence Assistant

An AI-powered document intelligence application that extracts
information from PDF documents and answers user questions using
Retrieval-Augmented Generation (RAG). It supports text-based PDFs and
scanned PDFs using PyMuPDF and Tesseract OCR.

## Overview

The application allows users to upload PDF documents, process their
content, and ask natural-language questions. Relevant document chunks
are retrieved from ChromaDB and passed to a Large Language Model (LLM)
to generate context-aware answers with source references.

## Key Features

-   **PDF Upload:** Upload PDF documents through a Streamlit web
    interface.
-   **Text Extraction:** Extract text from regular PDFs using PyMuPDF.
-   **OCR Processing:** Use Tesseract OCR for scanned or image-based
    PDFs.
-   **Document Chunking:** Split extracted content into smaller chunks
    for retrieval.
-   **Vector Search:** Generate embeddings and store document chunks in
    ChromaDB.
-   **RAG Question Answering:** Retrieve relevant context and generate
    answers using Groq.
-   **Source References:** Display source information alongside
    generated answers.
-   **Guardrails:** Apply validation to help prevent unsupported or
    irrelevant responses.
-   **AWS Deployment:** Run the application on Amazon EC2, with Amazon
    S3 for document storage and AWS Lambda for document processing.

## Architecture

1.  Users upload PDF documents through the Streamlit application hosted
    on Amazon EC2.
2.  Uploaded documents are stored in Amazon S3.
3.  An S3 event triggers AWS Lambda for document processing.
4.  PyMuPDF extracts text from regular PDFs, while Tesseract OCR
    processes scanned documents.
5.  Extracted text is saved in Amazon S3.
6.  The ingestion pipeline chunks the text and generates embeddings
    using the Hugging Face `all-MiniLM-L6-v2` model.
7.  ChromaDB stores the document chunks, embeddings, and associated
    metadata.
8.  When a user asks a question, the retrieval pipeline finds relevant
    document chunks.
9.  Groq generates an answer using the retrieved context.
10. The Streamlit interface displays the answer with source references.

## Technology Stack

  Component                   Technology
  --------------------------- -------------------------------------
  Programming language        Python 3.12
  Web interface               Streamlit
  Cloud compute               Amazon EC2
  Document storage            Amazon S3
  Event-driven processing     AWS Lambda
  PDF text extraction         PyMuPDF
  OCR                         Tesseract OCR
  Embedding model             Hugging Face `all-MiniLM-L6-v2`
  Vector database             ChromaDB
  RAG framework               LangChain
  LLM provider                Groq API
  Validation and guardrails   Python / Pydantic, where configured

## Project Structure

``` text
idp-rag-assistant/
├── app.py
├── ingest.py
├── guardrails.py
├── requirements.txt
├── .gitignore
├── src/
│   ├── embeddings.py
│   ├── rag_chain.py
│   └── vector_store.py
├── data/
├── chroma_db/
└── README.md
```

*The exact contents of `src/`, `data/`, and `chroma_db/` may vary
according to your local setup. Do not commit secrets or unnecessary
generated database files.*

## Installation and Setup

### 1. Clone the repository

``` bash
git clone https://github.com/Harshith945/idp-rag-assistant.git
cd idp-rag-assistant
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on Windows:

``` bash
.venv\Scripts\activate
```

On Linux:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

Tesseract OCR must also be installed and configured for your operating
system. If your Lambda function uses a packaged OCR dependency, ensure
that the deployed package contains the required binaries and libraries.

### 4. Configure environment variables

Create a local `.env` file and configure the required Groq API key and
any other environment variables used by your application.

Never commit `.env` or expose API keys in source code.

### 5. Configure AWS

Configure AWS credentials or an IAM role with the minimum required
permissions for the application's S3 and Lambda workflow.

Set the S3 bucket name and other AWS configuration values to match your
deployment. Configure the S3 event notification to invoke the
appropriate Lambda function.

### 6. Run the ingestion pipeline

Run the ingestion script after the extracted document text is available:

``` bash
python ingest.py
```

### 7. Launch the application

``` bash
streamlit run app.py
```

Open the local URL displayed in the terminal. For EC2 deployment,
configure Streamlit's host and port appropriately and ensure the
security group permits only the necessary traffic.

## Example Questions

-   What is the invoice number?
-   What is the invoice date?
-   What is the total invoice amount?
-   List the items and their quantities.
-   What information is available about a specific product or
    transaction?

The application should indicate when requested information cannot be
found in the retrieved document context.

## Security Considerations

-   Store API keys in environment variables rather than source code.
-   Use IAM roles and least-privilege permissions.
-   Enable appropriate S3 access controls and encryption.
-   Avoid exposing Streamlit publicly without suitable access controls.
-   Configure AWS budgets and billing alerts.
-   Stop unused EC2 instances and remove unused cloud resources.

## Limitations

-   OCR accuracy depends on scan quality, orientation, and image
    resolution.
-   Complex tables may not preserve their original row-and-column
    structure during text extraction.
-   RAG answer quality depends on document extraction, chunking,
    embeddings, and retrieval quality.
-   Groq is an external LLM API; the language model is not hosted on
    Amazon EC2 or SageMaker by this application.

## Future Improvements

-   Improve structured extraction of tables from complex PDFs.
-   Add automated evaluation for retrieval quality and answer accuracy.
-   Add authentication and user-level access controls.
-   Improve observability, error handling, and processing-status
    reporting.
-   Evaluate an AWS-hosted language model if required by deployment
    constraints.
