import time
from pathlib import Path

import boto3
import streamlit as st
from botocore.exceptions import ClientError

import ingest
from src.rag_chain import RAGChain


# ---------------- CONFIG ----------------

AWS_REGION = "us-east-1"
BUCKET_NAME = "idp-rag-docs-2026-hk739-demo"

s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="centered",
)


# ---------------- STYLING ----------------

st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8fc;
        color: #172033;
    }

    [data-testid="stHeader"] {
        background: rgba(246, 248, 252, 0.95);
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #172b4d;
        letter-spacing: -0.4px;
    }

    .brand {
        color: #2563eb;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
        line-height: 1.6;
        margin-top: -8px;
        margin-bottom: 22px;
    }

    .upload-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        margin: 16px 0 24px 0;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
    }

    .section-label {
        color: #334155;
        font-size: 16px;
        font-weight: 650;
        margin-bottom: 4px;
    }

    .helper-text {
        color: #64748b;
        font-size: 13px;
        line-height: 1.5;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 12px;
    }

    div[data-testid="stFileUploader"] section {
        border: 1px dashed #b9c8df;
        border-radius: 12px;
        background: #fbfdff;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
        transition: all 0.15s ease;
    }

    div.stButton > button[kind="primary"] {
        background: #2563eb;
        border: 1px solid #2563eb;
        color: #ffffff;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #e6ebf3;
        border-radius: 14px;
        padding: 12px 15px;
        background: #ffffff;
        margin-bottom: 12px;
    }

    div[data-testid="stChatInput"] {
        border-color: #d8e1ee;
        border-radius: 14px;
    }

    .source-label {
        color: #64748b;
        font-size: 12px;
        margin-top: 8px;
    }

    hr {
        border-color: #e2e8f0;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- SESSION STATE ----------------

if "rag" not in st.session_state:
    st.session_state.rag = None

if "active_source" not in st.session_state:
    st.session_state.active_source = None

if "active_filename" not in st.session_state:
    st.session_state.active_filename = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- HELPERS ----------------

def wait_for_extraction(key: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            s3.head_object(
                Bucket=BUCKET_NAME,
                Key=key,
            )
            return True

        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")

            if code not in ("404", "NoSuchKey", "NotFound"):
                raise

        time.sleep(2)

    return False


def process_pdf(uploaded_file) -> None:
    filename = Path(uploaded_file.name).name
    source_name = f"{Path(filename).stem}.txt"
    extracted_key = f"extracted/{source_name}"

    # Clear the previous document's chat so it cannot be mistaken
    # for the newly selected document.
    st.session_state.rag = None
    st.session_state.active_source = None
    st.session_state.active_filename = None
    st.session_state.messages = []

    with st.spinner("Uploading PDF to Amazon S3..."):
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=uploaded_file.getvalue(),
            ContentType="application/pdf",
        )

    with st.spinner("Waiting for document extraction..."):
        if not wait_for_extraction(extracted_key):
            raise TimeoutError(
                f"Could not find {extracted_key} within 60 seconds. "
                "Check the Lambda logs and output filename."
            )

    with st.spinner("Indexing document for retrieval..."):
        # Existing ingestion indexes extracted text into local ChromaDB.
        ingest.main()

    with st.spinner("Preparing the assistant..."):
        st.session_state.rag = RAGChain()
        st.session_state.active_source = source_name
        st.session_state.active_filename = filename
        st.session_state.messages = []


# ---------------- HEADER ----------------

st.markdown(
    '<div class="brand">DOCUMENT INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.title("DocuMind AI")

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF, then ask questions answered from that document.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------- UPLOAD ----------------

st.markdown(
    """
    <div class="upload-panel">
        <div class="section-label">Upload your document</div>
        <div class="helper-text">
            PDF files are stored in Amazon S3, processed by Lambda,
            and indexed for semantic search.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Upload the PDF you want to query.",
)

if uploaded_file is not None:
    st.caption(
        f"Selected: {uploaded_file.name} · "
        f"{uploaded_file.size / 1024:.1f} KB"
    )

    if st.button(
        "Process document",
        type="primary",
        use_container_width=True,
    ):
        try:
            process_pdf(uploaded_file)
            st.success("Document ready. You can ask questions below.")

        except Exception as exc:
            st.error(f"Could not process the document: {exc}")


# ---------------- ACTIVE DOCUMENT ----------------

if st.session_state.active_filename:
    st.markdown("---")
    st.markdown("#### Current document")

    st.markdown(
        f"📄 **{st.session_state.active_filename}**"
    )

    st.caption("Answers are restricted to this document.")


# ---------------- CHAT ----------------

st.markdown("---")
st.markdown("#### Ask your document")

if st.session_state.rag is None:
    st.info("Upload and process a PDF to start asking questions.")

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                st.caption(
                    "Source: " + ", ".join(message["sources"])
                )

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your document..."):
                try:
                    result = st.session_state.rag.ask(
                        question,
                        source=st.session_state.active_source,
                    )

                    answer = result.get(
                        "answer",
                        "Information not found in the provided documents.",
                    )

                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        st.caption("Source: " + ", ".join(sources))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception:
                    st.error(
                        "Something went wrong while retrieving or "
                        "generating the answer. Check your application logs."
                    )
