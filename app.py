import time
import re
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
    layout="wide",
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
        background: rgba(246,248,252,0.95);
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #172b4d;
    }
    .brand {
        color: #2563eb;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }
    .subtitle {
        color: #64748b;
        font-size: 15px;
        line-height: 1.6;
    }
    div[data-testid="stChatMessage"] {
        border: 1px solid #e6ebf3;
        border-radius: 14px;
        padding: 12px 15px;
        background: #ffffff;
        margin-bottom: 12px;
    }
    div.stButton > button {
        border-radius: 10px;
        min-height: 40px;
        font-weight: 600;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- SESSION STATE ----------------

defaults = {
    "rag": None,
    "active_source": None,
    "active_filename": None,
    "messages": [],
    "loaded_document": None,
    "processed_upload": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- GREETINGS ----------------

def handle_greeting(question: str):
    """Answer common conversational messages without calling the LLM."""
    text = re.sub(r"\s+", " ", question.lower().strip())
    text = re.sub(r"[.!?,]+$", "", text).strip()

    if re.fullmatch(
        r"(hi|hello|hey|hiya|hi there|hello there|hey there)( everyone)?",
        text,
    ):
        return (
            "Hello! 👋 Welcome to DocuMind AI. "
            "Upload a PDF or choose a previous document, "
            "and I'll help you find answers."
        )

    if text in {
        "good morning",
        "good afternoon",
        "good evening",
    }:
        return (
            f"{text.capitalize()}! 😊 "
            "How can I help you with your documents today?"
        )

    if text in {
        "thanks",
        "thank you",
        "thanks a lot",
        "thank you so much",
    }:
        return "You're welcome! 😊 Feel free to ask another question."

    if text in {
        "who are you",
        "what are you",
        "what is your name",
    }:
        return (
            "I'm DocuMind AI, your Document Intelligence Assistant. "
            "I retrieve relevant information from your selected PDF "
            "and use it to answer your questions."
        )

    if text in {
        "help",
        "what can you do",
        "how does this work",
    }:
        return (
            "I can help you explore PDF documents. Upload a PDF or "
            "select a previous file, then ask a question about its "
            "contents. I'll use document retrieval to find relevant "
            "information."
        )

    return None


# ---------------- S3 HELPERS ----------------

def list_pdf_files():
    """Return PDFs stored at the root of the configured S3 bucket."""
    files = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET_NAME):
        for item in page.get("Contents", []):
            key = item["Key"]

            if (
                "/" not in key
                and key.lower().endswith(".pdf")
            ):
                files.append(key)

    return sorted(files, key=str.lower)


def wait_for_extraction(key: str, timeout: int = 90) -> bool:
    """Wait for the Lambda-generated text file to appear in S3."""
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=key)
            return True

        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")

            if code not in ("404", "NoSuchKey", "NotFound"):
                raise

        time.sleep(2)

    return False


def get_extracted_source(filename: str) -> str:
    return f"{Path(filename).stem}.txt"


def reset_document_chat():
    st.session_state.messages = []


def load_document(filename: str):
    """Prepare RAG and set the active source to one document."""
    source_name = get_extracted_source(filename)
    extracted_key = f"extracted/{source_name}"

    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=extracted_key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")

        if code in ("404", "NoSuchKey", "NotFound"):
            raise FileNotFoundError(
                f"Extracted text for {filename} is not available yet. "
                "Check whether the S3-triggered Lambda completed."
            ) from exc

        raise

    with st.spinner("Preparing the document index..."):
        ingest.main()

    st.session_state.rag = RAGChain()
    st.session_state.active_source = source_name
    st.session_state.active_filename = filename
    st.session_state.loaded_document = filename
    reset_document_chat()


def process_pdf(uploaded_file):
    """Upload a PDF and wait for its Lambda-generated text."""
    filename = Path(uploaded_file.name).name
    extracted_key = f"extracted/{get_extracted_source(filename)}"

    # Clear an old output before re-uploading a same-named PDF.
    try:
        s3.delete_object(
            Bucket=BUCKET_NAME,
            Key=extracted_key,
        )
    except ClientError:
        pass

    with st.spinner("Uploading PDF to Amazon S3..."):
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=uploaded_file.getvalue(),
            ContentType="application/pdf",
        )

    with st.spinner("Waiting for Lambda document extraction..."):
        if not wait_for_extraction(extracted_key):
            raise TimeoutError(
                "The extracted text did not appear in S3 within "
                "90 seconds. Check your Lambda logs."
            )

    load_document(filename)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("📄 DocuMind AI")
    st.caption("Your document workspace")
    st.divider()

    st.subheader("Previous Files")

    try:
        previous_files = list_pdf_files()

        if not previous_files:
            st.info("Your uploaded PDFs will appear here.")

        else:
            current_index = 0

            if st.session_state.active_filename in previous_files:
                current_index = previous_files.index(
                    st.session_state.active_filename
                )

            selected_file = st.selectbox(
                "Choose a PDF",
                options=previous_files,
                index=current_index,
                format_func=lambda name: f"📄 {name}",
            )

            if selected_file != st.session_state.loaded_document:
                if st.button(
                    "Open selected document",
                    use_container_width=True,
                ):
                    try:
                        load_document(selected_file)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not open document: {exc}")

            if st.session_state.active_filename:
                st.caption(
                    f"Active: {st.session_state.active_filename}"
                )

    except Exception as exc:
        st.error(f"Could not list S3 files: {exc}")

    st.divider()
    st.caption("Documents are stored in Amazon S3.")


# ---------------- HEADER ----------------

st.markdown(
    '<div class="brand">DOCUMENT INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.title("DocuMind AI")

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF or choose a previous file, then ask questions '
    'about its contents.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------- UPLOAD ----------------

with st.container(border=True):
    st.subheader("Upload your document")
    st.caption(
        "PDFs are stored in Amazon S3, extracted by Lambda, "
        "and indexed for retrieval."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="Upload a PDF you want to ask questions about.",
    )

    if uploaded_file is not None:
        st.caption(
            f"{uploaded_file.name} · "
            f"{uploaded_file.size / 1024:.1f} KB"
        )

        upload_id = (
            uploaded_file.name,
            uploaded_file.size,
        )

        if st.button(
            "Process document",
            type="primary",
            use_container_width=True,
        ):
            try:
                process_pdf(uploaded_file)
                st.session_state.processed_upload = upload_id
                st.success(
                    "Document ready! You can ask questions below."
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Could not process the document: {exc}")


# ---------------- ACTIVE DOCUMENT ----------------

if st.session_state.active_filename:
    st.success(
        f"Active document: {st.session_state.active_filename}"
    )
    st.caption(
        "Document questions are restricted to the selected file."
    )
else:
    st.info(
        "You can greet the assistant now. To ask document questions, "
        "upload a PDF or open a previous file."
    )


# ---------------- CHAT ----------------

st.divider()
st.subheader("Chat with DocuMind")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            st.caption(
                "Source: " + ", ".join(message["sources"])
            )


question = st.chat_input(
    "Say hello or ask a question about your PDF..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    greeting = handle_greeting(question)

    if greeting is not None:
        with st.chat_message("assistant"):
            st.markdown(greeting)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": greeting,
                "sources": [],
            }
        )

    elif st.session_state.rag is None:
        answer = (
            "Please upload a PDF or select a previous document "
            "before asking questions about document contents."
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [],
            }
        )

    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching your selected document..."):
                try:
                    result = st.session_state.rag.ask(
                        question,
                        source=st.session_state.active_source,
                    )

                    answer = result.get(
                        "answer",
                        "Information not found in the selected document.",
                    )
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        st.caption(
                            "Source: " + ", ".join(sources)
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception:
                    st.error(
                        "I couldn't retrieve an answer. "
                        "Check the EC2 terminal logs and try again."
                    )

