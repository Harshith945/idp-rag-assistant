
import hashlib
from pathlib import Path

import boto3
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# AWS S3 configuration
AWS_REGION = "us-east-1"
BUCKET_NAME = "idp-rag-docs-2026-hk739-demo"
S3_PREFIX = "extracted/"

# Local paths
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

# ChromaDB configuration
COLLECTION_NAME = "idp_documents"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


s3 = boto3.client("s3", region_name=AWS_REGION)


def download_extracted_files():
    """Download extracted text files from S3 to the local data folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    downloaded = []

    print(f"Checking S3 bucket: {BUCKET_NAME}")
    print(f"S3 prefix: {S3_PREFIX}")

    for page in paginator.paginate(
        Bucket=BUCKET_NAME,
        Prefix=S3_PREFIX,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.lower().endswith(".txt"):
                continue

            filename = key[len(S3_PREFIX):]

            # This version supports files directly under extracted/
            if not filename or "/" in filename:
                continue

            local_path = DATA_DIR / filename

            s3.download_file(
                BUCKET_NAME,
                key,
                str(local_path),
            )

            downloaded.append(local_path)

            print(f"Downloaded: s3://{BUCKET_NAME}/{key}")
            print(f"Saved locally: {local_path.resolve()}")

    if not downloaded:
        print("No extracted .txt files found in S3.")

    return downloaded


def calculate_hash(text):
    """Return a SHA-256 hash of the document text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    # 1. Download extracted text files from S3
    files = download_extracted_files()

    if not files:
        print("Nothing to index. Check the S3 bucket and extracted/ prefix.")
        return

    # 2. Connect to persistent ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 3. Load the embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 4. Configure text chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    indexed_count = 0
    skipped_count = 0
    total_chunks_created = 0

    # 5. Process each downloaded document
    for file_path in files:
        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            skipped_count += 1
            continue

        content_hash = calculate_hash(text)
        hash_id = f"document_hash:{file_path.name}"

        # Check whether this document's content has changed
        existing = collection.get(
            ids=[hash_id],
            include=["metadatas"],
        )

        if existing["ids"]:
            old_metadata = existing["metadatas"][0] or {}

            if old_metadata.get("content_hash") == content_hash:
                print(f"Unchanged; skipping: {file_path.name}")
                skipped_count += 1
                continue

        # Split the document into chunks
        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": file_path.name}],
        )

        if not chunks:
            print(f"No chunks created: {file_path.name}")
            skipped_count += 1
            continue

        # Remove the previous chunks for this document
        previous = collection.get(
            where={"source": file_path.name},
            include=["metadatas"],
        )

        old_chunk_ids = [
            chunk_id
            for chunk_id, metadata in zip(
                previous["ids"],
                previous["metadatas"] or [],
            )
            if metadata.get("record_type") != "document_hash"
        ]

        if old_chunk_ids:
            collection.delete(ids=old_chunk_ids)

        # Prepare chunks, IDs, and metadata
        documents = [chunk.page_content for chunk in chunks]

        chunk_ids = [
            f"{file_path.name}:{chunk.metadata.get('start_index', i)}"
            for i, chunk in enumerate(chunks)
        ]

        metadatas = [
            {
                "source": file_path.name,
                "chunk_index": i,
                "start_index": chunk.metadata.get("start_index", 0),
                "content_hash": content_hash,
                "record_type": "chunk",
            }
            for i, chunk in enumerate(chunks)
        ]

        # Generate embeddings locally
        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Save document chunks and embeddings
        collection.upsert(
            ids=chunk_ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        # Save a marker to detect unchanged documents on later runs
        marker_text = f"Indexed document version: {file_path.name}"
        marker_embedding = model.encode(
            [marker_text],
            normalize_embeddings=True,
        ).tolist()[0]

        collection.upsert(
            ids=[hash_id],
            documents=[marker_text],
            metadatas=[
                {
                    "source": file_path.name,
                    "content_hash": content_hash,
                    "record_type": "document_hash",
                }
            ],
            embeddings=[marker_embedding],
        )

        indexed_count += 1
        total_chunks_created += len(chunks)

        print(
            f"Indexed {file_path.name}: "
            f"{len(chunks)} chunks"
        )

    # 6. Print indexing summary
    all_records = collection.get(
        include=["metadatas"],
    )

    record_types = all_records["metadatas"] or []

    total_chunk_records = sum(
        1
        for metadata in record_types
        if metadata.get("record_type") == "chunk"
    )

    total_document_markers = sum(
        1
        for metadata in record_types
        if metadata.get("record_type") == "document_hash"
    )

    print("\n========== Indexing Summary ==========")
    print(f"Text files downloaded from S3: {len(files)}")
    print(f"Documents newly indexed/updated: {indexed_count}")
    print(f"Unchanged or empty documents skipped: {skipped_count}")
    print(f"Chunks created in this run: {total_chunks_created}")
    print(f"Total chunk records in ChromaDB: {total_chunk_records}")
    print(f"Document hash markers: {total_document_markers}")
    print(f"Total ChromaDB records: {collection.count()}")
    print(f"ChromaDB location: {CHROMA_DIR.resolve()}")
    print("======================================")


if __name__ == "__main__":
    main()
