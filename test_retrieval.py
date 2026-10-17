
import re

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "idp_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3
RRF_K = 60


def tokenize(text):
    """Split text into searchable tokens."""
    return re.findall(r"\b[\w.-]+\b", text.lower())


def reciprocal_rank_fusion(vector_results, bm25_results):
    """Combine vector and BM25 rankings using RRF."""
    scores = {}
    matched_by = {}

    for rank, doc_id in enumerate(vector_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (
            1 / (RRF_K + rank)
        )
        matched_by.setdefault(doc_id, set()).add("Vector")

    for rank, doc_id in enumerate(bm25_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (
            1 / (RRF_K + rank)
        )
        matched_by.setdefault(doc_id, set()).add("BM25")

    ranked_ids = sorted(
        scores,
        key=lambda doc_id: scores[doc_id],
        reverse=True,
    )

    return ranked_ids, scores, matched_by


def main():
    # Connect to the persistent ChromaDB database.
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Load only actual chunks, excluding document hash markers.
    records = collection.get(
        where={"record_type": "chunk"},
        include=["documents", "metadatas"],
    )

    documents = records["documents"]
    metadatas = records["metadatas"]
    doc_ids = records["ids"]

    if not documents:
        print("No indexed chunks found. Run python ingest.py first.")
        return

    print(f"Loaded {len(documents)} document chunks.")

    # Build BM25 keyword search over the indexed chunks.
    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    # Load the same embedding model used during ingestion.
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    question = input(
        "\nAsk a question about your documents: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    # 1. Vector search with ChromaDB.
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    vector_output = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, len(documents)),
        where={"record_type": "chunk"},
        include=["documents", "metadatas", "distances"],
    )

    vector_ids = vector_output["ids"][0]

    # 2. BM25 keyword search.
    bm25_scores = bm25.get_scores(tokenize(question))

    bm25_indices = sorted(
        range(len(documents)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:TOP_K]

    bm25_ids = [doc_ids[i] for i in bm25_indices]

    # 3. Fuse both ranked lists using Reciprocal Rank Fusion.
    fused_ids, rrf_scores, matched_by = reciprocal_rank_fusion(
        vector_ids,
        bm25_ids,
    )

    id_to_index = {
        doc_id: index
        for index, doc_id in enumerate(doc_ids)
    }

    print("\n========== HYBRID SEARCH RESULTS ==========")
    print(f"Question: {question}")

    for rank, doc_id in enumerate(fused_ids[:TOP_K], start=1):
        index = id_to_index[doc_id]

        print(f"\n--- Result {rank} ---")
        print(f"Source: {metadatas[index].get('source', 'Unknown')}")
        print(
            f"Chunk index: "
            f"{metadatas[index].get('chunk_index', 'N/A')}"
        )
        print(f"Actual RRF score: {rrf_scores[doc_id]:.5f}")
        print(
            "Retrieved by: "
            + ", ".join(sorted(matched_by[doc_id]))
        )
        print("Content:")
        print(documents[index])

    print("\n===========================================")


if __name__ == "__main__":
    main()
