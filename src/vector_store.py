
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "idp_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    """Search ChromaDB chunks, optionally restricted to one document."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self.model = SentenceTransformer(MODEL_NAME)

    def search(
        self,
        query: str,
        k: int = 3,
        source: str | None = None,
    ) -> list[dict]:
        if not query.strip():
            return []

        # Count actual chunks, excluding document hash markers.
        chunk_records = self.collection.get(
            where={"record_type": "chunk"},
            include=["metadatas"],
        )

        chunk_count = len(chunk_records["ids"])

        if chunk_count == 0:
            return []

        filters = {"record_type": "chunk"}

        if source:
            filters = {
                "$and": [
                    {"record_type": "chunk"},
                    {"source": source},
                ]
            }

            # Avoid querying a document that has no indexed chunks.
            matching_records = self.collection.get(
                where=filters,
                include=["metadatas"],
            )

            if not matching_records["ids"]:
                return []

            result_count = len(matching_records["ids"])
        else:
            result_count = chunk_count

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, result_count),
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        documents = results["documents"][0] or []
        metadatas = results["metadatas"][0] or []
        distances = results["distances"][0] or []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            matches.append({
                "content": document,
                "metadata": metadata or {},
                "distance": distance,
            })

        return matches
