
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local embedding model on CPU."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
