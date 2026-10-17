
from guardrails import validate_question, check_retrieved_context
import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "idp_documents"


def retrieve_context(question: str):
    valid, message = validate_question(question)
    if not valid:
        return {"answer": message, "sources": []}

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    results = collection.get(
        where={"record_type": "chunk"},
        include=["documents", "metadatas"],
    )

    documents = results["documents"] or []
    metadatas = results["metadatas"] or []

    if not documents:
        return {
            "answer": "Information not found in the provided documents.",
            "sources": [],
        }

    # Initial safe baseline: choose the first stored chunk.
    # Next, we'll connect this function to your existing hybrid retrieval.
    context = [documents[0]]
    usable, message = check_retrieved_context(context)

    if not usable:
        return {"answer": message, "sources": []}

    return {
        "answer": "Context retrieved successfully. LLM generation is not connected yet.",
        "context": context,
        "sources": [
            {
                "source": metadatas[0].get("source", "Unknown"),
                "page": metadatas[0].get("page", "Not available"),
            }
        ],
    }


if __name__ == "__main__":
    question = input("Ask a question: ")
    result = retrieve_context(question)

    print("\nResult:")
    print(result)
