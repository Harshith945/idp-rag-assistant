
from src.rag_chain import RAGChain

rag = RAGChain()

result = rag.ask(
    "What information is available in the uploaded documents?"
)

print("Success:", result["success"])
print("Answer:", result["answer"])
print("Sources:", result["sources"])
