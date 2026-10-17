
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.vector_store import VectorStore
from guardrails import (
    validate_question,
    check_retrieved_context,
    validate_answer,
    no_answer_response,
)

load_dotenv()

NO_ANSWER = "Information not found in the provided documents."


class RAGChain:
    """Retrieve relevant chunks and answer using the selected document."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing from your .env file."
            )

        self.vector_store = VectorStore()

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=api_key,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a document intelligence assistant.

Answer the question using ONLY the provided document context.
Do not use outside knowledge or invent facts.
If the answer is not supported by the context, say exactly:
Information not found in the provided documents.

Use the source filename when relevant.
Be clear and concise.

Document context:
{context}""",
            ),
            ("human", "{question}"),
        ])

    def ask(
        self,
        question: str,
        k: int = 3,
        source: str | None = None,
    ) -> dict:
        valid, message = validate_question(question)

        if not valid:
            return {
                "success": False,
                "answer": message,
                "sources": [],
            }

        try:
            results = self.vector_store.search(
                query=question,
                k=k,
                source=source,
            )

            chunks = [
                item
                for item in results
                if item.get("content", "").strip()
            ]

            if not chunks:
                return {
                    "success": True,
                    "answer": no_answer_response(),
                    "sources": [],
                }

            context_ok, _ = check_retrieved_context(
                [item["content"] for item in chunks]
            )

            if not context_ok:
                return {
                    "success": True,
                    "answer": no_answer_response(),
                    "sources": [],
                }

            context_parts = []

            for item in chunks:
                filename = item["metadata"].get(
                    "source",
                    "Unknown",
                )

                # Extra protection: don't include other files in context.
                if source and filename != source:
                    continue

                context_parts.append(
                    f"Source: {filename}\n{item['content']}"
                )

            if not context_parts:
                return {
                    "success": True,
                    "answer": no_answer_response(),
                    "sources": [],
                }

            context = "\n\n".join(context_parts)

            response = self.llm.invoke(
                self.prompt.format_messages(
                    question=question,
                    context=context,
                )
            )

            answer_ok, answer = validate_answer(response.content)

            if not answer_ok:
                answer = no_answer_response()

            sources = list(dict.fromkeys(
                item["metadata"].get("source", "Unknown")
                for item in chunks
                if (
                    not source
                    or item["metadata"].get("source") == source
                )
            ))

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
            }

        except Exception:
            # Keep API or retrieval details out of the user-facing answer.
            raise
