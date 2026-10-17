
import re


MAX_QUESTION_LENGTH = 1000
MIN_QUESTION_LENGTH = 3


def validate_question(question: str) -> tuple[bool, str]:
    """Validate a user's question before retrieval."""

    if not isinstance(question, str):
        return False, "Please enter a valid text question."

    question = question.strip()

    if len(question) < MIN_QUESTION_LENGTH:
        return False, "Please enter a meaningful question."

    if len(question) > MAX_QUESTION_LENGTH:
        return False, (
            f"Your question is too long. "
            f"Maximum length is {MAX_QUESTION_LENGTH} characters."
        )

    if not re.search(r"[a-zA-Z0-9]", question):
        return False, "Please enter a question containing text or numbers."

    return True, ""


def check_retrieved_context(context_chunks: list[str]) -> tuple[bool, str]:
    """Check whether retrieval returned usable document context."""

    if not context_chunks:
        return False, "No relevant information was found in the documents."

    usable_chunks = [
        chunk.strip()
        for chunk in context_chunks
        if isinstance(chunk, str) and chunk.strip()
    ]

    if not usable_chunks:
        return False, "No relevant information was found in the documents."

    return True, ""


def validate_answer(answer: str) -> tuple[bool, str]:
    """Reject empty or malformed LLM responses."""

    if not isinstance(answer, str) or not answer.strip():
        return False, "The assistant could not generate an answer."

    return True, answer.strip()


def no_answer_response() -> str:
    """Safe response when document evidence is unavailable."""

    return "Information not found in the provided documents."
