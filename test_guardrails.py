
from guardrails import (
    validate_question,
    check_retrieved_context,
    validate_answer,
    no_answer_response,
)

print("Test 1 — Valid question:")
print(validate_question("What is the invoice total?"))

print("\nTest 2 — Empty question:")
print(validate_question(""))

print("\nTest 3 — Missing context:")
print(check_retrieved_context([]))

print("\nTest 4 — Valid context:")
print(check_retrieved_context(["Invoice total: Rs. 1280.30"]))

print("\nTest 5 — Empty answer:")
print(validate_answer(""))

print("\nTest 6 — No-answer response:")
print(no_answer_response())
