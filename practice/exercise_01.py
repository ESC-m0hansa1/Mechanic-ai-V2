"""
EXERCISE 1 — write build_prompt() yourself, from blank.

RULES
  - Do NOT open app/generation/llm.py. Peeking = zero learning.
  - Write your code where the TODO is. Delete the `return ""` line.
  - Run it:   python practice/exercise_01.py
  - The test at the bottom tells you if you're right. Failing is expected. Iterate.
  - Stuck for more than ~10 minutes? Tell me WHERE you're stuck. That's not cheating.

STEP 1 — THE CONTRACT (you'd decide this before writing any logic)
  INPUT   question : str        e.g. "how much oil?"
  INPUT   chunks   : list[dict] each dict looks like {"content": "...", "page": 5}
  OUTPUT  str                   all chunks in one string, each labelled with its
                                page number, with the question at the end

STEP 2 — WHAT DO I HAVE  ->  WHAT DO I WANT
  I have :  a LIST of dicts        (many things)
  I want :  ONE string             (one thing)
  So     :  many -> one means I must LOOP over the list, build a small string
            for each item, then JOIN those small strings together.

STEP 3 — THE EXACT TARGET
  For the test data at the bottom, your function must return EXACTLY this text
  (blank line between the two excerpts, blank line before QUESTION):

MANUAL EXCERPTS:
[page 5] check the dipstick

[page 9] add oil slowly

QUESTION: how much oil?

PYTHON YOU NEED (the only 3 pieces of syntax involved)
  f-string    ->  f"page {n}"              inserts a variable into text
  loop        ->  [ ... for c in chunks ]  builds a new list from an old one
  join        ->  "\n\n".join(my_list)     glues a list of strings into one string
                                           ("\n" means newline)
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Turn retrieved chunks + a question into one prompt string."""
    # TODO — your code goes here. Suggested order of attack:
    #   a) make a list of strings, one per chunk, each like:  [page 5] check the dipstick
    #      (print it, check it looks right, THEN move on)
    #   b) join that list into one string called `context`, separated by blank lines
    #      (print it again)
    #   c) return the final string: header + context + the question
    return ""


# ---------------------------------------------------------------------------
# The test. Don't edit below this line.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    chunks = [
        {"content": "check the dipstick", "page": 5},
        {"content": "add oil slowly", "page": 9},
    ]
    got = build_prompt("how much oil?", chunks)

    print("=== YOUR OUTPUT ===")
    print(got)
    print("=== END ===")

    expected = (
        "MANUAL EXCERPTS:\n"
        "[page 5] check the dipstick\n\n"
        "[page 9] add oil slowly\n\n"
        "QUESTION: how much oil?"
    )
    if got == expected:
        print("\nPASS - you just wrote a real function from blank.")
    else:
        print("\nFAIL - not matching yet. Compare your output above to the")
        print("target in the docstring at the top of this file.")
        print(f"(your length: {len(got)} chars, target length: {len(expected)} chars)")
