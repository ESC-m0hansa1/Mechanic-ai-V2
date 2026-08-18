"""One definition of "this chunk cannot answer anything", used everywhere.

A 900-page manual carries a table of contents, a pictorial index and per-chapter
contents pages. Ingested as text they become chunks of dotted leaders:

    "Side turn signal lights . . . . . . . . . . . . . P. 304"

They pass a word-count filter, they are topically on-point, and they answer
nothing. Dense retrieval mostly ignored them, but BM25 loves them - every rare
term in the manual appears in the index, so a keyword query matches the index
entry as readily as the real section. Measured on the golden set, that dragged
hybrid recall@5 below the dense baseline until these were excluded.

Why dot density and not digit density: the specifications tables are legitimately
digit-heavy ("Engine coolant capacity 1GR-FE 8.4 L") and an earlier
dots+digits+spaces filter threw those real answers away. Dots are the signature
of a leader line and of nothing else in this document.
"""

import re

# Fraction of characters that are literal '.' above which a chunk is index noise.
# Prose sits near 0.02 (sentence ends); leader lines run 0.20+. Measured on this
# corpus: 0.10 flags 89 of 1110 chunks and flags none of the golden-set answers.
DOT_RATIO_LIMIT = 0.10

_LEADER = re.compile(r"\.\s?\.\s?\.")


def dot_ratio(text: str) -> float:
    return text.count(".") / max(len(text), 1)


def is_index_noise(text: str) -> bool:
    """True if the chunk looks like a contents/index line rather than content."""
    return dot_ratio(text) > DOT_RATIO_LIMIT and bool(_LEADER.search(text))
