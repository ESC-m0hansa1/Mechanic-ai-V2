"""One-off discovery pass used to build golden_set.json (see inspect_chunks.py).

Kept in the repo because "how did you pick your golden questions" needs an
answer with an artefact behind it. Two filters matter here:

* the manual's table of contents survived ingestion as dotted-leader chunks
  ("Engine oil . . . . . P. 560"), which look relevant to a keyword scan but
  answer nothing - so anything mostly dots/digits is dropped from candidates;
* the PDF uses private-use glyphs for its icons, which a Windows cp1252 console
  cannot print, so text is forced to ASCII before display, not before storage.

    python -m eval.discover
"""

import re

from app.core.db import get_connection

TERMS = [
    "engine oil", "tire pressure", "spare tire", "jack", "wiper blade",
    "brake fluid", "engine coolant", "battery", "fuse", "headlight",
    "seat belt", "child restraint", "air conditioning", "Bluetooth",
    "fuel tank cap", "towing", "differential lock", "warning light",
    "overheat", "parking brake", "washer fluid", "floor mat", "oil filter",
    "tire rotation", "break-in", "immobilizer", "cruise control", "airbag",
]

# A chunk is TOC/index noise if this much of it is dots, digits and spaces.
NOISE_RATIO = 0.35


def is_noise(text: str) -> bool:
    filler = sum(ch in ". \t\n0123456789" for ch in text)
    return filler / max(len(text), 1) > NOISE_RATIO


def ascii_only(text: str) -> str:
    """Squash whitespace and drop unprintable PDF icon glyphs for display."""
    return re.sub(r"\s+", " ", text).encode("ascii", "ignore").decode().strip()


def main(per_term: int = 3, width: int = 130) -> None:
    sql = """
        SELECT id, metadata->>'page', content
        FROM chunks
        WHERE content ILIKE %s
        ORDER BY id;
    """
    with get_connection() as conn, conn.cursor() as cur:
        for term in TERMS:
            cur.execute(sql, (f"%{term}%",))
            kept = [r for r in cur.fetchall() if not is_noise(r[2])][:per_term]
            print(f"\n### {term}  ({len(kept)} shown)")
            for cid, page, content in kept:
                print(f"  {cid:<5} p{page:<5} {ascii_only(content)[:width]}")


if __name__ == "__main__":
    main()
