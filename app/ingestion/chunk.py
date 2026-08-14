def chunk_page(text: str, page: int, size: int = 220, overlap: int = 40) -> list[dict]:
    """Split ONE page's text into overlapping word-windows, tagged with the page."""
    words = text.split()          # crude but transparent: split on whitespace
    if not words:                 # blank page -> no chunks (but page number preserved upstream)
        return []

    chunks = []
    step = size - overlap         # how far the window advances each time (220-40=180)
    start = 0
    local_idx = 0                 # chunk position WITHIN this page
    while start < len(words):
        window = words[start:start + size]   # take up to `size` words
        chunks.append({
            "content": " ".join(window),     # rejoin the window into a string
            "page": page,                    # metadata: which page it came from
            "local_index": local_idx,        # metadata: order within the page
            "word_count": len(window),       # for the size-distribution log later
        })
        local_idx += 1
        start += step             # advance by size-overlap => windows OVERLAP by `overlap`
    return chunks


if __name__ == "__main__":
    from app.ingestion.extract import extract_pages
    pages = extract_pages("data/manual.pdf")
    # chunk every page; enumerate gives us the page index (0-based) -> page number = i+1
    all_chunks = [c for i, p in enumerate(pages) for c in chunk_page(p, page=i + 1)]
    counts = [c["word_count"] for c in all_chunks]
    print(f"total chunks: {len(all_chunks)}")
    print(f"word counts -> min {min(counts)}, max {max(counts)}, avg {sum(counts)//len(counts)}")
    print("--- sample chunk ---")
    print(all_chunks[0])