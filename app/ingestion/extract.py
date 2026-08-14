from pypdf import PdfReader  # the PDF-reading library we chose


def extract_pages(pdf_path: str) -> list[str]:
    """Return a list where each item is one page's text."""
    reader = PdfReader(pdf_path)          # opens + parses the PDF file
    pages = []                            # we'll collect page texts here
    for page in reader.pages:             # reader.pages is iterable, one per page
        text = page.extract_text() or ""  # extract_text() can return None -> use ""
        pages.append(text)                # keep even empty pages to preserve numbering
    return pages


# `__name__ == "__main__"` is True only when this file is run directly
# (python -m app.ingestion.extract), NOT when it's imported elsewhere.
if __name__ == "__main__":
    pages = extract_pages("data/manual.pdf")
    print(f"pages: {len(pages)}")                 # how many pages we read
    # find the first non-empty page and preview it, so we SEE real text
    first = next((p for p in pages if p.strip()), "")
    print("--- first non-empty page (500 chars) ---")
    print(first[:500])