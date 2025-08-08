from typing import List


def chunk_text(text: str, max_len: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping character chunks.

    - max_len: maximum characters per chunk
    - overlap: number of characters overlapping between consecutive chunks
    """
    if not text:
        return []

    if overlap >= max_len:
        raise ValueError("overlap must be smaller than max_len")

    chunks: List[str] = []
    start = 0
    step = max_len - overlap

    while start < len(text):
        end = min(start + max_len, len(text))

        # Try to break at a whitespace before the hard boundary for nicer chunks
        if end < len(text):
            window = text[start:end]
            last_space = window.rfind(" ")
            if last_space != -1 and (end - (start + last_space)) <= 60:
                end = start + last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0

    return chunks
