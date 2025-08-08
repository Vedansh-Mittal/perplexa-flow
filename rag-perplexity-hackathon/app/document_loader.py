import io
from typing import Literal

import pdfplumber
from docx import Document


def load_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from supported file types (.txt, .pdf, .docx).

    Raises ValueError for unsupported or unreadable files.
    """
    name_lower = filename.lower()

    if name_lower.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            raise ValueError(f"Failed to decode TXT file: {e}")

    if name_lower.endswith(".pdf"):
        try:
            text_parts: list[str] = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts).strip()
        except Exception as e:
            raise ValueError(f"Failed to read PDF file: {e}")

    if name_lower.endswith(".docx"):
        try:
            doc = Document(io.BytesIO(file_bytes))
            paras = [p.text for p in doc.paragraphs]
            return "\n".join(paras).strip()
        except Exception as e:
            raise ValueError(f"Failed to read DOCX file: {e}")

    raise ValueError("Unsupported file type. Please upload .txt, .pdf, or .docx")
