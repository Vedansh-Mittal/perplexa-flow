import re
from typing import List, Dict

_QA_REGEX = re.compile(
    r"(?ims)^\s*(?:Q(?:uestion)?[:\-]\s*)(.+?)\s*(?:\n|\r\n)\s*(?:A(?:nswer)?[:\-]\s*)(.+?)(?=(?:\n\s*Q(?:uestion)?[:\-]\s*)|\Z)",
)


def parse_qa_pairs(text: str) -> List[Dict[str, str]]:
    """Parse Q&A pairs in formats like:

    Q: What is X?
    A: Y

    or

    Question: ...
    Answer: ...
    """
    pairs: List[Dict[str, str]] = []
    if not text:
        return pairs

    for m in _QA_REGEX.finditer(text):
        q = m.group(1).strip()
        a = m.group(2).strip()
        if q and a:
            pairs.append({"question": q, "answer": a})
    return pairs


def is_qa_document(text: str) -> bool:
    return len(parse_qa_pairs(text)) >= 3
