import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from a .env file if present
load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_perplexity_api_key() -> str | None:
    """Return the Perplexity API key from environment variables."""
    return os.getenv("PERPLEXITY_API_KEY")


def get_model_name() -> str:
    """Model name for Perplexity chat completions."""
    return os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-large-32k-chat")


def get_chroma_dir() -> str:
    """Directory for persistent ChromaDB storage."""
    default_dir = os.path.join(BASE_DIR, "db")
    return os.getenv("CHROMA_DB_DIR", default_dir)


def get_qa_confidence_threshold() -> float:
    """Similarity threshold (0-1) to trust a QA hit and return the saved answer."""
    try:
        return float(os.getenv("QA_CONFIDENCE_THRESHOLD", "0.85"))
    except ValueError:
        return 0.85


def get_system_prompt() -> str:
    """System prompt to enforce strict, policy-grounded answers."""
    return os.getenv(
        "SYSTEM_PROMPT",
        (
            "You are a cautious assistant for insurance policy Q&A. "
            "Use ONLY the provided context (policy excerpts and approved Q&A). "
            "If the answer is not fully supported by the context, reply exactly with: Not in policy"
        ),
    )
