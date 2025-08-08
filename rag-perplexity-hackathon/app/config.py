import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


def get_perplexity_api_key() -> str | None:
    """Return the Perplexity API key from environment variables.

    The key should be set in a .env file (or environment) under PERPLEXITY_API_KEY.
    """
    return os.getenv("PERPLEXITY_API_KEY")
