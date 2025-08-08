# Shim entry so you can run: uvicorn main:app --reload
from app.main import app  # noqa: F401
