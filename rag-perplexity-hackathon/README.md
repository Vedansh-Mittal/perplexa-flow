# RAG + Perplexity Hackathon Backend (Enhanced)

This folder contains a FastAPI-based Retrieval Augmented Generation (RAG) pipeline with:
- ChromaDB (disk persistence) for vector storage
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- Perplexity AI for generation
- Hybrid search that prioritizes your approved Q&A dataset

## What's New
- Upload and parse custom Q&A datasets (TXT/PDF/DOCX) in `Q:` / `A:` format
- Hybrid retrieval: searches your Q&A and uploaded document chunks together
- High-confidence QA hits return the exact saved answer without LLM calls
- Management endpoints: list, delete, clear
- Persistent storage folder configurable via `.env`
- Seed loader: auto-ingest `data/mediclaim_qa.txt` on startup (if present)
- Minimal web UI at `/` for quick manual testing

---

## Quickstart

1) Python 3.11+

2) Setup environment:
- Copy `.env.example` to `.env` and set `PERPLEXITY_API_KEY`
- Optionally adjust `QA_CONFIDENCE_THRESHOLD` and `CHROMA_DB_DIR`
- Create a virtualenv and install deps

```bash
cd rag-perplexity-hackathon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit .env
```

3) Run the API
```bash
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000

---

## Endpoints

- GET /               → Minimal UI to upload and ask questions
- GET /health         → Healthcheck
- POST /upload        → Upload a document (.txt/.pdf/.docx) or Q&A dataset
- POST /query         → Ask a question; may return saved QA or Perplexity result
- GET /list           → List all uploaded items and their types
- DELETE /delete/{id} → Delete a specific uploaded item (vectors and registry)
- POST /clear         → Wipe the vector store and registry

### Upload
- If the uploaded file is recognized as a Q&A dataset (3+ parsed pairs), it's stored as type `qa`.
- Otherwise, it's chunked (500 chars, 50 overlap) and stored as type `doc`.

Q&A format examples:
```
Q: What is X?
A: Y

Question: What is X?
Answer: Y
```

### Query flow
1. Retrieve top results across both QA and doc chunks.
2. If the best QA hit has similarity ≥ `QA_CONFIDENCE_THRESHOLD`, return its saved answer.
3. Otherwise, build a context from top QA pairs and doc chunks and query Perplexity.
4. System prompt enforces: use ONLY the provided context. If not covered, reply exactly `Not in policy`.

---

## Environment

- `PERPLEXITY_API_KEY`            → Your Perplexity key
- `PERPLEXITY_MODEL`              → Defaults to `llama-3.1-sonar-large-32k-chat`
- `QA_CONFIDENCE_THRESHOLD`       → Defaults to `0.85`
- `CHROMA_DB_DIR`                 → Persistent db folder, defaults to `./db` inside this folder
- `SYSTEM_PROMPT`                 → Optional custom system message

---

## cURL examples

```bash
# Upload Q&A dataset
curl -F "file=@tests/data/sample_qa.txt" http://127.0.0.1:8000/upload

# Upload generic document
echo "Hello world" > sample.txt
curl -F "file=@sample.txt" http://127.0.0.1:8000/upload

# Ask a question
curl -H "Content-Type: application/json" \
     -d '{"query":"What is Foo?"}' \
     http://127.0.0.1:8000/query

# List
curl http://127.0.0.1:8000/list

# Delete
curl -X DELETE http://127.0.0.1:8000/delete/<doc_id>

# Clear
curl -X POST http://127.0.0.1:8000/clear
```

---

## Docker

```bash
cd rag-perplexity-hackathon
cp .env.example .env
docker build -t rag-perplexity .
# persist the DB folder
docker run -p 8000:8000 --env-file .env -v $(pwd)/db:/app/db rag-perplexity
```

Or with docker-compose:
```bash
docker compose up --build
```

---

## Notes
- Vector DB persists under `./db` (or `CHROMA_DB_DIR` if set)
- Ensure your `PERPLEXITY_API_KEY` is valid and has quota
- The model used: `sentence-transformers/all-MiniLM-L6-v2` and Perplexity (default model above)
- Seed dataset: place `data/mediclaim_qa.txt` to auto-load on startup
