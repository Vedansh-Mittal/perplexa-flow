# RAG + Perplexity Hackathon Backend

This folder contains a complete FastAPI-based Retrieval Augmented Generation (RAG) pipeline that uses:
- ChromaDB (local persistence) for vector storage
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- Perplexity AI for generation

## Quickstart

1) Python 3.11+

2) Setup environment:
- Copy .env.example to .env and set PERPLEXITY_API_KEY
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

## Endpoints

- POST /upload (multipart/form-data)
  - file: .txt, .pdf, or .docx
  - Extracts text, chunks (500 chars, 50 overlap), embeds and stores in ChromaDB

- POST /query (application/json)
```json
{
  "query": "Your question"
}
```
  - Retrieves top 3 chunks and asks Perplexity (model: llama-3.1-sonar-large-32k-chat)

- POST /clear
  - Clears the ChromaDB collection

## cURL examples

```bash
# Upload
echo "Hello world from a text file" > sample.txt
curl -F "file=@sample.txt" http://127.0.0.1:8000/upload

# Query
curl -H "Content-Type: application/json" \
     -d '{"query":"What does the text say?"}' \
     http://127.0.0.1:8000/query

# Clear
curl -X POST http://127.0.0.1:8000/clear
```

## Docker

```bash
cd rag-perplexity-hackathon
cp .env.example .env
docker build -t rag-perplexity .
docker run -p 8000:8000 --env-file .env -v $(pwd)/db:/app/db rag-perplexity
```

Or with docker-compose:
```bash
docker compose up --build
```

## Notes
- Vector DB persists under ./db
- Ensure your PERPLEXITY_API_KEY is valid and has quota
- The model used: sentence-transformers/all-MiniLM-L6-v2 and Perplexity llama-3.1-sonar-large-32k-chat
