import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .rag_pipeline import RAGPipeline
from .registry import DocumentRegistry
from .qa_parser import is_qa_document
from .document_loader import load_text

app = FastAPI(title="RAG + Perplexity API", version="1.1.0")

# Enable permissive CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = DocumentRegistry()
pipeline = RAGPipeline(registry=registry)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def load_seed_dataset():
    """Auto-load seed Q&A from data/mediclaim_qa.txt if present."""
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        seed_path = os.path.join(base_dir, "data", "mediclaim_qa.txt")
        if os.path.exists(seed_path):
            with open(seed_path, "rb") as f:
                content = f.read()
            # Avoid re-registering the same seed on each hot-reload is out of scope;
            # For simplicity we ingest at startup. Clearing resets.
            pipeline.ingest_qa_text(content, os.path.basename(seed_path))
    except Exception as e:
        print(f"[startup] Failed to load seed dataset: {e}")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        filename = file.filename or "uploaded_file"
        content = await file.read()

        # Basic guard on extension
        name_lower = filename.lower()
        if not (name_lower.endswith(".txt") or name_lower.endswith(".pdf") or name_lower.endswith(".docx")):
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt, .pdf, or .docx")

        # Detect whether this is a Q&A dataset
        text_preview = load_text(content, filename)
        if is_qa_document(text_preview):
            doc_id, count = pipeline.ingest_qa_text(content, filename)
            return {"status": "ok", "filename": filename, "type": "qa", "doc_id": doc_id, "count": count}
        else:
            doc_id, count = pipeline.ingest_file(content, filename)
            return {"status": "ok", "filename": filename, "type": "doc", "doc_id": doc_id, "count": count}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.get("/")
async def root_page() -> HTMLResponse:
    html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Policy Q&A RAG Demo</title>
      <meta name="description" content="Upload policy docs or Q&A sets, ask questions, and get grounded answers." />
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, sans-serif; margin: 2rem; }
        .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
        .row { display: flex; gap: 1rem; align-items: center; }
        textarea { width: 100%; height: 120px; }
        pre { background: #f9fafb; padding: 1rem; border-radius: 6px; white-space: pre-wrap; }
        button { cursor: pointer; padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid #e5e7eb; background: #111827; color: white; }
        button:disabled { opacity: 0.5; }
        .muted { color: #6b7280; font-size: 0.9rem; }
        code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
      </style>
    </head>
    <body>
      <h1>Policy Q&A RAG Demo</h1>
      <p class="muted">This simple UI calls the FastAPI endpoints to upload datasets/documents and ask questions.</p>

      <div class="card">
        <h2>Upload document or Q&A set</h2>
        <input id="fileInput" type="file" />
        <button id="uploadBtn">Upload</button>
        <div id="uploadResult" class="muted"></div>
      </div>

      <div class="card">
        <h2>Ask a question</h2>
        <textarea id="question" placeholder="Type your policy question..."></textarea>
        <div class="row">
          <button id="askBtn">Ask</button>
          <span id="askStatus" class="muted"></span>
        </div>
        <h3>Answer</h3>
        <pre id="answer"></pre>
      </div>

      <div class="card">
        <h2>Uploaded items</h2>
        <button id="refreshBtn">Refresh</button>
        <table>
          <thead><tr><th>Type</th><th>Filename</th><th>Vectors</th><th>Doc ID</th><th>Action</th></tr></thead>
          <tbody id="listBody"></tbody>
        </table>
      </div>

      <script>
        async function refreshList() {
          const res = await fetch('/list');
          const data = await res.json();
          const tbody = document.getElementById('listBody');
          tbody.innerHTML = '';
          data.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${item.type}</td><td>${item.filename}</td><td>${item.count}</td><td><code>${item.doc_id}</code></td><td><button data-id="${item.doc_id}" class="delBtn">Delete</button></td>`;
            tbody.appendChild(tr);
          });
          document.querySelectorAll('.delBtn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
              const id = e.target.getAttribute('data-id');
              if (!confirm('Delete ' + id + '?')) return;
              await fetch('/delete/' + id, { method: 'DELETE' });
              refreshList();
            });
          });
        }

        document.getElementById('refreshBtn').addEventListener('click', refreshList);
        refreshList();

        document.getElementById('uploadBtn').addEventListener('click', async () => {
          const fi = document.getElementById('fileInput');
          if (!fi.files.length) return alert('Choose a file first');
          const fd = new FormData();
          fd.append('file', fi.files[0]);
          const res = await fetch('/upload', { method: 'POST', body: fd });
          const data = await res.json();
          document.getElementById('uploadResult').innerText = JSON.stringify(data, null, 2);
          refreshList();
        });

        document.getElementById('askBtn').addEventListener('click', async () => {
          const q = document.getElementById('question').value.trim();
          if (!q) return alert('Type a question');
          document.getElementById('askStatus').innerText = 'Asking...';
          document.getElementById('answer').innerText = '';
          try {
            const res = await fetch('/query', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ query: q })
            });
            const data = await res.json();
            if (res.ok) {
              document.getElementById('answer').innerText = data.answer;
            } else {
              document.getElementById('answer').innerText = 'Error: ' + (data.detail || res.status);
            }
          } catch (err) {
            document.getElementById('answer').innerText = 'Request failed: ' + err;
          } finally {
            document.getElementById('askStatus').innerText = '';
          }
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must be a non-empty string")
    try:
        answer = pipeline.query(req.query)
        return QueryResponse(answer=answer)
    except RuntimeError as e:
        # Typically missing API key or Perplexity error
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/list")
async def list_items():
    try:
        return registry.list()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List failed: {e}")


@app.delete("/delete/{doc_id}")
async def delete_item(doc_id: str):
    try:
        pipeline.vs.delete_by_doc_id(doc_id)
        registry.delete(doc_id)
        return {"status": "ok", "deleted": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


@app.post("/clear")
async def clear():
    try:
        pipeline.vs.clear()
        # reset registry
        for item in registry.list():
            registry.delete(item["doc_id"])  # simple reset
        return {"status": "ok", "message": "Vector store cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {e}")
