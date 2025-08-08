from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_pipeline import RAGPipeline

app = FastAPI(title="RAG + Perplexity API", version="1.0.0")

# Enable permissive CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        filename = file.filename or "uploaded_file"
        content = await file.read()

        # Basic guard on extension
        name_lower = filename.lower()
        if not (name_lower.endswith(".txt") or name_lower.endswith(".pdf") or name_lower.endswith(".docx")):
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt, .pdf, or .docx")

        chunks = pipeline.ingest_file(content, filename)
        return {"status": "ok", "filename": filename, "chunks": chunks}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


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


@app.post("/clear")
async def clear():
    try:
        pipeline.vs.clear()
        return {"status": "ok", "message": "Vector store cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {e}")
