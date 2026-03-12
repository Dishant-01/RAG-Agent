import os
import shutil
import tempfile
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import RAGEngine

load_dotenv()

app = FastAPI(title="Categorical RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine(
    persist_directory=os.getenv("CHROMA_PERSIST_DIR", "chroma_db"),
    collection_name=os.getenv("CHROMA_COLLECTION", "categorical_rag"),
    google_api_key=os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY"),
    embedding_model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
    embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
    local_embedding_model=os.getenv(
        "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ),
    llm_model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash"),
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    enable_intelligence_synthesis: bool | None = Field(default=None)
    session_id: str | None = Field(default=None)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    ingested = []
    failed = []
    total_chunks = 0

    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in {".pdf", ".xml", ".docx"}:
            failed.append({"file": file.filename, "reason": f"Unsupported type: {ext}"})
            continue

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                temp_path = tmp.name
                shutil.copyfileobj(file.file, tmp)

            chunk_count = rag_engine.ingest_file(temp_path, file.filename or "uploaded_file")
            total_chunks += chunk_count
            ingested.append({"file": file.filename, "chunks": chunk_count})
        except Exception as exc:
            failed.append({"file": file.filename, "reason": str(exc)})
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            file.file.close()

    if not ingested and failed:
        raise HTTPException(status_code=400, detail={"ingested": ingested, "failed": failed})

    return {"ingested": ingested, "failed": failed, "total_chunks": total_chunks}


@app.post("/query")
def query_documents(payload: QueryRequest):
    try:
        answer, sources = rag_engine.query(
            question=payload.question,
            enable_intelligence_synthesis=payload.enable_intelligence_synthesis,
            session_id=payload.session_id,
        )
        return {"answer": answer, "sources": sources}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/stats")
def index_stats():
    try:
        counts = rag_engine.machine_row_counts()
        return {
            "machine_counts": counts,
            "total_machine_rows": sum(counts.values()),
            "machine_count_entries": len(counts),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/files")
def list_files():
    try:
        files = rag_engine.list_indexed_sources()
        return {"files": files, "count": len(files)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/files")
def delete_file(source_name: str = Query(..., min_length=1)):
    try:
        deleted_chunks = rag_engine.delete_source(source_name)
        return {
            "source": source_name,
            "deleted_chunks": deleted_chunks,
            "deleted": deleted_chunks > 0,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
