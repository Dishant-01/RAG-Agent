# Categorical RAG (FastAPI + Streamlit + LangChain)

Production-style RAG + analytics assistant for plant/quality leadership use cases.

## Stack
- Backend: FastAPI
- Frontend: Streamlit
- LLM: Gemini (`gemini-2.5-flash` by default, configurable)
- Embeddings: Local SentenceTransformers (default) or Google embeddings
- Vector DB: Chroma (persistent)
- Retrieval: Hybrid BM25 + Vector
- Data types: PDF, DOCX, XML

## Project Structure
- `main.py`: FastAPI app and endpoints
- `app.py`: Streamlit UI
- `rag_engine.py`: Orchestration (ingest, retrieval, synthesis)
- `document_parsers.py`: PDF/XML/DOCX parsing + row reconstruction
- `intelligence_layer.py`: Full-dataset analytics context generation
- `tests/test_intelligence_layer.py`: Unit tests

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Set required values in `.env`:
- `GOOGLE_API_KEY` (required for LLM synthesis/fallback)
- `EMBEDDING_PROVIDER=local` (recommended)
- `LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

## Run
Backend:
```powershell
python -m uvicorn main:app --reload
```

Frontend:
```powershell
streamlit run app.py
```

## API Endpoints
- `GET /health`
- `POST /upload`
- `POST /query`
- `GET /files`
- `DELETE /files?source_name=<filename>`
- `GET /stats`

## Tests
```powershell
python -m unittest tests.test_intelligence_layer -v
```

## GitHub Push Ready Checklist
1. Ensure `.env` is not committed (`.gitignore` already excludes it).
2. Ensure `chroma_db/` is not committed (`.gitignore` already excludes it).
3. Initialize git (if needed):
```powershell
git init
git add .
git commit -m "Initial production-ready Categorical RAG app"
```
4. Add remote and push:
```powershell
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```
