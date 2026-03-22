import os
import socket
import uuid
import logging
from typing import Dict, List, Tuple

try:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain.retrievers import EnsembleRetriever

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from google import genai
    from document_parsers import DocumentParser
    from intelligence_layer import IntelligenceLayer
except ModuleNotFoundError as exc:
    missing = exc.name or "unknown module"
    raise ModuleNotFoundError(
        f"Missing dependency '{missing}'. Install project dependencies in your active interpreter: "
        f"`python -m pip install -r requirements.txt`"
    ) from exc


class GeminiGenAIEmbeddings(Embeddings):
    """LangChain-compatible embeddings wrapper using google-genai API key auth."""

    def __init__(self, client: genai.Client, model: str) -> None:
        self.client = client
        self.model = model

    @staticmethod
    def _extract_vectors(response) -> List[List[float]]:
        if hasattr(response, "embeddings") and response.embeddings:
            vectors = []
            for emb in response.embeddings:
                values = getattr(emb, "values", None)
                if values is not None:
                    vectors.append([float(v) for v in values])
            if vectors:
                return vectors
        if hasattr(response, "embedding") and response.embedding:
            values = getattr(response.embedding, "values", None)
            if values is not None:
                return [[float(v) for v in values]]
        raise ValueError("Gemini embedding response did not contain vector values.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return self._extract_vectors(response)

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []


class RAGEngine:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "categorical_rag",
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        local_embedding_model: str | None = None,
        llm_model: str | None = None,
        gemini_api_key: str | None = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_provider = (embedding_provider or os.getenv("EMBEDDING_PROVIDER", "local")).strip().lower()
        self.embedding_model = embedding_model or os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.local_embedding_model = local_embedding_model or os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.llm_model = llm_model or os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")
        self.gemini_api_key = (
            gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        ).strip().strip("'").strip('"')
        self.enable_intelligence_synthesis = (
            os.getenv("ENABLE_INTELLIGENCE_SYNTHESIS", "true").strip().lower() == "true"
        )
        self.llm_strict_errors = os.getenv("LLM_STRICT_ERRORS", "false").strip().lower() == "true"
        self.logger = logging.getLogger(__name__)

        self._genai_client: genai.Client | None = None
        self.embeddings = self._build_embeddings()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.parser = DocumentParser()
        self.intelligence = IntelligenceLayer(self.vectorstore)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._session_scopes: Dict[str, Dict[str, object]] = {}

    def _build_embeddings(self):
        if self.embedding_provider == "google":
            client = self._get_genai_client()
            return GeminiGenAIEmbeddings(
                client=client,
                model=self.embedding_model,
            )
        if self.embedding_provider == "local":
            return HuggingFaceEmbeddings(
                model_name=self.local_embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        raise ValueError("Invalid EMBEDDING_PROVIDER. Use 'local' or 'google'.")

    def _ensure_api_key(self) -> None:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

    def _get_genai_client(self) -> genai.Client:
        if self._genai_client is not None:
            return self._genai_client
        self._ensure_api_key()
        self._genai_client = genai.Client(api_key=self.gemini_api_key)
        return self._genai_client

    @staticmethod
    def _extract_generated_text(response) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not parts:
                continue
            chunks = [getattr(part, "text", "") for part in parts if getattr(part, "text", "")]
            if chunks:
                return "\n".join(chunks).strip()
        return ""

    def _generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        client = self._get_genai_client()
        response = client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
            config={"temperature": temperature},
        )
        text = self._extract_generated_text(response)
        if not text:
            raise ValueError("Gemini returned an empty response.")
        return text

    def ingest_file(self, file_path: str, source_name: str) -> int:
        docs = self.parser.load_documents(file_path, source_name)
        if not docs:
            return 0

        chunks: List[Document] = []
        for doc in docs:
            if doc.metadata.get("doc_kind") == "categorical_row":
                chunks.append(doc)
            else:
                chunks.extend(self.splitter.split_documents([doc]))

        ids = [str(uuid.uuid4()) for _ in chunks]
        try:
            self.vectorstore.delete(where={"source": source_name})
        except Exception:
            pass

        try:
            self.vectorstore.add_documents(chunks, ids=ids)
        except socket.gaierror as exc:
            raise ConnectionError("Cannot reach embedding endpoint (DNS/network issue).") from exc
        except OSError as exc:
            if "getaddrinfo failed" in str(exc).lower():
                raise ConnectionError("Cannot reach embedding endpoint (DNS/network issue).") from exc
            raise
        return len(chunks)

    def _build_hybrid_retriever(self, k: int = 6) -> EnsembleRetriever:
        store_dump = self.vectorstore.get(include=["documents", "metadatas"])
        raw_docs = store_dump.get("documents", [])
        raw_meta = store_dump.get("metadatas", [])
        if not raw_docs:
            raise ValueError("Vector database is empty. Upload at least one file first.")
        docs = [Document(page_content=raw_docs[i], metadata=raw_meta[i] or {}) for i in range(len(raw_docs))]
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = k
        vector = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return EnsembleRetriever(retrievers=[bm25, vector], weights=[0.45, 0.55])

    def _build_retrieval_prompt(self, question: str, retrieved_docs: List[Document]) -> str:
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content[:2200]}"
            for doc in retrieved_docs[:8]
        )
        return f"""
You are a production RAG assistant.
Use only the retrieved context to answer. Do not invent facts.

Context:
{context or "No context provided."}

Question:
{question}

Answer with:
Direct Answer:
- concise answer

Key Evidence:
- 3-6 concrete values/facts from context

Recommendations:
- 2-5 practical actions tied to evidence

Assumptions/Limitations:
- explicit caveats and missing-data notes
"""

    def _synthesize_intelligence_answer(
        self,
        question: str,
        raw_answer: str,
        sources: List[Dict[str, str]],
        enable_synthesis: bool,
    ) -> str:
        if not enable_synthesis:
            return raw_answer

        try:
            source_lines = []
            for idx, src in enumerate(sources[:12], start=1):
                source_lines.append(
                    f"{idx}. {src.get('source', 'unknown')} | {src.get('type', 'unknown')} | {src.get('snippet', '')}"
                )
            source_text = "\n".join(source_lines) if source_lines else "No source snippets provided."

            prompt = f"""
You are a data intelligence analyst.
Transform the raw analytics output into a clear, practical response for end users.

Hard rules:
1. Use only facts present in RAW_ANALYTICS and SOURCE_SNIPPETS.
2. Do not invent metrics, dimensions, dates, categories, statuses, or calculations.
3. If data is missing, state it explicitly and recommend what data should be captured.
4. Recommendations must be directly tied to evidence.
5. If RAW_ANALYTICS includes `ANALYTICS_CONTEXT_JSON`, parse it and use `scoped_summary`, `global_summary`, `universal_analytics`, and `dynamic_metrics` as the primary evidence.
6. Prefer computed metrics over textual snippets whenever both exist.
7. For counts/rates/trends/comparisons, cite exact values and their coverage (rows/records used).
8. For "entire dataset"/"overall" questions, prioritize `universal_analytics.numeric_field_coverage`, `high_coverage_numeric_fields`, and `dataset_wide_composites` over sparse low-coverage fields.

Output format (exact headings):
Direct Answer:
- concise answer to user's question

Key Evidence:
- 3-6 bullets with concrete values

Recommendations:
- 2-5 practical actions

Assumptions/Limitations:
- explicit caveats and missing-data notes

USER_QUESTION:
{question}

RAW_ANALYTICS:
{raw_answer}

SOURCE_SNIPPETS:
{source_text}
"""
            return self._generate_text(prompt, temperature=0.1)
        except Exception as exc:
            self.logger.exception("Intelligence synthesis failed: %s", exc)
            if self.llm_strict_errors:
                raise
            return raw_answer

    def _synthesize_retrieval_answer(
        self,
        question: str,
        raw_answer: str,
        sources: List[Dict[str, str]],
        enable_synthesis: bool,
    ) -> str:
        if not enable_synthesis:
            return raw_answer

        try:
            source_lines = []
            for idx, src in enumerate(sources[:12], start=1):
                source_lines.append(
                    f"{idx}. {src.get('source', 'unknown')} | {src.get('type', 'unknown')} | {src.get('snippet', '')}"
                )
            source_text = "\n".join(source_lines) if source_lines else "No source snippets provided."

            prompt = f"""
You are a data intelligence analyst.
Reformat RAW_ANSWER into a strict user-facing structure. Use only RAW_ANSWER and SOURCE_SNIPPETS.
Do not invent facts, numbers, dates, or categories.

Output format (exact headings):
Direct Answer:
- concise answer to user's question

Key Evidence:
- 3-6 bullets with concrete values from RAW_ANSWER or SOURCE_SNIPPETS

Recommendations:
- 2-5 practical actions tied to evidence

Assumptions/Limitations:
- explicit caveats and missing-data notes

USER_QUESTION:
{question}

RAW_ANSWER:
{raw_answer}

SOURCE_SNIPPETS:
{source_text}
"""
            return self._generate_text(prompt, temperature=0.1)
        except Exception as exc:
            self.logger.exception("Retrieval synthesis failed: %s", exc)
            if self.llm_strict_errors:
                raise
            return raw_answer

    def query(
        self,
        question: str,
        k: int = 6,
        enable_intelligence_synthesis: bool | None = None,
        session_id: str | None = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        synthesis_enabled = (
            self.enable_intelligence_synthesis
            if enable_intelligence_synthesis is None
            else bool(enable_intelligence_synthesis)
        )
        prior_scope = self._session_scopes.get(session_id) if session_id else None
        structured = self.intelligence.answer(question, prior_scope=prior_scope)
        if structured is not None:
            answer, sources, scope_state = structured
            if session_id is not None:
                if IntelligenceLayer._scope_has_filters(scope_state):
                    self._session_scopes[session_id] = scope_state
                else:
                    self._session_scopes.pop(session_id, None)
            polished = self._synthesize_intelligence_answer(
                question,
                answer,
                sources,
                synthesis_enabled,
            )
            return polished, sources

        retriever = self._build_hybrid_retriever(k=k)
        retrieved_docs = retriever.invoke(question)
        try:
            prompt = self._build_retrieval_prompt(question, retrieved_docs)
            answer = self._generate_text(prompt, temperature=0.2)
        except socket.gaierror as exc:
            raise ConnectionError("Cannot reach Gemini endpoint (DNS/network issue).") from exc
        except OSError as exc:
            if "getaddrinfo failed" in str(exc).lower():
                raise ConnectionError("Cannot reach Gemini endpoint (DNS/network issue).") from exc
            raise

        sources: List[Dict[str, str]] = []
        for doc in retrieved_docs:
            sources.append(
                {
                    "source": str(doc.metadata.get("source", "unknown")),
                    "type": str(doc.metadata.get("type", "unknown")),
                    "snippet": doc.page_content[:220].replace("\n", " "),
                }
            )
        polished = self._synthesize_retrieval_answer(
            question=question,
            raw_answer=answer,
            sources=sources,
            enable_synthesis=synthesis_enabled,
        )
        return polished, sources

    def machine_row_counts(self) -> Dict[str, int]:
        return self.intelligence.machine_row_counts()

    def list_indexed_sources(self) -> List[Dict[str, object]]:
        dump = self.vectorstore.get(include=["metadatas"])
        metadatas = dump.get("metadatas", [])
        summary: Dict[str, Dict[str, object]] = {}
        for meta in metadatas:
            item = meta or {}
            source = str(item.get("source", "")).strip()
            if not source:
                continue
            if source not in summary:
                summary[source] = {"source": source, "chunks": 0, "types": set()}
            summary[source]["chunks"] = int(summary[source]["chunks"]) + 1
            source_type = str(item.get("type", "")).strip()
            if source_type:
                summary[source]["types"].add(source_type)

        out: List[Dict[str, object]] = []
        for src in sorted(summary.keys()):
            row = summary[src]
            out.append(
                {
                    "source": row["source"],
                    "chunks": row["chunks"],
                    "types": sorted(list(row["types"])),
                }
            )
        return out

    def delete_source(self, source_name: str) -> int:
        source_name = (source_name or "").strip()
        if not source_name:
            raise ValueError("source_name is required")
        files = self.list_indexed_sources()
        existing = next((f for f in files if f["source"] == source_name), None)
        if not existing:
            return 0
        deleted_chunks = int(existing["chunks"])
        self.vectorstore.delete(where={"source": source_name})
        return deleted_chunks

    def schema_summary(self) -> Dict[str, object]:
        return self.intelligence.schema_summary()
