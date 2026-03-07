import os
import socket
import uuid
from typing import Dict, List, Tuple

try:
    try:
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        from langchain.chains.combine_documents import create_stuff_documents_chain

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
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from document_parsers import DocumentParser
    from intelligence_layer import IntelligenceLayer
except ModuleNotFoundError as exc:
    missing = exc.name or "unknown module"
    raise ModuleNotFoundError(
        f"Missing dependency '{missing}'. Install project dependencies in your active interpreter: "
        f"`python -m pip install -r requirements.txt`"
    ) from exc


class RAGEngine:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "categorical_rag",
        google_api_key: str | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        local_embedding_model: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        raw_key = google_api_key or os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
        self.google_api_key = (raw_key or "").strip().strip("'").strip('"')
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_provider = (embedding_provider or os.getenv("EMBEDDING_PROVIDER", "local")).strip().lower()
        self.embedding_model = embedding_model or os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.local_embedding_model = local_embedding_model or os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.llm_model = llm_model or os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")
        self.enable_intelligence_synthesis = (
            os.getenv("ENABLE_INTELLIGENCE_SYNTHESIS", "true").strip().lower() == "true"
        )

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

    def _build_embeddings(self):
        if self.embedding_provider == "google":
            return GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=self.google_api_key,
            )
        if self.embedding_provider == "local":
            return HuggingFaceEmbeddings(
                model_name=self.local_embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        raise ValueError("Invalid EMBEDDING_PROVIDER. Use 'local' or 'google'.")

    def _ensure_api_key(self) -> None:
        if not self.google_api_key or self.google_api_key == "YOUR_GOOGLE_API_KEY":
            raise ValueError("GOOGLE_API_KEY is not set. Replace placeholder with a valid API key.")

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

    def _build_qa_chain(self):
        self._ensure_api_key()
        llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            google_api_key=self.google_api_key,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_template(
            """
You are a production RAG assistant.
Use only the retrieved context to answer.

When answering:
1. Separate categorical facts (exact labels, categories, codes, tags, enumerations) from descriptive context.
2. Prefer exact categorical values when there is a mismatch.
3. If context is insufficient, say what is missing.

Context:
{context}

Question:
{input}

Answer with:
- Categorical Facts:
- Descriptive Context:
- Final Answer:
"""
        )
        return create_stuff_documents_chain(llm=llm, prompt=prompt)

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
            self._ensure_api_key()
            llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.google_api_key,
                temperature=0.1,
            )
            source_lines = []
            for idx, src in enumerate(sources[:12], start=1):
                source_lines.append(
                    f"{idx}. {src.get('source', 'unknown')} | {src.get('type', 'unknown')} | {src.get('snippet', '')}"
                )
            source_text = "\n".join(source_lines) if source_lines else "No source snippets provided."

            prompt = ChatPromptTemplate.from_template(
                """
You are an industrial operations intelligence analyst.
Transform the raw analytics output into a clear, practical response for end users.

Hard rules:
1. Use only facts present in RAW_ANALYTICS and SOURCE_SNIPPETS.
2. Do not invent metrics, dates, machines, jobs, statuses, or calculations.
3. If data is missing, state it explicitly and recommend what data should be captured.
4. Recommendations must be directly tied to evidence.
5. If RAW_ANALYTICS includes `ANALYTICS_CONTEXT_JSON`, parse it and prioritize `scoped_summary` for direct answers.
6. Use `global_summary` for benchmark/comparison context.
7. When user asks a count/total, prefer exact numeric fields from the JSON.

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
            )
            messages = prompt.format_messages(
                question=question,
                raw_answer=raw_answer,
                source_text=source_text,
            )
            response = llm.invoke(messages)
            text = getattr(response, "content", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
            return raw_answer
        except Exception:
            return raw_answer

    def query(
        self,
        question: str,
        k: int = 6,
        enable_intelligence_synthesis: bool | None = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        synthesis_enabled = (
            self.enable_intelligence_synthesis
            if enable_intelligence_synthesis is None
            else bool(enable_intelligence_synthesis)
        )
        ql = question.lower()
        if any(token in ql for token in ["all jobs", "list jobs", "what are all the jobs", "which jobs"]):
            synthesis_enabled = False
        structured = self.intelligence.answer(question)
        if structured is not None:
            answer, sources = structured
            polished = self._synthesize_intelligence_answer(
                question,
                answer,
                sources,
                synthesis_enabled,
            )
            return polished, sources

        retriever = self._build_hybrid_retriever(k=k)
        chain = self._build_qa_chain()
        retrieved_docs = retriever.invoke(question)
        try:
            answer = chain.invoke({"input": question, "context": retrieved_docs})
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
        return answer, sources

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
