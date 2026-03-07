import os
from typing import List, Tuple

import requests
import streamlit as st

st.set_page_config(page_title="Categorical RAG Chat", page_icon="📚", layout="wide")

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def upload_to_api(base_url: str, files) -> Tuple[bool, dict]:
    file_payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
    response = requests.post(f"{base_url}/upload", files=file_payload, timeout=180)
    if response.ok:
        return True, response.json()
    return False, response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": response.text}


def query_api(
    base_url: str,
    question: str,
    enable_intelligence_synthesis: bool | None = None,
) -> Tuple[bool, dict]:
    payload = {
        "question": question,
        "enable_intelligence_synthesis": enable_intelligence_synthesis,
    }
    response = requests.post(f"{base_url}/query", json=payload, timeout=180)
    if response.ok:
        return True, response.json()
    return False, response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": response.text}


def list_files_api(base_url: str) -> Tuple[bool, dict]:
    response = requests.get(f"{base_url}/files", timeout=60)
    if response.ok:
        return True, response.json()
    return False, response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": response.text}


def delete_file_api(base_url: str, source_name: str) -> Tuple[bool, dict]:
    response = requests.delete(f"{base_url}/files", params={"source_name": source_name}, timeout=60)
    if response.ok:
        return True, response.json()
    return False, response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": response.text}


st.title("Categorical Assistant")
st.caption("Analyze your documents with ease. Upload PDFs, DOCX, or XML files and ask questions to get insights based on their content.")

with st.sidebar:
    backend_url = DEFAULT_BACKEND_URL.rstrip("/")
    enable_intelligence_synthesis = True
    with st.expander("Advanced Retrieval Settings", expanded=False):
        enable_intelligence_synthesis = st.toggle(
            "Enable Intelligence Synthesis",
            value=True,
            help="Reformats deterministic analytics into clearer, recommendation-oriented answers.",
        )
    st.divider()
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, XML",
        type=["pdf", "docx", "xml"],
        accept_multiple_files=True,
    )
    if st.button("Index Files", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Select at least one file.")
        else:
            ok, payload = upload_to_api(backend_url, uploaded_files)
            if ok:
                st.success(f"Indexed successfully. Total chunks: {payload.get('total_chunks', 0)}")
                ingested = payload.get("ingested", [])
                failed = payload.get("failed", [])
                if ingested:
                    st.write("Ingested:")
                    st.json(ingested)
                if failed:
                    st.write("Failed:")
                    st.json(failed)
            else:
                st.error("Upload failed.")
                st.json(payload)

    st.divider()
    st.header("Indexed Files")
    ok_files, payload_files = list_files_api(backend_url)
    if ok_files:
        files = payload_files.get("files", [])
        if not files:
            st.caption("No indexed files yet.")
        else:
            for item in files:
                source = str(item.get("source", "unknown"))
                chunks = item.get("chunks", 0)
                types = ", ".join(item.get("types", []))
                col1, col2 = st.columns([5, 2])
                with col1:
                    st.caption(f"{source} | chunks: {chunks} | types: {types}")
                with col2:
                    if st.button("Delete", key=f"delete_{source}", use_container_width=True):
                        ok_del, payload_del = delete_file_api(backend_url, source)
                        if ok_del:
                            st.success(f"Deleted: {source}")
                            st.rerun()
                        else:
                            st.error(f"Delete failed: {source}")
                            st.json(payload_del)
    else:
        st.error("Failed to fetch indexed files.")
        st.json(payload_files)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.json(msg["sources"])

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Querying..."):
            ok, payload = query_api(
                backend_url,
                question,
                enable_intelligence_synthesis=enable_intelligence_synthesis,
            )
            if ok:
                answer = payload.get("answer", "No answer generated.")
                sources: List[dict] = payload.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        st.json(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            else:
                st.error("Query failed.")
                st.json(payload)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Query failed: {payload}"}
                )
