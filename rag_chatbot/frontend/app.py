import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Self-Hosted RAG Chatbot", layout="wide")
st.title("📚 Self-Hosted RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: document upload ---
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])
    if uploaded_file and st.button("Ingest document"):
        with st.spinner("Chunking and embedding..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(f"{API_URL}/ingest", files=files, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested {data['chunks_ingested']} chunks from {data['filename']}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    if st.button("Check system health"):
        try:
            health = requests.get(f"{API_URL}/health", timeout=10).json()
            st.json(health)
        except Exception as e:
            st.error(f"Health check failed: {e}")

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {msg['sources']}")

# --- Chat input ---
query = st.chat_input("Ask a question about your documents...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        sources = ""
        try:
            with requests.post(
                f"{API_URL}/chat", json={"query": query}, stream=True, timeout=180
            ) as resp:
                sources = resp.headers.get("X-Sources", "")
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        placeholder.write(full_response)
        except Exception as e:
            full_response = f"Error contacting API: {e}"
            placeholder.write(full_response)

        if sources:
            st.caption(f"Sources: {sources}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )
