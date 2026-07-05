import os
import uuid
import io

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- Config ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500          # approx tokens per chunk (using words as a proxy)
CHUNK_OVERLAP = 50
TOP_K = 4

app = FastAPI(title="Self-Hosted RAG Chatbot API")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


class ChatRequest(BaseModel):
    query: str


# --- Helpers ---

def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def extract_text(filename: str, raw_bytes: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return raw_bytes.decode("utf-8", errors="ignore")


def ensure_collection(vector_size: int):
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# --- Endpoints ---

@app.get("/health")
def health():
    status = {"api": "ok"}
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        status["ollama"] = "ok"
    except Exception as e:
        status["ollama"] = f"unreachable: {e}"
    try:
        qdrant.get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"unreachable: {e}"
    return status


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    raw = await file.read()
    text = extract_text(file.filename, raw)
    if not text.strip():
        raise HTTPException(400, "No extractable text found in file")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "Document produced no chunks")

    first_vector = get_embedding(chunks[0])
    ensure_collection(vector_size=len(first_vector))

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=get_embedding(chunk) if i > 0 else first_vector,
            payload={"text": chunk, "source": file.filename, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    return {"filename": file.filename, "chunks_ingested": len(chunks)}


@app.post("/chat")
def chat(request: ChatRequest):
    query_vector = get_embedding(request.query)

    try:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=TOP_K,
        )
    except Exception:
        results = []

    context_chunks = [r.payload["text"] for r in results]
    sources = list({r.payload["source"] for r in results}) if results else []
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {request.query}

Answer:"""

    def stream_response():
        with requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        ) as r:
            for line in r.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]

    response = StreamingResponse(stream_response(), media_type="text/plain")
    response.headers["X-Sources"] = ", ".join(sources)
    return response
