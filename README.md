# Self-Hosted RAG Chatbot

A fully containerized Retrieval-Augmented Generation chatbot: Ollama (LLM + embeddings),
Qdrant (vector DB), FastAPI (backend), and Streamlit (frontend).

## Prerequisites
- Docker + Docker Compose installed
- ~8GB free RAM (more if using a larger model)
- Optional: NVIDIA GPU with the NVIDIA Container Toolkit for faster inference

## Setup

1. **Start Ollama first and pull the models:**
   ```bash
   docker compose up -d ollama
   docker exec -it rag-ollama ollama pull llama3
   docker exec -it rag-ollama ollama pull nomic-embed-text
   ```

2. **Build and start everything:**
   ```bash
   docker compose up --build
   ```

3. **Open the app:**
   - Frontend: http://localhost:8501
   - API docs: http://localhost:8000/docs
   - Qdrant dashboard: http://localhost:6333/dashboard

## Usage

1. In the sidebar, upload a PDF, TXT, or Markdown file and click "Ingest document."
2. Ask a question in the chat box — the AI will answer using only the retrieved context.
3. Check "Sources" under each answer to see which document it pulled from.

## Using a GPU (optional)

Add this to the `ollama` service in `docker-compose.yml`:
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Tuning

- **Chunk size/overlap**: edit `CHUNK_SIZE` / `CHUNK_OVERLAP` in `api/main.py`
- **Retrieval depth**: edit `TOP_K` in `api/main.py`
- **Model**: change `LLM_MODEL` / `EMBED_MODEL` in `docker-compose.yml` (pull the new model into Ollama first)

## Swapping in Claude API instead of a local model

Replace the `stream_response()` call in `api/main.py` with a call to the Anthropic API
using the retrieved `context` as part of the prompt — useful if you want higher-quality
answers and don't mind an external API dependency.

## Troubleshooting

- **"Ollama unreachable"**: wait a few seconds after startup, or check `docker logs rag-ollama`
- **Empty/bad answers**: verify a document was actually ingested (`/health` and Qdrant dashboard)
- **Slow responses**: local LLMs are CPU-bound without a GPU; try a smaller model like `phi3` or `qwen2:1.5b`
