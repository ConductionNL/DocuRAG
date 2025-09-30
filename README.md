## DocuSearch

This repo provides Retrieval-Augmented Generation (RAG) over WOO documents using Apache Solr for vector search. It retrieves the most relevant chunks from Solr and lets an LLM generate the final answer.

See the Solr demo with a small testset in `notebooks/docs_to_solr.ipynb`.

### Setup
- Python 3.10+
- Install dependencies (with `uv`):
```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv and install project deps
uv sync
```

If you run in Jupyter, also install `ipywidgets` for nicer progress bars:
```bash
uv add ipywidgets
```

### Configuration (no secrets in env)
- Credentials are passed per request to the API or function calls. Do not store Solr or Fireworks secrets in `.env`.
- Optional environment variables for non-secrets:
```
LOG_LEVEL=INFO
TOP_K=5
MODEL_NAME=intfloat/multilingual-e5-small
LLM_NAME=llama-v3p3-70b-instruct
```

Prompts are loaded from `src/prompts/prompts.yaml`. The default chat model is `llama-v3p3-70b-instruct` if `LLM_NAME` is not set.

### 1) RAG over Solr from Python
From Python (run from repo root):
```python
from src.rag import rag

result = rag(
    "Wat is er besproken over afval?",
    solr_url="http://<solr-host>/solr/<chunks_collection>",
    solr_username="<username>",
    solr_password="<password>",
    fireworks_api_key="<fireworks_api_key>",
)
print(result["answer"])        # Final answer
print(result["sources"])       # Attributed sources
```

Notes:
- Retrieval queries Solr (KNN over the `emb` field) and then calls the LLM using prompts from `src/prompts/prompts.yaml`.

### 2) REST API
Run the FastAPI server (from repo root):
```bash
uv run -- uvicorn app:app --reload --app-dir .
```

Call the endpoint (credentials in the request body):
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Wat is er besproken over afval?",
    "fireworks_api_key": "<fireworks_api_key>",
    "solr_url": "http://<solr-host>/solr/<chunks_collection>",
    "solr_username": "<username>",
    "solr_password": "<password>"
  }'
```

#### Streaming API (NDJSON)
For incremental tokens, use the streaming endpoint which emits NDJSON lines:
```bash
curl -N -X POST http://localhost:8000/process_stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Wat is er besproken over afval?",
    "fireworks_api_key": "<fireworks_api_key>",
    "solr_url": "http://<solr-host>/solr/<chunks_collection>",
    "solr_username": "<username>",
    "solr_password": "<password>"
  }'
```

Behavior:
- Each line is a JSON object ending with a newline
- Lines look like `{\"delta\":\"...\"}` for partial content
- The last line is `{\"event\":\"done\"}`


### 3) Docker
Build and run the API server with Docker:
```bash
docker build -t docusearch-api .
docker run --rm -p 8000:8000 docusearch-api
```

Or with Docker Compose (also mounts a Hugging Face cache volume):
```bash
docker compose up --build
```

Then call the API as shown above. Credentials are always provided in the request payload; they are not read from container environment variables.


### Notebooks
- `notebooks/docs_to_solr.ipynb`: step-by-step Solr ingestion and KNN search

### Repository layout
```
app.py                 # FastAPI server
src/
  rag.py               # RAG entrypoints (sync + streaming, session-aware variants)
  llm.py               # Fireworks LLM helpers and JSON parsing
  embeddings.py        # Solr KNN search utilities
  prompts/prompts.yaml # Prompt templates
notebooks/             # Demos and ingestion
Dockerfile
docker-compose.yml
pyproject.toml
```

### Troubleshooting
- Ensure Solr is reachable and the `emb` vector field is configured for KNN
- If LLM calls fail, verify the Fireworks API key used in your request payload
- If the embedding model fails to load, verify `MODEL_NAME` and network access for model weights

