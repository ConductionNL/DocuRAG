## DocuSearch

This repo provides Retrieval-Augmented Generation (RAG) over WOO documents using Apache Solr for vector search. It retrieves the most relevant chunks from Solr and lets an LLM generate the final answer.

See the the Solr demo with a small testset in `docs_to_solr.ipynb`

### Setup
- Python 3.10+
- Install dependencies:
```bash
pip install -r requirements.txt
```

If you run in Jupyter, also install ipywidgets for nicer progress bars:
```bash
pip install ipywidgets
```

### Environment
Set the following in your environment
```
SOLR_RAG_URL=
SOLR_USER=
SOLR_PASSWORD=
FIREWORKS_API_KEY=
TOP_K=5
MODEL_NAME=intfloat/multilingual-e5-small
```

### Data
- Example corpus: `woo_docs.json`
- Index the corpus into Solr using the `docs_to_solr.ipynb` notebook (creates vector field and documents)

### 1) RAG over Solr (recommended)
From Python:
```python
from RAG import rag

QUESTION = "Wat is er besproken over vuilnis?"
result = rag(QUESTION)
print(result["answer"])        # Final answer
print(result["sources"])       # Attributed sources
```

Via CLI:
```bash
python RAG.py "Wat is er besproken over vuilnis?"
```

Notes:
- `rag()` queries Solr (KNN over the `emb` field) to retrieve context and then calls the LLM using prompts from `prompts.yaml` and the model `llama-v3p3-70b-instruct` via Fireworks.

### 2) Direct semantic search in Solr
`embeddings.search_solr` performs KNN over a vector field in Solr using the configured embedding model.

Prerequisites:
- A running Solr instance with a core/collection (e.g., `docuRAG`) that contains:
  - A dense vector field (e.g., `emb`) configured for KNN
  - Text/metadata fields such as `id`, `doc_id`, `chunk_id`, `text`, `municipality`, `date`, `title`, `section`

Example usage:
```python
import os
import pysolr
from embeddings import search_solr

solr = pysolr.Solr(
    os.getenv("SOLR_RAG_URL"),
    always_commit=True,
    timeout=10,
    auth=(os.getenv("SOLR_USER"), os.getenv("SOLR_PASSWORD")),
)

query = "Wat zegt het over verkeersveiligheid op de Vestdijk?"
results = search_solr(query, solr, top_k=5)
for r in results:
    print(r)
```

For a complete walkthrough on indexing documents into Solr and running a vector search, open the notebook `docs_to_solr.ipynb`.

### 3) REST API (optional)
Run the FastAPI server:
```bash
uvicorn app:app --reload
```

Call the endpoint:
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"text":"Wat is er besproken over vuilnis?"}'
```

### Notebooks
- `docs_to_solr.ipynb`: step-by-step Solr ingestion and KNN search

### Troubleshooting
- Ensure Solr is reachable and the `emb` vector field is configured for KNN
- If embeddings are slow, ensure CPU threading env vars are set (already configured in `embeddings.py`)
- If LLM calls fail: verify `FIREWORKS_API_KEY` and your network access
- If the embedding model fails to load, verify `MODEL_NAME` and that model weights can be downloaded

