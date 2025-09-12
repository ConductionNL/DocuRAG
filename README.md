## DocuSearch

DocuSearch provides Retrieval-Augmented Generation (RAG) over WOO documents. It supports:
- Semantic retrieval with FAISS, followed by LLM answering via `augmented_generation()`
- Vector KNN search directly in Apache Solr via `search_solr`

See the end-to-end Solr demo in `docs_to_solr.ipynb` and the RAG workflow demo in `DocuSearch.ipynb`.

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
Set the folling in your environent
```
SOLR_RAG_URL=
SOLR_USER=
SOLR_PASSWORD=
export FIREWORKS_API_KEY=
```

### Data
- Example corpus: `woo_docs.json`
- Local FAISS files are written as you run retrieval: `woo.faiss`, `woo_idmap.json`, `woo_meta.json`

### 1) Augmented generation (RAG)
From Python (recommended):
```python
from RAG import retrieve_docs, augmented_generation

DOCSET = "woo_docs.json"
QUESTION = "Wat is er besproken over vuilnis?"

retrieved_docs = retrieve_docs(DOCSET, QUESTION, top_k=5)
answer = augmented_generation(retrieved_docs, QUESTION)
print(answer)
```

Via CLI:
```bash
python RAG.py woo_docs.json "Wat is er besproken over vuilnis?"
```

Notes:
- `retrieve_docs` will (re)build the local FAISS index from `woo_docs.json` if needed; the first run may take longer.
- `augmented_generation()` uses prompts from `prompts.yaml` and model `llama-v3p3-70b-instruct` via Fireworks.

### 2) Semantic search in Solr
`embeddings.search_solr` performs KNN over a vector field in Solr using the same embedding model as FAISS.

Prerequisites:
- A running Solr instance with a core/collection (e.g., `docuRAG`) that contains:
  - A dense vector field (e.g., `emb`) configured for KNN
  - Text/metadata fields such as `id`, `doc_id`, `chunk_id`, `text`, `municipality`, `date`, `title`, `section`

Example usage:
```python
import pysolr
from embeddings import search_solr

solr = pysolr.Solr("http://localhost:8983/solr/docuRAG", always_commit=True, timeout=10)

query = "Wat zegt het over verkeersveiligheid op de Vestdijk?"
results = search_solr(query, solr, top_k=5)
for r in results:
    print(r)
```

For a complete walkthrough on indexing documents into Solr and running a vector search, open the notebook `docs_to_solr.ipynb`.

### Notebooks
- `docs_to_solr.ipynb`: step-by-step Solr ingestion and KNN search
- `DocuSearch.ipynb`: example RAG flow using `retrieve_docs` and `augmented_generation`

### Troubleshooting
- If FAISS is missing: `pip install faiss-cpu`
- If embeddings are slow, ensure CPU threading env vars are set (already configured in `embeddings.py`)
- If LLM calls fail: verify `FIREWORKS_API_KEY` and your network access

