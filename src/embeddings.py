import hashlib
import json
import logging
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from threading import Lock
from typing import Dict, List

load_dotenv()

# --- CPU perf knobs (adjust to your cores) ---
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

MODEL_NAME = str(os.getenv("MODEL_NAME"))

logger = logging.getLogger(__name__)


# --- Per-process cache for SentenceTransformer models ---
_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_MODEL_CACHE_LOCK = Lock()


def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance for the given model name.

    Ensures the model is loaded once per process/worker. Thread-safe double-checked
    locking avoids repeated loads under concurrent requests.
    """
    model = _MODEL_CACHE.get(model_name)
    if model is not None:
        return model
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name, device="cpu")
            _MODEL_CACHE[model_name] = model
    return model


def normalize_text(d: Dict) -> str:
    """Concatenate title, summary, and body into one normalized string.

    @param d: Document record with optional keys 'title', 'summary', and 'body'.
    @return: Normalized text string.
    """
    return " ".join(
        [d.get("title", ""), d.get("summary", ""), d.get("text", "")]
    ).strip()


def chunk_text(txt: str, size=1000, overlap=150) -> List[str]:
    """Split text into overlapping word chunks.

    @param txt: Input text to chunk.
    @param size: Maximum number of words per chunk.
    @param overlap: Word overlap between consecutive chunks.
    @return: List of text chunks.
    """
    words = txt.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += max(1, size - overlap)
    return chunks or [""]


def checksum(text: str) -> str:
    """Compute SHA-256 checksum of input text.

    @param text: Input text to hash.
    @return: Hex digest string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_items(
    docs: List[Dict], chunk_size: int = 1000, overlap: int = 150
) -> List[Dict]:
    """Build vectorizable items with metadata from raw documents.

    @param docs: List of raw document dictionaries.
    @return: List of item dictionaries with 'text' chunks and metadata.
    """
    items = []
    for d in docs:
        text = normalize_text(d)
        ver = checksum(text)[:8]
        for i, ch in enumerate(chunk_text(text, size=chunk_size, overlap=overlap)):
            items.append(
                {
                    "vector_id": f"{d['id']}:{ver}:{i}",
                    "doc_id": d["id"],
                    "version": ver,
                    "municipality": d.get("municipality", ""),
                    "date": d.get("date", ""),
                    "title": d.get("title", ""),
                    "chunk_id": i + 1,
                    "text": ch,
                }
            )
    return items


def encode_cpu(texts: List[str], model_name=MODEL_NAME, batch_size=16):
    """Encode texts into normalized embeddings using SentenceTransformer on CPU.

    @param texts: List of input strings to encode.
    @param model_name: Model name for SentenceTransformer.
    @param batch_size: Batch size used during encoding.
    @return: NumPy array of shape (n_samples, n_dims) with dtype float32.
    """
    print(f"Encoding {len(texts)} texts with model {model_name}")
    model = get_sentence_transformer(model_name)
    # normalize_embeddings=True -> cosine-ready
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


def search_solr(query: str, solr, top_k=5, model_name=MODEL_NAME):
    """Search the Solr index for items most similar to a query."""
    qv = encode_cpu([query], batch_size=1, model_name=model_name)[0].tolist()

    res = solr.search(
        q=f"{{!knn f=emb topK={top_k}}}" + json.dumps(qv),
        fl="id,doc_id,chunk_id,score,text,municipality,date,title,section,chunk_id",
        rows=top_k,
    )
    return res


if __name__ == "__main__":

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
