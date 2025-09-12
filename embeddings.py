# pip install sentence-transformers faiss-cpu numpy
import os, json, hashlib, time, numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss

# --- CPU perf knobs (adjust to your cores) ---
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

MODEL_NAME = "intfloat/multilingual-e5-small"  # fast & multilingual; or "paraphrase-multilingual-MiniLM-L12-v2"

def normalize_text(d: Dict) -> str:
    """Concatenate title, summary, and body into one normalized string.

    @param d: Document record with optional keys 'title', 'summary', and 'body'.
    @return: Normalized text string.
    """
    return " ".join([d.get("title",""), d.get("summary",""), d.get("text","")]).strip()

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
        chunks.append(" ".join(words[i:i+size]))
        i += max(1, size - overlap)
    return chunks or [""]

def checksum(text: str) -> str:
    """Compute SHA-256 checksum of input text.

    @param text: Input text to hash.
    @return: Hex digest string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def build_items(docs: List[Dict], chunk_size: int = 1000, overlap: int = 150) -> List[Dict]:
    """Build vectorizable items with metadata from raw documents.

    @param docs: List of raw document dictionaries.
    @return: List of item dictionaries with 'text' chunks and metadata.
    """
    items = []
    for d in docs:
        text = normalize_text(d)
        ver = checksum(text)[:8]
        for i, ch in enumerate(chunk_text(text, size=chunk_size, overlap=overlap)):
            items.append({
                "vector_id": f"{d['id']}:{ver}:{i}",
                "doc_id": d["id"],
                "version": ver,
                "municipality": d.get("municipality",""),
                "date": d.get("date",""),
                "title": d.get("title",""),
                "chunk_id": i + 1,
                "text": ch
            })
    return items

def encode_cpu(texts: List[str], model_name=MODEL_NAME, batch_size=16):
    """Encode texts into normalized embeddings using SentenceTransformer on CPU.

    @param texts: List of input strings to encode.
    @param model_name: Model name for SentenceTransformer.
    @param batch_size: Batch size used during encoding.
    @return: NumPy array of shape (n_samples, n_dims) with dtype float32.
    """
    model = SentenceTransformer(model_name, device="cpu")
    # normalize_embeddings=True -> cosine-ready
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True).astype("float32")

def build_or_update_faiss(emb: np.ndarray, items: List[Dict],
                          index_path="woo.faiss", map_path="woo_idmap.json", meta_path="woo_meta.json"):
    """Create or update a cosine-similarity FAISS index and metadata files.

    @param emb: Embedding matrix aligned with the given items.
    @param items: List of items corresponding to the embeddings.
    @param index_path: Filesystem path to the FAISS index file.
    @param map_path: Filesystem path to the JSON id-map file.
    @param meta_path: Filesystem path to the JSON metadata file (without 'text').
    @return: Tuple of (faiss.Index, idmap dictionary).
    """
    d = emb.shape[1]
    if os.path.exists(index_path) and os.path.exists(map_path):
        index = faiss.read_index(index_path)
        with open(map_path, "r") as f: idmap = json.load(f)
    else:
        index = faiss.IndexFlatIP(d)  # cosine on normalized vectors
        idmap = {}

    # add only new vectors
    to_add, keys = [], []
    for i, it in enumerate(items):
        if it["vector_id"] not in idmap:
            to_add.append(emb[i])
            keys.append(it["vector_id"])

    if to_add:
        mat = np.vstack(to_add)
        start = time.time()
        index.add(mat)
        print(f"Added {mat.shape[0]} vectors in {(time.time()-start)*1000:.1f} ms")
        base = len(idmap)
        for j, k in enumerate(keys): idmap[k] = base + j

    # persist index + maps (metadata without large text)
    faiss.write_index(index, index_path)
    with open(map_path, "w") as f: json.dump(idmap, f)
    with open(meta_path, "w") as f:
        json.dump({it["vector_id"]: {k:v for k,v in it.items() if k != "text"} for it in items}, f)

    return index, idmap

def search(query: str, top_k=5, model_name=MODEL_NAME,
           index_path="woo.faiss", map_path="woo_idmap.json", meta_path="woo_meta.json"):
    """Search the index for items most similar to a query.

    @param query: Natural-language query string.
    @param top_k: Number of results to return.
    @param model_name: Model name to use for query embedding.
    @param index_path: Path to the FAISS index file.
    @param map_path: Path to the JSON id-map file.
    @param meta_path: Path to the JSON metadata file.
    @return: List of result dicts with ranks, scores, vector_id and metadata.
    """
    # embed query on CPU
    q = encode_cpu([query], model_name=model_name, batch_size=1)
    index = faiss.read_index(index_path)
    D, I = index.search(q, top_k)
    with open(map_path, "r") as f: idmap = json.load(f)
    inv = {v:k for k,v in idmap.items()}
    with open(meta_path, "r") as f: meta = json.load(f)

    results = []
    for rank, (score, pos) in enumerate(zip(D[0], I[0]), 1):
        vid = inv[pos]
        results.append({"rank": rank, "score": float(score), "vector_id": vid, **meta[vid]})
    return results

def search_solr(query: str, solr, top_k=5, model_name=MODEL_NAME):
    """Search the Solr index for items most similar to a query."""
    qv = encode_cpu([query],batch_size=1, model_name=model_name)[0].tolist()

    res = solr.search(
        q=f"{{!knn f=emb topK={top_k}}}" + json.dumps(qv),
        fl="id,doc_id,chunk_id,score,text,municipality,date,title,section,chunk_id",
        rows=top_k
    )
    return res

def index_docs(docs_path: str,
               batch_size: int = 16,
               model_name: str = MODEL_NAME,
               index_path: str = "woo.faiss",
               map_path: str = "woo_idmap.json",
               meta_path: str = "woo_meta.json"):
    """Build or update the FAISS index and metadata from a docs JSON file.

    @param docs_path: Path to the JSON file containing documents.
    @param batch_size: Batch size for embedding computation.
    @param model_name: SentenceTransformer model name for embeddings.
    @param index_path: Output path for the FAISS index file.
    @param map_path: Output path for the id-map JSON file.
    @param meta_path: Output path for the metadata JSON file.
    @return: Tuple of (faiss.Index, idmap dictionary).
    """
    with open(docs_path, "r") as f:
        docs = json.load(f)

    items = build_items(docs)
    texts = [it["text"] for it in items]
    emb = encode_cpu(texts, model_name=model_name, batch_size=batch_size)
    index, idmap = build_or_update_faiss(
        emb,
        items,
        index_path=index_path,
        map_path=map_path,
        meta_path=meta_path,
    )
    return index, idmap

if __name__ == "__main__":
    import sys

    docs_path = sys.argv[1] if len(sys.argv) > 1 else "woo_docs.json"
    index, idmap = index_docs(docs_path, batch_size=16)

    # Example query
    # print(search("AZC capaciteit en afspraken met COA in Wageningen"))