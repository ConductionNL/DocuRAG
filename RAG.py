import json
from embeddings import index_docs, search, search_solr
from llm import load_llm_message, get_llm_response, parse_llm_response
import os
import pysolr
from dotenv import load_dotenv

load_dotenv()

SOLR_RAG_URL = os.getenv("SOLR_RAG_URL")
USERNAME = os.getenv("SOLR_USER")
PASSWORD = os.getenv("SOLR_PASSWORD")
TOP_K = os.getenv("TOP_K")
MODEL_NAME = os.getenv("MODEL_NAME")


def retrieve_docs(corpus_path: str, query: str, top_k: int = 5) -> list[dict]:
    """For FAISS: Retrieve top-k documents relevant to the query from the corpus.

    @param corpus_path: Path to the JSON file with the documents corpus.
    @param query: Natural-language search query.
    @param top_k: Number of documents to retrieve.
    @return: List of document dicts corresponding to the top matches.
    """
    # Open corpus to load docs
    corpus = json.load(open(corpus_path))
    # Build/update index from a given docs file
    index, idmap = index_docs(corpus_path)
    # Match the top k docs to the query
    results = search(query, top_k)
    top_doc_ids = [result["doc_id"] for result in results[:top_k]]
    retrieved_docs = [doc for doc in corpus if doc["id"] in top_doc_ids]
    return retrieved_docs


def augmented_generation(retrieved_docs: list[dict], query: str) -> str:
    """Generate an answer using the LLM with retrieved context documents.

    @param retrieved_docs: List of top-k documents to include as context.
    @param query: User's question.
    @return: Parsed LLM response.
    """
    print("loading llm message")
    llm_message = load_llm_message(retrieved_docs, query)
    print(f"llm message: {llm_message}")
    print("getting llm response")
    llm_response = get_llm_response(llm_message)
    print("parsing llm response")
    parsed_llm_response = parse_llm_response(llm_response)
    return parsed_llm_response


def rag(user_input: str) -> str:
    """Run retrieval-augmented generation for the given inputs.

    @param solr_rag: Solr instance for querying the index.
    @param user_input: Natural-language query from the user.
    @return: The parsed LLM response string. In the format {'answer': '...', 'sources': ['...']}
    """

    solr_rag = pysolr.Solr(
        SOLR_RAG_URL, always_commit=True, timeout=10, auth=(USERNAME, PASSWORD)
    )
    retrieved_docs = search_solr(
        user_input, solr_rag, top_k=TOP_K, model_name=MODEL_NAME
    )
    parsed_llm_response = augmented_generation(retrieved_docs, user_input)
    return parsed_llm_response


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python RAG.py <user_input>")
        sys.exit(1)
    user_input = " ".join(sys.argv[1:])

    output = rag(user_input)
    print(output)
