from embeddings import search_solr
from llm import (
    load_llm_message,
    get_llm_response,
    parse_llm_response,
    astream_llm_ndjson,
)
import os
import pysolr
from dotenv import load_dotenv
import logging

load_dotenv()

SOLR_RAG_URL = os.getenv("SOLR_RAG_URL")
USERNAME = os.getenv("SOLR_USER")
PASSWORD = os.getenv("SOLR_PASSWORD")
TOP_K = os.getenv("TOP_K")
MODEL_NAME = os.getenv("MODEL_NAME")

logger = logging.getLogger(__name__)


def augmented_generation(retrieved_docs: list[dict], query: str) -> str:
    """Generate an answer using the LLM with retrieved context documents.

    @param retrieved_docs: List of top-k documents to include as context.
    @param query: User's question.
    @return: Parsed LLM response.
    """
    llm_message = load_llm_message(retrieved_docs, query)
    logger.debug(
        "LLM message prepared: %s",
        llm_message[:500] + ("..." if len(llm_message) > 500 else ""),
    )
    logger.info("Getting LLM response")
    llm_response = get_llm_response(llm_message)
    logger.debug("Parsing LLM response")
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
    logger.info("Searching Solr for top_k=%s", TOP_K)
    retrieved_docs = search_solr(
        user_input, solr_rag, top_k=TOP_K, model_name=MODEL_NAME
    )
    logger.info("Retrieved %s docs from Solr", len(retrieved_docs))
    parsed_llm_response = augmented_generation(retrieved_docs, user_input)
    return parsed_llm_response


async def rag_stream(user_input: str):
    """Async generator that streams LLM output as NDJSON lines.

    Retrieval is performed synchronously first, then we stream the LLM output
    for the composed prompt.
    """
    solr_rag = pysolr.Solr(
        SOLR_RAG_URL, always_commit=True, timeout=10, auth=(USERNAME, PASSWORD)
    )
    logger.info("Searching Solr for top_k=%s", TOP_K)
    retrieved_docs = search_solr(
        user_input, solr_rag, top_k=TOP_K, model_name=MODEL_NAME
    )
    logger.info("Retrieved %s docs from Solr", len(retrieved_docs))
    llm_message = load_llm_message(retrieved_docs, user_input)
    async for line in astream_llm_ndjson(llm_message):
        yield line


if __name__ == "__main__":
    import sys

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if len(sys.argv) < 2:
        logger.error("Usage: python RAG.py <user_input>")
        sys.exit(1)
    user_input = " ".join(sys.argv[1:])

    output = rag(user_input)
    logger.info("RAG output: %s", output)
