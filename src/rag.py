from .embeddings import search_solr
from .llm import (
    load_llm_message,
    get_llm_response,
    parse_llm_response,
    astream_llm_response,
    classify_follow_up,
)
import os
import pysolr
from dotenv import load_dotenv
import logging
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

load_dotenv()

# Note: SOLR and Fireworks credentials are no longer read from env.
# They are passed per request from the API layer.
# Keeping TOP_K and MODEL_NAME as env-configurable.
TOP_K = os.getenv("TOP_K")
MODEL_NAME = os.getenv("MODEL_NAME")

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    last_query: Optional[str] = None
    last_answer: Optional[str] = None
    retrieved_docs: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)  # {"user": str, "assistant": str}


def is_follow_up_llm(
    user_input: str, session: SessionState, fireworks_api_key: str
) -> bool:
    """Decide via LLM if the input is a follow-up to the previous turn/sources."""
    try:
        return classify_follow_up(
            current_query=user_input,
            last_query=session.last_query,
            prior_sources=session.retrieved_docs,
            last_answer=session.last_answer,
            fireworks_api_key=fireworks_api_key,
        )
    except Exception:
        # Be conservative: default to starting a new topic if classifier fails
        logger.exception("Follow-up classifier failed; defaulting to new topic")
        return False


def augmented_generation(
    retrieved_docs: list[dict], query: str, fireworks_api_key: str
) -> str:
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
    llm_response = get_llm_response(llm_message, fireworks_api_key=fireworks_api_key)
    logger.debug("Parsing LLM response")
    parsed_llm_response = parse_llm_response(llm_response)
    return parsed_llm_response


def _get_solr_client(solr_url: str, solr_username: str, solr_password: str):
    return pysolr.Solr(
        solr_url, always_commit=True, timeout=10, auth=(solr_username, solr_password)
    )


def _prepare_session_prompt(
    user_input: str,
    session: SessionState,
    fireworks_api_key: str,
    solr=None,
):
    """Shared logic to decide follow-up vs new topic, prepare docs and prompt.

    Returns (docs, llm_message, follow_up).
    """
    solr_rag = solr

    follow_up = False
    if session.last_query and session.retrieved_docs:
        follow_up = is_follow_up_llm(user_input, session, fireworks_api_key)

    if follow_up:
        logger.info("Using retrieved docs for follow-up")
        docs = session.retrieved_docs
        history = session.history
    else:
        logger.info("Searching Solr for new topic")
        logger.info("Searching Solr for top_k=%s", TOP_K)
        docs = search_solr(user_input, solr_rag, top_k=TOP_K, model_name=MODEL_NAME)
        logger.info("Retrieved %s docs from Solr", len(docs))
        session.retrieved_docs = list(docs)
        session.history = []
        history = []

    llm_message = load_llm_message(
        docs, user_input, history=history if follow_up else None
    )
    logger.info(f"LLM message: {llm_message}")
    return docs, llm_message, follow_up


def _update_session_after_answer(
    session: SessionState, user_input: str, answer_text: str
):
    session.last_query = user_input
    session.last_answer = answer_text
    session.history.append({"user": user_input, "assistant": answer_text})


def rag(
    user_input: str,
    solr_url: str,
    solr_username: str,
    solr_password: str,
    fireworks_api_key: str,
) -> str:
    """Run retrieval-augmented generation for the given inputs.

    @param solr_rag: Solr instance for querying the index.
    @param user_input: Natural-language query from the user.
    @return: The parsed LLM response string. In the format {'answer': '...', 'sources': ['...']}
    """

    solr_rag = _get_solr_client(solr_url, solr_username, solr_password)
    logger.info("Searching Solr for top_k=%s", TOP_K)
    retrieved_docs = search_solr(
        user_input, solr_rag, top_k=TOP_K, model_name=MODEL_NAME
    )
    logger.info("Retrieved %s docs from Solr", len(retrieved_docs))
    parsed_llm_response = augmented_generation(
        retrieved_docs, user_input, fireworks_api_key
    )
    return parsed_llm_response


async def rag_stream(
    user_input: str,
    solr_url: str,
    solr_username: str,
    solr_password: str,
    fireworks_api_key: str,
):
    """Async generator that streams LLM output as NDJSON lines.

    Retrieval is performed synchronously first, then we stream the LLM output
    for the composed prompt.
    """
    solr_rag = _get_solr_client(solr_url, solr_username, solr_password)
    logger.info("Searching Solr for top_k=%s", TOP_K)
    retrieved_docs = search_solr(
        user_input, solr_rag, top_k=TOP_K, model_name=MODEL_NAME
    )
    logger.info("Retrieved %s docs from Solr", len(retrieved_docs))
    llm_message = load_llm_message(retrieved_docs, user_input)
    async for line in astream_llm_response(
        llm_message, fireworks_api_key=fireworks_api_key
    ):
        yield line


def rag_with_session(
    user_input: str,
    session: SessionState,
    solr_url: str,
    solr_username: str,
    solr_password: str,
    fireworks_api_key: str,
) -> dict:
    """Session-aware RAG: reuse prior sources for follow-ups, retrieve for new topics."""
    solr_rag = _get_solr_client(solr_url, solr_username, solr_password)
    _, llm_message, _ = _prepare_session_prompt(
        user_input, session, fireworks_api_key, solr=solr_rag
    )
    llm_response = get_llm_response(llm_message, fireworks_api_key=fireworks_api_key)
    parsed = parse_llm_response(llm_response)
    _update_session_after_answer(session, user_input, parsed.get("answer", ""))
    return parsed


async def rag_stream_with_session(
    user_input: str,
    session: SessionState,
    solr_url: str,
    solr_username: str,
    solr_password: str,
    fireworks_api_key: str,
):
    """Session-aware streaming RAG with NDJSON output."""
    solr_rag = _get_solr_client(solr_url, solr_username, solr_password)
    _, llm_message, _ = _prepare_session_prompt(
        user_input, session, fireworks_api_key, solr=solr_rag
    )

    buffer_parts: list[str] = []
    async for line in astream_llm_response(
        llm_message, fireworks_api_key=fireworks_api_key
    ):
        try:
            obj = json.loads(line)
            delta = obj.get("delta")
            if delta:
                buffer_parts.append(delta)
        except Exception:
            buffer_parts.append(line)
        yield line

    full = "".join(buffer_parts)
    try:
        parsed = parse_llm_response(full)
        answer_text = parsed.get("answer", full)
    except Exception:
        answer_text = full

    _update_session_after_answer(session, user_input, answer_text)


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
