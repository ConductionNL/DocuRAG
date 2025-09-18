from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from RAG import rag, rag_stream, rag_with_session, rag_stream_with_session, SessionState
import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocuRAG")


class In(BaseModel):
    text: str
    session_id: Optional[str] = None


class Out(BaseModel):
    answer: str
    sources: List[Dict]


@app.post("/process", response_model=Out)
def process_endpoint(payload: In):
    try:
        if payload.session_id:
            state = SESSIONS.setdefault(payload.session_id, SessionState())
            logger.info(f"Session state: {state.last_query}")
            result = rag_with_session(payload.text, state)
        else:
            result = rag(payload.text)
        return Out(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.exception("/process failed: %s", e)
        # make failures predictable for clients
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/process_stream")
async def process_stream_endpoint(payload: In):
    try:
        if payload.session_id:
            state = SESSIONS.setdefault(payload.session_id, SessionState())
            logger.info(f"Session state: {state.last_query}")
            gen = rag_stream_with_session(payload.text, state)
        else:
            gen = rag_stream(payload.text)
        return StreamingResponse(
            gen,
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.exception("/process_stream failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


# simple in-memory session storage (sufficient for single-process deployment)
SESSIONS: dict[str, SessionState] = {}
