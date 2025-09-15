from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
from RAG import rag, rag_stream
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


class Out(BaseModel):
    answer: str
    sources: List[Dict]


@app.post("/process", response_model=Out)
def process_endpoint(payload: In):
    try:
        result = rag(payload.text)
        return Out(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.exception("/process failed: %s", e)
        # make failures predictable for clients
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/process_stream")
async def process_stream_endpoint(payload: In):
    try:
        return StreamingResponse(
            rag_stream(payload.text),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.exception("/process_stream failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
