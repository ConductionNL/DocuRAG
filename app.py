from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from RAG import rag

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
        # make failures predictable for clients
        raise HTTPException(status_code=400, detail=str(e))
