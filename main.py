from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pipeline import query_pipeline

app = FastAPI(title="Ocean-Bot API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 20

@app.get("/")
def read_root():
    """Serve a simple health check."""
    return {"status": "Ocean-Bot API is running"}

@app.post("/query")
def run_query(payload: QueryRequest):
    """
    Executes the Text-to-SQL and RAG pipeline.
    
    Returns a JSON payload with the summary, base64 charts, and debug info.
    """
    if not payload.query.strip():
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty."})

    output = query_pipeline(payload.query, top_k=payload.top_k)
    return output