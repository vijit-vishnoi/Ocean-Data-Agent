from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pipeline import query_pipeline

app = FastAPI(title="Ocean-Bot API")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 20

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Serve the root HTML page (if any)."""
    return templates.TemplateResponse(request=request, name="index.html")

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