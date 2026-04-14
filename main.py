from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pipeline import query_pipeline
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
def run_query(payload: dict):
    user_query = payload.get("query", "")
    output = query_pipeline(user_query)

    if isinstance(output, dict) and "error" in output:
        summary = output.get("summary", "An error occurred while processing the query.")
        line_plot_base64 = None
        step_plot_base64 = None
    else:
        df, sql_text, raw_llm, retrieved_ctx, summary, line_plot_base64, step_plot_base64 = output

    paragraphs = summary.split("\n\n")
    formatted_summary = "".join(f"<p>{p}</p>" for p in paragraphs if p.strip())

    if line_plot_base64:
        formatted_summary += f'<img src="data:image/png;base64,{line_plot_base64}" alt="Line Chart" style="max-width:100%; display:block; margin:20px auto;">'
    if step_plot_base64:
        formatted_summary += f'<img src="data:image/png;base64,{step_plot_base64}" alt="Step Chart" style="max-width:100%; display:block; margin:20px auto;">'

    return {"summary": formatted_summary}