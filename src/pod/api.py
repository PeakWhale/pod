"""FastAPI HTTP service for PeakWhale™ Pod."""
from __future__ import annotations

from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .engine import ClusterResult, cluster_csv, cluster_dataframe, list_examples, load_example

app = FastAPI(
    title="PeakWhale™ Pod API",
    description="Domain-agnostic cohort and regional clustering service with plain-English segment profiles.",
    version="1.0.0",
)


class ClusterRequest(BaseModel):
    rows: list[dict[str, Any]] | None = None
    csv_text: str | None = None
    example: str | None = None
    k: int | None = None
    max_k: int = 6
    id_column: str | None = None
    feature_columns: list[str] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "peakwhale-pod"}


@app.get("/api/examples")
def get_examples() -> dict[str, list[str]]:
    return {"examples": list_examples()}


@app.post("/api/cluster", response_model=ClusterResult)
def post_cluster(req: ClusterRequest) -> ClusterResult:
    try:
        if req.example:
            df = load_example(req.example)
            return cluster_dataframe(df, k=req.k, max_k=req.max_k, id_column=req.id_column, feature_columns=req.feature_columns)
        if req.csv_text:
            return cluster_csv(req.csv_text, k=req.k, max_k=req.max_k, id_column=req.id_column, feature_columns=req.feature_columns)
        if req.rows:
            df = pd.DataFrame(req.rows)
            return cluster_dataframe(df, k=req.k, max_k=req.max_k, id_column=req.id_column, feature_columns=req.feature_columns)
        raise ValueError("Provide 'example', 'csv_text', or 'rows'.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>PeakWhale™ Pod</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 780px; margin: 40px auto; padding: 0 20px;">
  <h1>PeakWhale™ Pod</h1>
  <p>Domain-agnostic cohort and regional clustering engine. Explore the interactive API docs at <a href="/docs">/docs</a>.</p>
</body>
</html>"""
