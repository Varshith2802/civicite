"""FastAPI app: JSON API + a single-page UI."""
from __future__ import annotations

from importlib import resources
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..agent.core import CiviCite


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    mode: Literal["auto", "extractive", "llm"] = "auto"


def create_app(engine: CiviCite, default_mode: str | None = None) -> FastAPI:
    app = FastAPI(title="CiviCite", version="0.1.0",
                  description="Verified, citation-first answers about Swedish public services.")
    ui = resources.files("civicite.api").joinpath("ui.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return ui

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(engine.index.chunks),
                "dense": engine.index.embeddings is not None,
                "llm": getattr(engine.llm, "name", None)}

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        mode = None if req.mode == "auto" else req.mode
        mode = mode or default_mode
        if mode == "llm" and engine.llm is None:
            raise HTTPException(400, "No LLM configured. Start the server with --llm-base-url/--llm-model.")
        return engine.ask(req.question, mode=mode).to_dict()

    return app
