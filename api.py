"""FastAPI service for the fine-tuned dialogue summarizer."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from model import summarize

app = FastAPI(
    title="dialogue-summarizer",
    description="Summarize chat conversations with a LoRA-fine-tuned Qwen2.5-0.5B.",
    version="0.1.0",
)

_INDEX = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


class SummarizeRequest(BaseModel):
    dialogue: str


class SummarizeResponse(BaseModel):
    summary: str


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(_INDEX)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def do_summarize(req: SummarizeRequest) -> SummarizeResponse:
    text = req.dialogue.strip()
    if not text:
        return SummarizeResponse(summary="Please paste a conversation to summarize.")
    if len(text) > 6000:
        return SummarizeResponse(summary="Conversation too long (max 6000 characters).")
    return SummarizeResponse(summary=summarize(text))