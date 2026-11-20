from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException # for web framework and HTTP error type
from pydantic import BaseModel, Field # for request and response schemas

from .config import Settings
from .llm_client import LLMError, start_usage_tracking, chat

app = FastAPI(title="LLM Gateway", version="1.0.0")
settings = Settings()

class AskRequest(BaseModel): # client sends
    question: str = Field(min_length=1, max_length=4000)

class AskResponse(BaseModel): # what we send back
    answer: str

@app.post("/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if not settings.OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": req.question}
    ]

    try:
        text = await chat(messages, temperature=0.0)
    except LLMError as exc:
        status = 429 if exc.status_code == 429 else 502
        raise HTTPException(status_code=status, detail=exc.message)

    return AskResponse(answer=text.strip())
