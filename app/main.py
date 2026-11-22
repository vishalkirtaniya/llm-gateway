from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException # for web framework and HTTP error type
from pydantic import BaseModel, Field # for request and response schemas

from .config import Settings
from .llm_client import LLMError, start_usage_tracking, chat
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .reasoning import choose_strategy, run_strategy
from .structured import StructuredAnswer, extract_structured

app = FastAPI(title="LLM Gateway", version="1.0.0")

settings = Settings()

class AskRequest(BaseModel): # client sends
    question: str = Field(min_length=1, max_length=4000)
    strategy: Literal["auto", "direct", "cot", "self_consistency", "tree_of_thought"] = "auto"
    structured: bool = False

class AskResponse(BaseModel): # what we send back
    answer: str
    strategy_used: str
    confidence: Optional[float] = None
    reasoning: str
    details: dict = Field(default_factory=dict)
    structured: Optional[StructuredAnswer] = None
    structured_attempts: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)

@app.post("/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if not settings.OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

    strategy = choose_strategy(req.question) if req.strategy == "auto" else req.strategy
    mode = "direct" if strategy == "direct" else "cot"
    user_prompt = build_user_prompt(req.question, "", "", mode)
    plain_prompt = build_user_prompt(req.question, "", "", "plain")

    try:
        result = await run_strategy(strategy, SYSTEM_PROMPT, user_prompt, plain_prompt)
        structured_obj, attempts = None, None
        warnings: list[str] = []
        if req.structured:
            try:
                structured_obj, attempts = await extract_structured(req.question, result.final_answer)
            except (ValueError, LLMError) as exc:
                warnings.append(str(exc))

    except LLMError as exc:
        status = 429 if exc.status_code == 429 else 502
        raise HTTPException(status_code=status, detail=exc.message)

    return AskResponse(
            answer=result.final_answer,
            strategy_used=result.strategy,
            confidence=result.confidence,
            reasoning=result.full_text,
            details=result.details,
            structured=structured_obj,
            structured_attempts=attempts,
            warnings=warnings,
        )
