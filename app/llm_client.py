import asyncio
import contextvars
from typing import Optional

import httpx

from .config import Settings

settings = Settings()

class LLMError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

_usage_var: contextvars.ContextVar = contextvars.ContextVar("usage", default=None)

def start_usage_tracking() -> dict:
    # counters, all start at zero
    tracker = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    _usage_var.set(tracker) # this makes this dict the current request's tracker
    return tracker

def _record_usage(usage: dict) -> None:
    tracker = _usage_var.get()
    if tracker is None:
        print("No Tracing started")
        return
    tracker["llm_calls"] += 1
    tracker["prompt_tokens"] += usage.get("prompt_tokens", 0)
    tracker["completion_tokens"] += usage.get("completion_tokens", 0)

def _header() -> dict: # this builds the http headers every openrouter request needs
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referor": settings.APP_URL, #(optional) identifies the app on openrouter
        "X-Title": settings.APP_NAME, #(optional) display name on openrouter dashboards
    }

async def _post(path: str, payload: dict) -> dict:
    url = f"{settings.OPENROUTER_BASE_URL}{path}"
    last_error: Optional[LLMError] = None # remembers the most recent retryable failure
    for attempt in range(settings.MAX_RETRIES):
        try:
            # open the client
            async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
                # send the request
                resp = await client.post(url, headers=_header(), json=payload)
        # can be DNS failure, timeout, connection reset etc
        except httpx.RequestError as exc:
            last_error = LLMError(503, f"Network error talking to OpenRouter: {exc}")
        else:
            if resp.status_code == 401:
                raise LLMError(401, "OpenRouter rejected the API key, Check OPENROUTER_API_KEY.")
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = LLMError(resp.status_code, f"Upstream error {resp.status_code}: {resp.text[:200]}")
            elif resp.status_code >=400:
                raise LLMError(resp.status_code, f"Open router error: {resp.status_code}: {resp.text[:200]}")
            else:
                return resp.json()
        if attempt < settings.MAX_RETRIES - 1:
            await asyncio.sleep(2 ** attempt)
    raise last_error or LLMError(503, "Unknown Upstream Failure")

async def chat(
        messages: list, 
        temperature: float = 0.0, 
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        reasoning: Optional[bool] = None,
    ) -> str:
    payload = {
        "model": settings.DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or settings.MAX_OUTPUT_TOKENS,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning is not None:
        payload["reasoning"] = {"type": reasoning}

    data = await _post("/chat/completions", payload)
    if "error" in data:
        raise LLMError(502, f"Model error: {data['error']}")

    _record_usage(data.get("usage", {}))
    choice = data["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    if not text:
        raise LLMError(
            502,
            f"Empty reply (finish_reason={choice.get('finish_reason')})."
            "If it is 'length', raise max_tokens or call with reasoning=False."
        )
    return text

async def embed(texts: list) -> list:
    payload = {"model": settings.EMBEDDING_MODEL, "input": texts}
    data = await _post("/embeddings", payload)
    if "error" in data:
        raise LLMError(502, f"Embedding error: {data['error']}")
    items = sorted(data["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in items]

async def embed(texts: list) -> list:
    if not settings.EMBEDDING_MODEL:
        raise LLMError(500, "EMBEDDING_MODEL is not set. Add it to your .env file")
    payload = {"model": settings.EMBEDDING_MODEL, "input": texts}
    data = await _post("/embeddings", payload)
    if "error" in data:
        raise LLMError(502, f"Embedding error: {data['error']}")
    items = sorted(data['data'], key=lambda item: item["index"])
    return [item["embedding"] for item in items]
