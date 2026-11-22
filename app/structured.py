import json
from typing import List, Tuple

from pydantic import BaseModel, Field, ValidationError

from .config import settings
from .llm_client import chat

class StructuredAnswer(BaseModel):
    answer: str = Field(min_length=1, description="Final answer in one or two sentences")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 (guess) to 1.0 (certain)")
    key_points: List[str] = Field(default_factory=list, max_length=5, description="Up to 5 supporting points")
    needs_human_review: bool = Field(description="True when confidence is below 0.6")

def json_slice(text: str) -> str:
    start, end = text.find("{"), text.rfind("}") # position of the first "{" and the last "}"
    if start != -1 and end > start: # if both position found and in right order
        return text[start : end + 1] # return only part in between them
    return text # dont do nothing and let the json.loads fail with clear error 

def parse_structured(raw: str) -> StructuredAnswer:
    data = json.loads(json_slice(raw)) # checks if there is valid json or not?
    return StructuredAnswer.model_validate(data) # checks if the the json matches the StructuredAnswer or not

async def extract_structured(
    question: str, draft_answer: str, context: str = "", max_retries: int = None
    ) -> Tuple[StructuredAnswer, int]:
    """
        Convert a draft answer to validate JSON. Returns (object, attempts used).
    """
    max_retries = max_retries or settings.STRUCTURED_MAX_RETRIES
    schema = json.dumps(StructuredAnswer.model_json_schema())
    #extraction-engine schema
    system = (
        "You are a data extraction engine. Output Only a valid JSON object that matches this JSON Schema. \n"
        "No markdown fences, no explanation, no text before or after the JSON. \n"
        f"Schema: {schema}"
    )
    parts = [f"<question>{question}</question>", f"<draft_answer>{draft_answer}</draft_answer>"]
    if context.strip():
        parts.append(f"<context>{context}</context>")
    parts.append(
        "Fill every field. confidence reflexts how well the draft answer is supported. "
        "Set needs_human_review to true if confidence is below 0.6"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]
    last_error = "unknown_error"
    for attempt in range(1, max_retries + 1):
        raw = await chat(messages, temperature=0.0, json_mode=settings.JSON_MODE_SUPPORTED)
        try:
            return parse_structured(raw), attempt
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)[:400]
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user",
                "content": f"Your previous outpur was invalid: {last_error}\n Return corrected JSON only."},
            )
    raise ValueError(f"Structured output failed after {max_retries} attemps: {last_error}")
