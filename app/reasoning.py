import asyncio
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .config import settings
from .llm_client import LLMError, chat

# at most N requests in flight at once
_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_CALLS)

@dataclass
# What every strategy returns
class ReasoningResult:
    final_answer: str
    full_text: str
    strategy: str
    confidence: Optional[float] = None
    details: dict = field(default_factory=dict)

def _messages(system: str, user: str) -> list:
    return [
        { "role": "system", "content": system }, 
        { "role": "user", "content": user },
    ]

async def _limited_chat(*args, **kwargs) -> str:
    #chat() but never more than N at the same time
    async with _semaphore: #wait here if N calls are already running
        return await chat(*args, **kwargs) # forward every argument to the real client

def extract_final(text: str) -> str:
    matches = re.findall(r"final answer[\s*:]*(.+)", text, flags=re.IGNORECASE)
    if not matches:
        return text.strip()
    return matches[-1].strip("*")

def _normalize(answer: str) -> str:
    cleaned = re.sub(r"[$,*]", "", answer.lower()) # drops currency sign, thousands, commas, markdowns stars
    cleaned = " ".join(cleaned.split()).rstrip(".")
    try:
        return str(float(cleaned))
    except ValueError:
        return cleaned

def choose_strategy(question: str) -> str:
     q = question.lower()
     has_digits = bool(re.search(r"\d", q)) # numbers suggest multi-step work
     has_keywords = any(w in q for w in ("why", "how many", "calculate", "compare", "explain", "difference"))
     return "cot" if has_digits or has_keywords else "direct"

async def solve_direct(system: str, user: str) -> ReasoningResult:
     text = await _limited_chat(_messages(system, user), temperature=0.0)
     return ReasoningResult(final_answer=text.strip(), full_text=text, strategy="direct")

async def solve_cot(system: str, user: str) -> ReasoningResult:
     text = await _limited_chat(_messages(system, user), temperature=0.0)
     return ReasoningResult(final_answer=extract_final(text), full_text=text, strategy="cot")

async def solve_self_consistency(system: str, user: str, n: int) -> ReasoningResult:
     runs = await asyncio.gather(
                # start all samples together ( the samaphore limits real concurrency)
                *[_limited_chat(_messages(system, user), temperature=0.7) for _ in range(n)],
                return_exceptions=True 
                # failed samples are returned as an exception instead of crashing everything
            )
     texts = [r for r in runs if isinstance(r, str)]
     if not texts:
         raise next(r for r in runs if isinstance(r, Exception))

     finals = [extract_final(t) for t in texts]
     votes = Counter(_normalize(f) for f in finals)

     winner, count = votes.most_common(1)[0]
     idx = next(i for i, f in enumerate(finals) if _normalize(f) == winner)
     return ReasoningResult(
                final_answer=finals[idx],
                full_text=texts[idx],
                strategy="self_consistency",
                confidence=round(count/len(texts), 2),
                details={"votes": dict(votes), "samples_ok":len(texts), "samples_failed":n-len(texts)},
            )

async def _extend(system: str, problem: str, path: str) -> str:
    prompt = (
        f"{problem}\n\n<reasoning_so_far>\n{path or '(nothing yet)'}\n</reasoning_so_far>"
        "Write ONLY the next reasoning step (1-3 sentences). Do not give the final answer yet."
    )
    text = await _limited_chat(
            _messages(system, prompt), 
            temperature=0.9, 
            max_tokens=300, 
            reasoning=False
        )
    return text.strip()

async def _score(system: str, problem: str, path: str) -> float:
    prompt = (
        f"{problem}\n\n<reasoning>\n{path}\n</reasoning>\n\n"
        "How promising and correct is this reasoning for solving problem? \n"
        "Reply with ONLY a number between 0.0 to 1.0"
    )
    try:
        text = await _limited_chat(_messages(system, prompt), temperature=0.9, max_tokens=300, reasoning=False)
    except LLMError:
        return 0.0
    match = re.search(r"\d*?\d", text)
    return max(0.0, min(1.0, float(match.group()))) if match else 0.0

async def solve_tot(system: str, problem: str) -> ReasoningResult:
    beams = [""]
    scores_log = []
    for _ in range(settings.TOT_DEPTH):
        parents = [p for p in beams for _ in range(settings.TOT_BREADTH)]
        steps = await asyncio.gather(*[_extend(system, problem, p) for p in parents])
        candidates = [f"{p}\n{s}".strip() for p, s in zip(parents, steps)]
        scores = await asyncio.gather(*[_score(system, problem, c) for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        beams = [c for _, c in ranked[:settings.TOT_BEAM]]
        scores_log.append([round(s,2) for s,_ in ranked])
        
    final_prompt = (
        f"{problem}\n\n<best_reasoning>\n{beams[0]}\n</best_reasoning>\n\n"
        "Continue from the reasoning above and finish with one last line: Final answer: <short answer>."
    )
    text = await _limited_chat(_messages(system, final_prompt), temperature=0.0)
        
    return ReasoningResult(
            final_answer=extract_final(text), #short answer from the conclusion
            full_text=f"{beams[0]}\n{text}", # best branch + conclusion
            strategy="tree_of_thought", 
            details={"scores_per_level": scores_log} # shows how the tree was pruned
        )

async def run_strategy(strategy: str, system: str, user_prompt: str, plain_prompt: str) -> ReasoningResult:
    if strategy == "direct":
        return await solve_direct(system, user_prompt)
    elif strategy == "cot":
        return await solve_cot(system, user_prompt)
    elif strategy == "self_consistency":
        return await solve_self_consistency(system, user_prompt, settings.SELF_CONSISTENCY_SAMPLES)
    elif strategy == "tree_of_thought":
        return await solve_tot(system, plain_prompt)
    raise ValueError(f"Unknown strategy: {strategy}")
