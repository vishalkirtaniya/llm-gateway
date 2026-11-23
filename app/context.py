import re
from collections import OrderedDict
from typing import List, Optional, Tuple

from .config import settings

def count_tokens(text: str) -> int:
    if not text:
        print(f"count_tokens: text not found so returning 0")
        return 0
    return int(len(text.split()) * 1.3)

def _trim_to_tokens(text: str, max_tokens: int) -> str:
    max_words = int(max_tokens / 1.3) # numbers of words fit in the allowance
    if max_words <= 0: # no reoom for even one word
        return ""
    end = 0 # character position where the kept text ends
    for i, match in enumerate(re.finditer(r"\S+", text), start=1):
        end = match.end() # end position of the current word
        if i >= max_words: # we have kept enough words
            break
    else: # the loop finished without breaking: the text has fewer words than the limit
        return text # nothing to trim
    return text[:end] # keep everything up to the last allowed word (line breaks are preserved)

class ContextBudget:
    def __init__(self, max_tokens: Optional[int] = None, generation_reserve: Optional[int] = None):
        self.max_tokens = max_tokens or settings.CONTEXT_RESERVE
        self.generation_reserve = generation_reserve or settings.GENERATION_RESERVE
        self.available = self.max_tokens - self.generation_reserve
        self.allocations: "OrderedDict[str, int]" = OrderedDict()

    def used(self) -> int:
        return sum(self.allocations.values())

    def remaining(self) -> int:
        return self.available - self.used()

    def allocate(self, component: str, content: str, max_tokens: Optional[int] = None) -> Tuple[str, int]:
        content = content or ""
        if max_tokens is not None and count_tokens(content) > max_tokens:
            content = _trim_to_tokens(content, max_tokens)
        remaining = self.remaining()
        if count_tokens(content) > remaining:
            content = _trim_to_tokens(content, max(remaining, 0))
        tokens = count_tokens(content)
        self.allocations[content] = tokens
        return content, tokens

    def report(self) -> str:
        lines = [f"Windot {self.max_tokens:,} tokens | reserved for answer {self.generation_reserve:,}"]
        for name, tokens in self.allocations.items():
            lines.append(f"{name:<20} {tokens:>6} tokens ({tokens / self.max_tokens * 100:.2f}% of window)")
            lines.append(f"{'remaining':<20} {self.remaining():>6} tokens")
            return "\n".join(lines)

def fit_to_budget(chunks: List[str], scores: List[float], max_tokens: int) -> Tuple[List[str], List[float]]:
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    kept = set()
    used = 0
    for i in order:
        cost = count_tokens(chunks[i])
        if used + cost <= max_tokens:
            kept.add(i)
            used += cost
    keep_idx = [i for i in range(len(chunks)) if i in kept]
    return [chunks[i] for i in keep_idx], [scores[i] for i in keep_idx]

def reorder_lost_in_middle(items: list, scores: list) -> list:
    ranked= sorted(zip(scores, items), key=lambda pair: pair[0], reverse=True)
    ordered = [item for _, item in ranked]
    if len(ordered) <= 2:
        return ordered
    front = ordered[::2]
    back = ordered[1::2]
    back.reverse()
    return front + back

class ConversationManager:
    def __init__(self, max_history_tokens: Optional[int] = None, keep_recent: Optional[int] = None):
        self.max_history_tokens = max_history_tokens or settings.MAX_HISTORY_TOKENS 
        self.keep_recent = keep_recent or settings.KEEP_RECENT_TURNS
        self.turns: List[dict] = []
        self.summaries: List[str] = []

    def token_count(self) -> int:
        return sum(count_tokens(s) for s in self.summaries) + sum(count_tokens(t["content"]) for t in self.turns)
    
    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    @staticmethod
    def _summarize(turns: List[dict]) -> str:
        parts = []
        for t in turns:
            text = t["content"]
            snippet = text[:100] + ("..." if len(text) > 100 else "")
            parts.append(f"{t['role']}: {snippet}")
        return "Earlier - | ".join(parts)

    def _compress_if_needed(self) -> None:
        while self.token_count() > self.max_history_tokens and len(self.turns) >= self.keep_recent + 2:
            old = self.turns[:2]
            self.summaries.append(self._summarize(old))
            self.turns = self.turns[2:]
        while self.token_count() > self.max_history_tokens and self.summaries:
            self.summaries.pop(0)

    def get_context(self) -> str:
        lines = list(self.summaries)
        lines.extend(f"{t['role']}: {t['content']}" for t in self.turns)

        return "\n".join(lines)

class SessionStore:
    def __init__(self, max_sessions: Optional[int] = None):
        self.max_sessions = max_sessions or settings.MAX_SESSIONS 
        self._sessions: "OrderedDict[str, ConversationManager]" = OrderedDict()

    def get(self, session_id: str) -> ConversationManager:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
        else:
            self._sessions[session_id] = ConversationManager()
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        return self._sessions[session_id]

    def __contains__(self, sessions_id: str) -> bool:
        return sessions_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)
