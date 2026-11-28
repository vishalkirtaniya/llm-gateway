"""Stage 14: virtual context (MemGPT pattern).

RAM  = MainContext   (the prompt: pinned core sections + recent messages, token-capped)
Disk = ArchivalStore (unbounded, searchable; every record carries session/turn citations)
The agent pages between them by calling memory tools through the Stage 9 ToolRegistry.
"""

import inspect  # tells us whether a decider returns a coroutine
import re  # regular expressions, used to check section names
from dataclasses import (
    dataclass,
    field,
)  # dataclasses give us plain record types with little code
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)  # type hints so the shapes of our data are visible

from ..config import settings  # Stage 1 settings object: all our limits live there
from ..foundation.context import (
    count_tokens,
)  # Stage 7 token estimator, reused so "tokens" mean the same thing everywhere
from .tools import (
    Tool,
    ToolRegistry,
)  # Stage 9 tool interface: memory operations become ordinary tools

_STOPWORDS = frozenset(  # words too common to say anything about relevance; ignored when searching
    "a an the is are was were be been am of to in on at by for with and or but not no it its this that these those "
    "i me my we our you your he she they them his her what which when where who whom how why do does did can could "
    "will would should about from as if so than then there here".split()
)
_SPLIT_RE = re.compile(
    r"[\s!-/:-@\[-`{-~]+"
)  # split on whitespace and ASCII punctuation, so accents and non-Latin scripts stay intact
_SECTION_RE = re.compile(
    r"[a-z][a-z0-9_]{0,31}"
)  # core section names: short, lowercase, one word (checked with fullmatch)


def _words(
    text: str,
) -> set:  # turn text into a set of lowercase words (punctuation removed)
    return {
        w for w in _SPLIT_RE.split(text.lower()) if w
    }  # "Ava," and "ava" match; "Zoë" and Hindi words are kept


def _content_words(
    text: str,
) -> set:  # the words of a query that actually carry meaning
    return _words(text) - _STOPWORDS  # "what is the weather" leaves only "weather"


def _one_line(
    text: str,
) -> str:  # collapse every run of whitespace, newlines included, into one space
    return " ".join(
        text.split()
    )  # a message can no longer start new lines in the prompt


# 1. Main context (RAM)


@dataclass  # auto-generates __init__ so Message(role, text, turn) just works
class Message:  # one conversation message held in the prompt
    role: str  # "user" or "assistant"
    text: str  # what was said (stored exactly as received)
    turn: int  # turn number when it was said, so it can be cited later


@dataclass  # MainContext is also a plain record with behaviour attached
class MainContext:  # the fixed-size prompt buffer: always visible to the model
    max_tokens: int = (
        settings.MEMORY_MAX_CONTEXT_TOKENS
    )  # token cap for the whole buffer (the "RAM size")
    core: Dict[str, str] = field(
        default_factory=dict
    )  # pinned named sections, e.g. "persona" and "user"
    messages: List[Message] = field(
        default_factory=list
    )  # recent messages, oldest first
    evicted: List[Message] = field(
        default_factory=list
    )  # messages pushed out of RAM: the "recall" store
    summary_lines: List[str] = field(
        default_factory=list
    )  # one-line traces of evicted messages, kept in the prompt
    turn: int = 0  # counts messages added so far, used as the turn id for citations

    def used_tokens(self) -> int:  # how much of the cap is currently occupied
        core_cost = sum(
            count_tokens(v) for v in self.core.values()
        )  # tokens held by pinned core sections
        summary_cost = sum(
            count_tokens(s) for s in self.summary_lines
        )  # tokens held by the eviction summary
        message_cost = sum(
            count_tokens(m.text) for m in self.messages
        )  # tokens held by live messages
        return (
            core_cost + summary_cost + message_cost
        )  # total prompt tokens this buffer would use

    def append(
        self, role: str, text: str
    ) -> None:  # add a message, then make room if we are over the cap
        self.turn += 1  # every message gets the next turn number
        self.messages.append(
            Message(role, text, self.turn)
        )  # store it at the newest end of the list
        self._evict_if_needed()  # page old messages out if the buffer is now too big

    def _evict_if_needed(
        self,
    ) -> None:  # the "page out" step: oldest messages leave RAM first
        while (
            self.used_tokens() > self.max_tokens and len(self.messages) > 1
        ):  # never evict the newest message
            old = self.messages.pop(0)  # remove the oldest message from the prompt
            self.evicted.append(
                old
            )  # but keep it: conversation_search can still find it later
            if (
                len(self.evicted) > settings.MEMORY_MAX_EVICTED
            ):  # the recall store must not grow forever
                del self.evicted[0]  # forget the very oldest evicted message
            snippet = _one_line(old.text)[
                : settings.MEMORY_SUMMARY_SNIPPET_CHARS
            ]  # one short line is enough for a trace
            self.summary_lines.append(
                f"turn {old.turn} {old.role}: {snippet}"
            )  # leave a breadcrumb in the prompt
            summary_cap = int(
                self.max_tokens * settings.MEMORY_SUMMARY_SHARE
            )  # breadcrumbs may only use a slice of the cap
            while (
                self.summary_lines
                and sum(count_tokens(s) for s in self.summary_lines) > summary_cap
            ):  # too many breadcrumbs?
                self.summary_lines.pop(
                    0
                )  # drop the oldest breadcrumb first (its message is still in self.evicted)

    def render(self) -> str:  # build the text block that goes into the prompt
        parts = ["[core]"]  # header for the pinned sections
        for key, value in sorted(
            self.core.items()
        ):  # sorted so the prompt is stable between calls
            parts.append(f"  {key}: {_one_line(value)}")  # one line per core section
        if self.summary_lines:  # only show the summary block when something was evicted
            parts.append(
                "[earlier, evicted from window]"
            )  # tells the model older turns exist outside the prompt
            parts.extend(
                f"  {line}" for line in self.summary_lines
            )  # the breadcrumbs, oldest first
        parts.append("[messages]")  # header for the live conversation
        for msg in self.messages:  # walk the messages that are still in RAM
            parts.append(
                f"  {msg.role}: {_one_line(msg.text)}"
            )  # ONE line per message: text cannot forge headers
        return "\n".join(parts)  # join everything into one prompt block


# 2. Archival store (disk)


@dataclass  # one stored fact plus where it came from
class ArchivalRecord:
    rid: str  # unique id such as "a001"
    text: str  # the fact itself
    tags: Tuple[str, ...] = ()  # optional labels, e.g. ("project", "ava")
    session_id: str = "s0"  # which session wrote it: the citation, so answers can say where a fact came from
    turn_id: int = 0  # which turn the fact came from: the other half of the citation


class ArchivalStore:  # unbounded external memory, searched by word overlap
    def __init__(self) -> None:  # start empty
        self._records: List[ArchivalRecord] = []  # every record ever inserted
        self._counter = 0  # used to mint ids a001, a002, ...

    def insert(
        self,
        text: str,
        tags: Tuple[str, ...] = (),
        session_id: str = "s0",
        turn_id: int = 0,
    ) -> ArchivalRecord:  # write one fact to disk
        self._counter += 1  # next id number
        record = ArchivalRecord(
            f"a{self._counter:03d}", text, tags, session_id, turn_id
        )  # build the record with its citation
        self._records.append(record)  # append it to the store
        return record  # hand it back so the caller can report the id

    def search(
        self, query: str, top_k: int
    ) -> List[Tuple[float, ArchivalRecord]]:  # page in the best matches
        q_words = _content_words(
            query
        )  # the meaningful words of the query (stop words removed)
        if not q_words:  # a query made only of stop words says nothing
            return []  # so it matches nothing
        scored: List[
            Tuple[float, ArchivalRecord]
        ] = []  # (score, record) pairs we will sort
        for record in self._records:  # score every record (fine for a small store)
            r_words = _words(
                record.text + " " + " ".join(record.tags)
            )  # tags are searchable too
            overlap = len(q_words & r_words)  # how many query words the record contains
            if overlap == 0:  # no shared words means not relevant at all
                continue  # skip it
            score = (
                overlap / len(q_words)
            )  # coverage: the share of the QUERY the record answers (a long record is not punished)
            if score < settings.MEMORY_MIN_SCORE:  # a weak match is worse than no match
                continue  # skip it
            scored.append((score, record))  # remember the match
        scored.sort(
            key=lambda pair: (pair[0], pair[1].rid), reverse=True
        )  # best score first; on a tie the newer record wins
        return scored[:top_k]  # keep only the top_k matches

    def count(self) -> int:  # how many records are stored
        return len(self._records)  # simple length


# 3. The memory system: RAM + disk + the tool surface


class MemorySystem:  # ties both tiers together and exposes the five MemGPT tools
    def __init__(
        self, session_id: str = "s0", max_tokens: Optional[int] = None
    ) -> None:  # one MemorySystem per session
        self.session_id = (
            session_id  # stamped onto every archival write as the citation
        )
        self.main = MainContext(
            max_tokens or settings.MEMORY_MAX_CONTEXT_TOKENS
        )  # the RAM tier
        self.archival = ArchivalStore()  # the disk tier

    def _commit_core(
        self, section: str, text: str
    ) -> int:  # the ONE place that checks and writes a core section
        text = _one_line(
            text
        )  # pinned text is single-line, so it cannot forge prompt headers
        if not _SECTION_RE.fullmatch(
            section
        ):  # section names end up in the prompt, so they must be plain words
            raise ValueError(
                f"invalid section name {section!r}: use 1-32 lowercase letters, digits or underscores, starting with a letter"
            )
        if (
            section not in self.main.core
            and len(self.main.core) >= settings.MEMORY_MAX_CORE_SECTIONS
        ):  # a new section needs room
            raise ValueError(
                f"core already has {len(self.main.core)} sections (limit {settings.MEMORY_MAX_CORE_SECTIONS}); reuse one or use archival_memory_insert"
            )
        if (
            len(text) > settings.MEMORY_CORE_MAX_CHARS
        ):  # core is always in the prompt, so it must stay small
            raise ValueError(  # execute_call turns this into an error observation the model can read
                f"core[{section}] would be {len(text)} chars, limit is "
                f"{settings.MEMORY_CORE_MAX_CHARS}; move details to archival_memory_insert"
            )
        budget = int(
            self.main.max_tokens * settings.MEMORY_CORE_SHARE
        )  # core may use only a share of the window
        others = sum(
            count_tokens(v) for k, v in self.main.core.items() if k != section
        )  # tokens held by the other sections
        if (
            others + count_tokens(text) > budget
        ):  # core must never crowd the conversation out of the window
            raise ValueError(
                f"core memory would use {others + count_tokens(text)} tokens, the budget is {budget}; move details to archival_memory_insert"
            )
        self.main.core[section] = text  # all checks passed: commit the change
        return len(text)  # new size of the section

    async def core_memory_append(
        self, args: dict
    ) -> dict:  # tool: add text to a pinned section
        section, text = args["section"], args["text"]  # unpack the validated arguments
        current = self.main.core.get(
            section, ""
        )  # existing text of that section, or empty
        size = self._commit_core(
            section, f"{current} {text}"
        )  # join old and new, then check and write
        return {
            "section": section,
            "chars": size,
        }  # observation: confirms what happened

    async def core_memory_replace(
        self, args: dict
    ) -> dict:  # tool: edit text inside a pinned section
        section, old, new = (
            args["section"],
            args["old"],
            args["new"],
        )  # unpack the arguments
        current = self.main.core.get(section, "")  # the section's current text
        if old not in current:  # we can only replace text that exists
            raise ValueError(
                f"{old!r} not found in core[{section}]"
            )  # tell the model exactly what failed
        size = self._commit_core(
            section, current.replace(old, new)
        )  # the same checks as append: replace cannot dodge the limits
        return {"section": section, "chars": size}  # observation: the new size

    async def archival_memory_insert(
        self, args: dict
    ) -> dict:  # tool: write a fact to disk
        record = self.archival.insert(  # store it together with its citation
            args["text"],  # the fact
            tuple(
                args.get("tags", [])
            ),  # optional tags (the schema gives a list, we store a tuple)
            self.session_id,  # which session wrote it
            args.get(
                "source_turn", self.main.turn
            ),  # the turn the fact came from (default: the latest turn)
        )
        return {
            "stored": record.rid,
            "total_records": self.archival.count(),
        }  # observation: id and store size

    async def archival_memory_search(
        self, args: dict
    ) -> dict:  # tool: page facts back in from disk
        top_k = args.get(
            "top_k", settings.MEMORY_SEARCH_TOP_K
        )  # use the default when the model omits it
        hits = self.archival.search(args["query"], top_k)  # best matches with scores
        return {  # observation returned to the model
            "results": [  # one entry per match
                {
                    "id": rec.rid,  # record id
                    "text": rec.text,  # the recalled fact
                    "tags": list(rec.tags),  # its labels
                    "score": round(score, 3),  # how strong the match is
                    "cite": f"{rec.session_id}#turn{rec.turn_id}",  # citation so the answer can say where it came from
                }
                for score, rec in hits  # loop over the matches
            ]
        }

    async def conversation_search(
        self, args: dict
    ) -> dict:  # tool: look through past messages, including evicted ones
        q_words = _content_words(args["query"])  # the meaningful words of the query
        matches = []  # messages that contain every query word
        for msg in reversed(
            list(self.main.evicted) + self.main.messages
        ):  # newest first, evicted and live together
            if q_words and q_words <= _words(
                msg.text
            ):  # every query word must appear in the message
                matches.append(
                    {"role": msg.role, "text": msg.text, "turn": msg.turn}
                )  # keep its text and turn
        return {
            "results": matches[: settings.MEMORY_SEARCH_TOP_K]
        }  # cap how many we return

    def render(self) -> str:  # what the model sees as its memory block
        status = (  # tell the model what exists OUTSIDE the window, or it will never think to look
            "[memory status]\n"
            f"  {len(self.main.evicted)} older messages (conversation_search), "
            f"{self.archival.count()} archival records (archival_memory_search)"
        )
        return self.main.render() + "\n" + status  # the window, then the status line


def with_memory(
    decide: Callable, memory: MemorySystem
) -> Callable:  # wrap a decider so every turn it sees the memory block
    async def decide_with_memory(history: List[dict], tools: List[dict]) -> dict:
        block = {
            "role": "system",
            "content": memory.render(),
        }  # rebuilt every turn, so edits show up immediately
        result = decide(
            [block] + list(history), tools
        )  # the block goes FIRST; the agent's own history is not modified
        return (
            await result if inspect.isawaitable(result) else result
        )  # accept sync and async deciders

    return decide_with_memory  # the loop calls this exactly like the original decider


# 4. Registering the tools (Stage 9 interface)


def build_memory_registry(
    memory: MemorySystem, registry: Optional[ToolRegistry] = None
) -> ToolRegistry:  # add memory tools to a registry
    registry = (
        registry or ToolRegistry()
    )  # reuse the caller's registry, or start a new one
    string = {
        "type": "string",
        "minLength": 1,
    }  # a reusable "non-empty string" schema piece
    registry.register(
        Tool(  # tool 1: append to core memory
            name="core_memory_append",  # name the model will call
            description="Use to pin a short, important fact in always-visible core memory. Do not use for long details.",  # when to use it
            input_schema={
                "type": "object",
                "properties": {"section": string, "text": string},
                "required": ["section", "text"],
            },  # arguments
            executor=memory.core_memory_append,  # the function that runs
        )
    )
    registry.register(
        Tool(  # tool 2: edit core memory
            name="core_memory_replace",  # name the model will call
            description="Use to correct or update text already pinned in a core memory section.",  # when to use it
            input_schema={
                "type": "object",
                "properties": {
                    "section": string,
                    "old": string,
                    "new": {"type": "string"},
                },
                "required": ["section", "old", "new"],
            },  # arguments
            executor=memory.core_memory_replace,  # the function that runs
        )
    )
    registry.register(
        Tool(  # tool 3: write to archival memory
            name="archival_memory_insert",  # name the model will call
            description="Use to save a fact for later recall once it no longer needs to stay in the prompt.",  # when to use it
            input_schema={
                "type": "object",
                "properties": {  # arguments
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": settings.MEMORY_ARCHIVAL_MAX_CHARS,
                    },  # one fact, not a document
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },  # optional labels
                    "source_turn": {
                        "type": "integer",
                        "minimum": 1,
                    },  # optional: the turn the fact was stated in
                },
                "required": ["text"],
            },
            executor=memory.archival_memory_insert,  # the function that runs
        )
    )
    registry.register(
        Tool(  # tool 4: search archival memory
            name="archival_memory_search",  # name the model will call
            description="Use to look up facts saved earlier when the answer is not in the visible context.",  # when to use it
            input_schema={
                "type": "object",
                "properties": {
                    "query": string,
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },  # arguments
            executor=memory.archival_memory_search,  # the function that runs
        )
    )
    registry.register(
        Tool(  # tool 5: search past conversation
            name="conversation_search",  # name the model will call
            description="Use to find something that was said earlier in this conversation, even if it left the window.",  # when to use it
            input_schema={
                "type": "object",
                "properties": {"query": string},
                "required": ["query"],
            },  # arguments
            executor=memory.conversation_search,  # the function that runs
        )
    )
    return registry  # the registry now describes and executes all memory tools
