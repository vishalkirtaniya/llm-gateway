import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "nvidia/nemotron-3.5-super-vl:free")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b:free"
    )
    APP_URL: str = os.getenv("APP_URL", "http://locahost:8000")
    APP_NAME: str = os.getenv("APP_NAME", "llm-gateway")

    REQUEST_TIMEOUT: float = 60.0
    MAX_RETRIES: int = 3
    MAX_OUTPUT_TOKENS: int = 2000

    CONTEXT_WINDOW: int = 128_000
    GENERATION_RESERVE: int = 4_000
    MAX_SYSTEM_TOKENS: int = 1_000
    MAX_QUESTION_TOKENS: int = 1_000
    MAX_CONTEXT_TOKENS: int = 3_000
    MAX_HISTORY_TOKENS: int = 3_000
    KEEP_RECENT_TURNS: int = 4
    MAX_SESSIONS: int = 500

    TOP_K: int = 4
    MIN_SCORE: float = 0.30
    CHUNK_WORDS: int = 120
    CHUNK_OVERLAP: int = 20

    SELF_CONSISTENCY_SAMPLES: int = 5
    TOT_BREADTH: int = 2
    TOT_DEPTH: int = 2
    TOT_BEAM: int = 2
    MAX_CONCURRENT_CALLS: int = 2

    KNOWLEDGE_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "knowledge"

    JSON_MODE_SUPPORTED: bool = (
        os.getenv("JSON_MODE_SUPPORTED", "false").lower() == "true"
    )
    STRUCTURED_MAX_RETRIES: INT = 3

    EMBED_QUERY_PREFIX: str = os.getenv("EMBED_QUERY_PREFIX", "")
    EMBED_DOC_PREFIX: str = os.getenv("EMBED_DOC_PREFIX", "")
    EMBED_CACHE_PATH: Path = (
        Path(__file__).resolve().parent.parent / ".cache" / "embeddings.json"
    )
    EMBED_BATCH_SIZE: int = 32
    MIN_SCORE: float = 0.18

    MAX_TOOL_TURNS: int = 5
    MAX_TOOL_RESULT_CHARS: int = 4000

    AGENT_MAX_TURNS: int = 12
    AGENT_MAX_TOKENS: int = 8000
    AGENT_MAX_CALLS_PER_TURN: int = 4

    TOOL_TIMEOUT_S: float = 10.0
    TOOL_MAX_CONCURRENCY: int = 4

    REWOO_MAX_STEPS: int = 8
    REWOO_MAX_REPLANS: int = 1
    REWOO_MAX_EVIDENCE_CHARS: int = 6000


settings = Settings()
