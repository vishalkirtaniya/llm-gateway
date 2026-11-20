import os
from pathlib import Path 

from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "openai/gpt/gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    APP_URL: str = os.getenv("APP_URL", "http://locahost:8000")
    APP_NAME: str = os.getenv("APP_NAME", "llm-gateway")
    
    REQUEST_TIMEOUT: float = 60.0
    MAX_RETRIES: int = 3 
    MAX_OUTPUT_TOKENS: int = 800

    CONTEXT_WINDOW: int = 128_000
    GENERATION_RESERVE: int = 4_000

    TOP_K: int = 4 
    MIN_SCORE: float = 0.30
    CHUNK_WORDS: int = 120
    CHUNK_OVERLAP: int = 20

    SELF_CONSISTENCY_SAMPLES: int = 5
    TOT_BREADTH: int = 2
    TOT_DEPTH: int = 2
    TOT_BEAM: int = 2

    KNOWLEDGE_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "knowledge"

settings = Settings()

