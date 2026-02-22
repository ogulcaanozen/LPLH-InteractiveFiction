"""Configuration for the LPLH framework."""

import os

# ── Game Settings ──────────────────────────────────────────────
NUM_EPOCHS = 2
MAX_STEPS_PER_EPOCH = 250
HISTORY_LENGTH = 10  # last N turns kept in context

# ── LLM Settings ──────────────────────────────────────────────
# Provider: "ollama" (free, local) or "openai" (paid, API)
LLM_PROVIDER = os.getenv("LPLH_LLM_PROVIDER", "ollama")

# Model name (depends on provider)
# Ollama: "qwen3:8b", "llama3:8b", "mistral", etc.
# OpenAI: "gpt-4o-mini", "gpt-o3-mini", etc.
LLM_MODEL = os.getenv("LPLH_LLM_MODEL", "qwen3:8b")

# Temperature for the main agent LLM (paper uses 0.6)
LLM_TEMPERATURE = 0.6

# Temperature for fine-tuned-style tasks (validation, extraction)
FM_TEMPERATURE = 0.1

# ── Experience Library Settings ───────────────────────────────
EXPERIENCE_TOP_K = 3          # retrieve top K experiences via RAG
CHROMA_COLLECTION = "lplh_experiences"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

# ── Paths ─────────────────────────────────────────────────────
GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "games")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "logs")

# ── Ollama Settings ───────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── OpenAI Settings ───────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
