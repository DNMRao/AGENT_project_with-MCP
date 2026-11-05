"""Read .env once so every module uses the same settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # copies KEY=value lines from .env into os.environ

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
EMBEDDING_REVISION = os.getenv("EMBEDDING_REVISION") or None
if EMBEDDING_REVISION is None:
    if EMBEDDING_MODEL != DEFAULT_EMBEDDING_MODEL:
        raise ValueError("Set EMBEDDING_REVISION when changing EMBEDDING_MODEL")
    EMBEDDING_REVISION = DEFAULT_EMBEDDING_REVISION

# Retrieval. MIN_SCORE was chosen with src/calibration.py on data/calibration/ only.
# The env name differs from the reference solution's MIN_SCORE (a lexical scale).
TOP_K = int(os.getenv("TOP_K") or 3)
MIN_SCORE = float(os.getenv("EMBEDDING_MIN_SCORE") or 0.40)  # middle of the 0.375-0.425 plateau
VECTOR_STORAGE_PATH = os.getenv("VECTOR_STORAGE_PATH") or None  # None = in-memory

# Agent control: a step is one planner decision (a tool call or the final answer).
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS") or 8)
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS") or 6)

# Durable case memory survives between runs of the CLI (git-ignored).
MEMORY_PATH = Path(os.getenv("MEMORY_PATH") or "artifacts/case_memory.json")

if TOP_K < 1 or not 0 < MIN_SCORE <= 1:
    raise ValueError("TOP_K must be positive and EMBEDDING_MIN_SCORE must be in (0, 1]")
if MAX_AGENT_STEPS < 1 or MAX_TOOL_CALLS < 1:
    raise ValueError("MAX_AGENT_STEPS and MAX_TOOL_CALLS must be positive")
