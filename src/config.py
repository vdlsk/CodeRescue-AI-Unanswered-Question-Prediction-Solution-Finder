import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("P2_RAW_DIR", ROOT / "data" / "raw"))        # extracted dump XML files
DATA_DIR = Path(os.environ.get("P2_DATA_DIR", ROOT / "data" / "processed"))
OUT_DIR = Path(os.environ.get("P2_OUT_DIR", ROOT / "outputs"))
MODEL_DIR = DATA_DIR / "models"

UNANSWERED_DAYS = 7
SEED = 42

# Temporal split by question creation time (fractions of the time-ordered questions)
TRAIN_FRAC, VALID_FRAC = 0.70, 0.15     # test = last 15%

# Retrieval corpus: solved questions from the training period only
CORPUS_MIN_ANSWER_SCORE = 1             # accepted answer, or best answer with score >= this
DENSE_MODEL = os.environ.get("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = 10
RRF_K = 60

# LLM (optional). Set ANTHROPIC_API_KEY to enable generation and the with/without-RAG evaluation.
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-5-5")
LLM_EVAL_N = int(os.environ.get("LLM_EVAL_N", "100"))
