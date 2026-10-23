import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("P3_RAW_DIR", ROOT / "data" / "raw"))
REG_DIR = Path(os.environ.get("P3_REG_DIR", ROOT / "data" / "regulations"))
DB_PATH = Path(os.environ.get("P3_DB_PATH", ROOT / "data" / "foodsafe.duckdb"))
OUT_DIR = Path(os.environ.get("P3_OUT_DIR", ROOT / "outputs"))
INDEX_PATH = Path(os.environ.get("P3_INDEX_PATH", ROOT / "data" / "reg_index.joblib"))

INSPECTIONS_URL = "https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD"
CSV_NAME = "food_inspections.csv"

# Chicago switched to a new violation numbering (FDA-Food-Code based) on 1 July 2018.
# Set P3_CODE_ERA_START=2000-01-01 for the 2010-2014 data, which uses only the older numbering.
NEW_CODE_START = os.environ.get("P3_CODE_ERA_START", "2018-07-01")
SEED = 42
FDR_Q = 0.05
MIN_INSPECTIONS_FOR_FLAG = 4

LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")
DENSE_MODEL = os.environ.get("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
