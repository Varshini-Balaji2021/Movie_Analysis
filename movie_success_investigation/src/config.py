"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "movies_raw.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

RANDOM_STATE = 42
MIN_GROUP_SIZE = 20        # minimum movies for a "reliable" group
MAX_MULTIPLE_CLEAN = 20    # cap used for the histogram of multiples
MIN_MEANINGFUL_BUDGET = 100_000

BUDGET_LABELS = ["Lower budget", "Lower-middle", "Upper-middle", "Higher budget"]

for _d in (FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
