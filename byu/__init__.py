"""BYU flagellar-motor competition harness — package init."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
PREPROCESSED = DATA / "preprocessed"
LABELS_CSV = ROOT / "train_labels.csv"

# physical tolerance from the competition metric
TAU_ANGSTROM = 1000.0
