"""Generate data/split_v1.csv. Run once."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from byu.split import save_split
import pandas as pd
p = save_split()
df = pd.read_csv(p)
print(f"wrote {p}  train={ (df.fold=='train').sum() }  val={ (df.fold=='val').sum() }")
