from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

for p in (PROCESSED, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)