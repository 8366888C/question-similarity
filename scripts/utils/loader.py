import json
from pathlib import Path


def load_contractions():
    base_dir = Path(__file__).parent
    with open(base_dir / "contractions.json", "r", encoding="utf-8") as f:
        return json.load(f)
