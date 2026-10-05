import json
import time
from pathlib import Path

KB_FILE = Path("knowledge_base") / "memories.jsonl"


def remember(note: str) -> dict:
    """Save a note to the long-term knowledge base. Survives restarts."""
    KB_FILE.parent.mkdir(exist_ok=True)
    with KB_FILE.open("a") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M"), "note": note}) + "\n")
    return {"saved": note}


def recall() -> dict:
    """Return everything saved in the knowledge base."""
    if not KB_FILE.exists():
        return {"memories": []}
    with KB_FILE.open() as f:
        return {"memories": [json.loads(line) for line in f if line.strip()]}
