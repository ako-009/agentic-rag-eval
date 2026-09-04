import json
from pathlib import Path

GOLDEN_DATASET_PATH = "data/golden_dataset.json"


def load_golden_dataset() -> list[dict]:
    """
    Load the golden QA pairs from disk.
    
    Each pair has:
    - question: the query
    - ground_truth: the expected correct answer
    - context: which policy section it tests
    """
    path = Path(GOLDEN_DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_DATASET_PATH}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    print(f"Loaded {len(pairs)} golden QA pairs (version {data.get('version', 'unknown')})")
    return pairs


def get_questions(pairs: list[dict]) -> list[str]:
    return [p["question"] for p in pairs]


def get_ground_truths(pairs: list[dict]) -> list[str]:
    return [p["ground_truth"] for p in pairs]