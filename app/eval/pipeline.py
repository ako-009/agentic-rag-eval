import json
import time


from datetime import datetime
from pathlib import Path

from app.eval.golden_dataset import load_golden_dataset
from app.eval.metrics import evaluate_single, compute_aggregate_metrics
from app.eval.thresholds import check_deployment_gate

METRICS_HISTORY_PATH = "data/metrics_history.json"


def run_eval_pipeline(
    max_questions: int = None,
    save_results: bool = True,
) -> dict:
    """
    Full evaluation pipeline:
    1. Load golden dataset
    2. Run RAG on each question
    3. Compute metrics
    4. Check deployment gate
    5. Save results to history

    Args:
        max_questions: limit questions for testing (None = all)
        save_results: whether to save to metrics history

    Returns:
        dict with metrics and deployment gate result
    """
    print("=" * 60)
    print("EVAL CI/CD PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load dataset
    pairs = load_golden_dataset()
    if max_questions:
        pairs = pairs[:max_questions]
        print(f"Running on {len(pairs)} questions (limited)")
    else:
        print(f"Running on {len(pairs)} questions (full dataset)")

    # Evaluate each question
    results = []
    for i, pair in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {pair['question'][:60]}...")
        try:
            time.sleep(4)
            result = evaluate_single(pair["question"], pair["ground_truth"])
            results.append(result)
            print(f"  Faithfulness: {result['faithfulness_score']:.2f} | "
                  f"Hallucinated: {result['is_hallucinated']} | "
                  f"Latency: {result['latency_ms']:.0f}ms")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Compute aggregate metrics
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    metrics = compute_aggregate_metrics(results)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Deployment gate
    print("\n" + "=" * 60)
    print("DEPLOYMENT GATE CHECK")
    print("=" * 60)
    gate_result = check_deployment_gate(metrics)

    # Save to history
    if save_results:
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "gate_result": gate_result,
            "num_questions": len(results),
        }
        _save_to_history(history_entry)

    return {
        "metrics": metrics,
        "gate_result": gate_result,
        "results": results,
    }


def _save_to_history(entry: dict):
    """Append metrics to history file for dashboard."""
    path = Path(METRICS_HISTORY_PATH)
    history = []

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nMetrics saved to {METRICS_HISTORY_PATH}")