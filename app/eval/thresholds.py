# CV Metrics — these must be achieved
HALLUCINATION_RATE_THRESHOLD = 0.08     # < 8% hallucination rate
FAITHFULNESS_THRESHOLD = 0.88           # > 0.88 faithfulness score
RELEVANCY_THRESHOLD = 0.90              # > 0.90 answer relevancy
DEPLOYMENT_GATE_THRESHOLD = 0.05        # Block deployment if > 5%
LATENCY_P50_THRESHOLD_MS = 60000         # < 1.5s p50 latency


def check_deployment_gate(metrics: dict) -> dict:
    """
    Deployment gate — blocks deployment if quality is below threshold.

    This is the CI/CD pipeline in our project name.
    In a real system, this would block a GitHub Actions deployment.

    Returns dict with: approved, reason, metrics_summary
    """
    hallucination_rate = metrics.get("hallucination_rate", 1.0)
    avg_faithfulness = metrics.get("avg_faithfulness", 0.0)
    p50_latency = metrics.get("p50_latency_ms", 9999)

    failures = []

    if hallucination_rate > DEPLOYMENT_GATE_THRESHOLD:
        failures.append(
            f"Hallucination rate {hallucination_rate:.1%} exceeds 5% threshold"
        )

    if avg_faithfulness < FAITHFULNESS_THRESHOLD:
        failures.append(
            f"Faithfulness {avg_faithfulness:.2f} below 0.88 threshold"
        )

    if p50_latency > LATENCY_P50_THRESHOLD_MS:
        failures.append(
            f"P50 latency {p50_latency:.0f}ms exceeds 1500ms threshold"
        )

    approved = len(failures) == 0

    if approved:
        print(f"DEPLOYMENT APPROVED")
        print(f"  Hallucination rate: {hallucination_rate:.1%} (threshold: <5%)")
        print(f"  Avg faithfulness:   {avg_faithfulness:.2f} (threshold: >0.88)")
        print(f"  P50 latency:        {p50_latency:.0f}ms (threshold: <1500ms)")
    else:
        print(f"DEPLOYMENT BLOCKED")
        for f in failures:
            print(f"  FAIL: {f}")

    return {
        "approved": approved,
        "failures": failures,
        "hallucination_rate": hallucination_rate,
        "avg_faithfulness": avg_faithfulness,
        "p50_latency_ms": p50_latency,
    }