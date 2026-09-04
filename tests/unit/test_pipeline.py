import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from app.graph.edges import should_retry
from app.graph.state import RAGState
from app.eval.thresholds import check_deployment_gate


class TestEdgeRouting:
    """Tests for the conditional edge routing logic."""

    def test_grounded_answer_goes_to_finalize(self):
        state = {
            "is_hallucinated": False,
            "retry_count": 0,
        }
        assert should_retry(state) == "finalize"

    def test_hallucinated_with_retries_goes_to_reformulate(self):
        state = {
            "is_hallucinated": True,
            "retry_count": 0,
        }
        assert should_retry(state) == "reformulate"

    def test_hallucinated_max_retries_goes_to_fallback(self):
        state = {
            "is_hallucinated": True,
            "retry_count": 2,
        }
        assert should_retry(state) == "fallback"

    def test_retry_count_1_still_reformulates(self):
        state = {
            "is_hallucinated": True,
            "retry_count": 1,
        }
        assert should_retry(state) == "reformulate"


class TestDeploymentGate:
    """Tests for the deployment gate logic."""

    def test_gate_approves_good_metrics(self):
        metrics = {
            "hallucination_rate": 0.02,
            "avg_faithfulness": 0.95,
            "p50_latency_ms": 1000,
        }
        result = check_deployment_gate(metrics)
        assert result["approved"] is True
        assert result["failures"] == []

    def test_gate_blocks_high_hallucination(self):
        metrics = {
            "hallucination_rate": 0.10,
            "avg_faithfulness": 0.95,
            "p50_latency_ms": 1000,
        }
        result = check_deployment_gate(metrics)
        assert result["approved"] is False
        assert len(result["failures"]) > 0

    def test_gate_blocks_low_faithfulness(self):
        metrics = {
            "hallucination_rate": 0.02,
            "avg_faithfulness": 0.70,
            "p50_latency_ms": 1000,
        }
        result = check_deployment_gate(metrics)
        assert result["approved"] is False

    def test_gate_blocks_high_latency(self):
        metrics = {
            "hallucination_rate": 0.02,
            "avg_faithfulness": 0.95,
            "p50_latency_ms": 80000,
        }
        result = check_deployment_gate(metrics)
        assert result["approved"] is False