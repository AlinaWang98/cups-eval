from cups_eval.metrics import compute_summary, is_correct_result, pareto_frontier
from cups_eval.schema import StrategyResult, Usage


def result(strategy, answer, gold, cost, calls=1):
    return StrategyResult(
        strategy=strategy,
        answer=answer,
        confidence=0.8,
        rationale="",
        calls=[],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        parse_ok=True,
        metadata={"estimated_cost_usd": cost, "gold": gold, "api_calls": calls},
    )


def test_compute_summary_handles_zero_correct_and_cost_metrics():
    rows = [
        result("direct", "A", "B", 0.01),
        result("direct", "C", "D", 0.01),
        result("debate_2_judge", "B", "B", 0.04, calls=3),
        result("debate_2_judge", "D", "D", 0.04, calls=3),
    ]

    summary = compute_summary(rows, baseline_strategy="direct", oracle_accuracy=1.0)

    assert summary["direct"]["accuracy"] == 0.0
    assert summary["direct"]["tokens_per_correct"] is None
    assert summary["debate_2_judge"]["accuracy"] == 1.0
    assert round(summary["debate_2_judge"]["collaboration_utility"], 3) == 16.667
    assert summary["debate_2_judge"]["oracle_router_gap"] == 0.0


def test_compute_summary_prefers_precomputed_normalized_correctness():
    rows = [
        StrategyResult(
            strategy="direct",
            answer="30",
            confidence=0.8,
            rationale="",
            calls=[],
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            parse_ok=True,
            metadata={
                "estimated_cost_usd": 0.01,
                "gold": "30^\\circ",
                "api_calls": 1,
                "correct": True,
            },
        )
    ]

    summary = compute_summary(rows)

    assert summary["direct"]["accuracy"] == 1.0


def test_is_correct_result_prefers_precomputed_normalized_correctness():
    item = StrategyResult(
        strategy="direct",
        answer="30",
        confidence=0.8,
        rationale="",
        calls=[],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        parse_ok=True,
        metadata={"gold": "30^\\circ", "correct": True},
    )

    assert is_correct_result(item) is True


def test_pareto_frontier_keeps_non_dominated_cost_accuracy_points():
    summaries = {
        "cheap_bad": {"accuracy": 0.4, "estimated_cost_usd": 0.01},
        "expensive_good": {"accuracy": 0.8, "estimated_cost_usd": 0.04},
        "expensive_bad": {"accuracy": 0.5, "estimated_cost_usd": 0.05},
    }

    frontier = pareto_frontier(summaries)

    assert [row["strategy"] for row in frontier] == ["cheap_bad", "expensive_good"]
