from cups_eval.runner import _verdict_block


def summary_row(accuracy, cost, n=100, parse_success=1.0):
    correct = int(round(accuracy * n))
    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "parse_success_rate": parse_success,
        "estimated_cost_usd": cost,
        "total_tokens": 0,
        "api_calls": 0,
        "tokens_per_correct": None,
        "api_calls_per_correct": None,
        "cost_per_correct": None,
        "collaboration_utility": None,
        "oracle_router_gap": None,
    }


def test_verdict_is_conditional_when_oracle_gap_exists_but_router_is_unvalidated():
    summary = {
        "direct": summary_row(0.75, 1.0),
        "router": summary_row(0.76, 1.1),
        "probe_router": summary_row(0.77, 1.4),
        "planner_verifier": summary_row(0.80, 2.5),
        "oracle_router": summary_row(0.85, 1.05),
    }
    learned_eval = [
        {
            "rule": "direct_conf_lt_0.90_to_planner_verifier",
            "selected": True,
            "deployable": True,
            "test_accuracy": 0.76,
            "test_direct_accuracy": 0.75,
            "test_cost_multiplier": 1.4,
            "test_retained_oracle_gain": 0.10,
        }
    ]
    paired_stats = [
        {"strategy": "router", "exact_p_value": 0.20},
        {"strategy": "probe_router", "exact_p_value": 0.12},
        {"strategy": "planner_verifier", "exact_p_value": 0.01},
    ]

    lines = _verdict_block(summary, learned_eval, paired_stats)

    assert "conditionally feasible" in lines[0]
    assert "Current deployable router signal" in "\n".join(lines)
    assert "Best non-oracle strategy" in "\n".join(lines)


def test_verdict_is_feasible_when_learned_router_retains_enough_oracle_gain():
    summary = {
        "direct": summary_row(0.75, 1.0),
        "router": summary_row(0.76, 1.1),
        "oracle_router": summary_row(0.85, 1.05),
    }
    learned_eval = [
        {
            "rule": "direct_conf_lt_0.90_to_planner_verifier",
            "selected": True,
            "deployable": True,
            "test_accuracy": 0.80,
            "test_direct_accuracy": 0.75,
            "test_cost_multiplier": 1.4,
            "test_retained_oracle_gain": 0.50,
        }
    ]

    lines = _verdict_block(summary, learned_eval, [])

    assert "feasible" in lines[0]
