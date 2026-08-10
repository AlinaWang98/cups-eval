# CUPS Policy Results

No provider calls are made for this step. CUPS is an offline, out-of-fold policy selector over saved `router_features.csv` rows.

## Method

Cost-aware Utility Policy Selection (CUPS) trains correctness models for candidate strategies and selects the strategy maximizing:

`P(correct | features, strategy=s) - lambda * estimated_cost_ratio(s)`

The deployed feature set uses question/direct-output features plus dataset/model identifiers. Selection uses train-fold mean cost ratios. Final metrics charge the realized cost of the selected strategy.

## Main Results

Values below are aligned with `results/cups/cups_best_policy_tests.csv` using `evaluation_regime=item_oof`, `selection_type=best_accuracy`, and `baseline=cot`.

| Scope | Best CUPS point | Accuracy | Cost x Direct | Relative CPC | Delta vs CoT | Escalation rate |
|---|---|---:|---:|---:|---:|---:|
| Medical GLM/Kimi, 800 rows | lambda=0.02 | 0.735 | 1.44 | 1.960 | +0.006 | 26.8% |
| Non-medical GLM/Kimi, 400 rows | lambda=0.15 | 0.882 | 1.24 | 1.410 | +0.030 | 15.8% |
| Non-medical DeepSeek Flash, 200 rows | lambda=0.05 | 0.785 | 1.90 | 2.415 | +0.095 | 59.0% |

## Baseline Comparison

Always-on strategy baselines below come from `results/tables/cross_domain_main_table.csv`; the CUPS points come from `results/cups/cups_best_policy_tests.csv`.

| Scope | Direct acc / CPC | CoT acc / CPC | Always planner acc / CPC | Always debate acc / CPC | Strategy-selection oracle acc / CPC | CUPS acc / CPC |
|---|---:|---:|---:|---:|---:|---:|
| Medical GLM/Kimi | 0.706 / 1.416 | 0.729 / 1.366 | 0.730 / 2.752 | 0.733 / 4.132 | 0.799 / 1.467 | 0.735 / 1.960 |
| Non-medical DeepSeek Flash | 0.695 / 1.439 | 0.690 / 1.459 | 0.800 / 2.682 | 0.775 / 4.250 | 0.845 / 1.417 | 0.785 / 2.415 |

## Interpretation

CUPS captures part of the sparse collaboration opportunity, but the tradeoff is scope-dependent. In these bundled results, CUPS improves accuracy over CoT on the headline medical and non-medical DeepSeek scopes, while still requiring careful cost-normalized interpretation.

## Artifacts

- `results/cups/cups_best_policy_tests.csv`: paired CUPS-vs-baseline tests.
- `results/cups/cups_curve_all_scopes.csv`: all swept policies and baseline points.
- `results/cups/cups_per_dataset_model_metrics.csv`: dataset-model paired metrics.
- `results/cups/cups_strategy_mix_by_scope.csv`: selected-strategy distributions.
