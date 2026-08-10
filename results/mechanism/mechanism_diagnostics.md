# Mechanism Diagnostics

Diagnostics are split into canonical rescue/harm decompositions and CUPS router FP/FN tables.

## Largest Canonical Net Gains

| dataset | model | comparison | direct | comparison | net |
|---|---|---|---:|---:|---:|
| gsm8k | deepseek_v4_flash | `planner_verifier` | 69.0% | 89.0% | +20.0 pp |
| gsm8k | deepseek_v4_flash | `debate_2_judge` | 69.0% | 88.0% | +19.0 pp |
| math500 | deepseek_v4_flash | `debate_2_judge` | 43.2% | 60.2% | +17.0 pp |
| math500 | glm_5_2 | `debate_2_judge` | 72.0% | 85.0% | +13.0 pp |
| math500 | glm_5_2 | `planner_verifier` | 72.0% | 84.0% | +12.0 pp |
| math500 | deepseek_v4_flash | `planner_verifier` | 43.2% | 54.6% | +11.4 pp |
| medqa | glm_5_2 | `debate_2_judge` | 78.0% | 88.0% | +10.0 pp |
| medqa | glm_5_2 | `planner_verifier` | 78.0% | 88.0% | +10.0 pp |
| medqa | glm_5_2 | `cot` | 78.0% | 87.0% | +9.0 pp |
| math500 | minimax_m3 | `debate_2_judge` | 89.0% | 94.0% | +5.0 pp |

## Oracle Opportunity Snapshot

| dataset | model | direct acc | oracle acc | gap | direct errors rescued |
|---|---|---:|---:|---:|---:|
| math500 | deepseek_v4_flash | 43.2% | 68.8% | +25.6 pp | 45.1% |
| gsm8k | deepseek_v4_flash | 69.0% | 93.0% | +24.0 pp | 77.4% |
| math500 | glm_5_2 | 72.0% | 93.0% | +21.0 pp | 75.0% |
| medqa | glm_5_2 | 78.0% | 95.0% | +17.0 pp | 77.3% |
| medxpertqa_text | kimi_k3 | 43.0% | 57.0% | +14.0 pp | 24.6% |
| medxpertqa_text | glm_5_2 | 28.0% | 40.0% | +12.0 pp | 16.7% |
| mmlu_pro | kimi_k3 | 74.0% | 84.0% | +10.0 pp | 38.5% |
| math500 | minimax_m3 | 89.0% | 98.0% | +9.0 pp | 81.8% |
| mmlu_pro | glm_5_2 | 73.0% | 81.0% | +8.0 pp | 29.6% |
| gsm8k | glm_5_2 | 90.0% | 98.0% | +8.0 pp | 80.0% |
| medmcqa | glm_5_2 | 80.0% | 88.0% | +8.0 pp | 40.0% |
| medmcqa | kimi_k3 | 84.0% | 91.0% | +7.0 pp | 43.8% |

## Router FP/FN Snapshot

| scope | dataset | model | policy | TP | FN | FP | TN | recall | FPR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `medical_glm_kimi` | medmcqa | glm_5_2 | `cups_lambda_0.02` | 7 | 1 | 21 | 71 | 87.5% | 22.8% |
| `medical_glm_kimi` | medmcqa | kimi_k3 | `cups_lambda_0.02` | 6 | 1 | 8 | 85 | 85.7% | 8.6% |
| `medical_glm_kimi` | medqa | glm_5_2 | `cups_lambda_0.02` | 15 | 2 | 21 | 62 | 88.2% | 25.3% |
| `medical_glm_kimi` | medqa | kimi_k3 | `cups_lambda_0.02` | 5 | 1 | 8 | 86 | 83.3% | 8.5% |
| `medical_glm_kimi` | medxpertqa_text | glm_5_2 | `cups_lambda_0.02` | 9 | 3 | 54 | 34 | 75.0% | 61.4% |
| `medical_glm_kimi` | medxpertqa_text | kimi_k3 | `cups_lambda_0.02` | 8 | 6 | 29 | 57 | 57.1% | 33.7% |
| `medical_glm_kimi` | pubmedqa | glm_5_2 | `cups_lambda_0.02` | 3 | 3 | 18 | 76 | 50.0% | 19.1% |
| `medical_glm_kimi` | pubmedqa | kimi_k3 | `cups_lambda_0.02` | 0 | 4 | 2 | 94 | 0.0% | 2.1% |
| `medical_glm_kimi` | medmcqa | glm_5_2 | `cups_lambda_0.02` | 7 | 1 | 21 | 71 | 87.5% | 22.8% |
| `medical_glm_kimi` | medmcqa | kimi_k3 | `cups_lambda_0.02` | 6 | 1 | 8 | 85 | 85.7% | 8.6% |
| `medical_glm_kimi` | medqa | glm_5_2 | `cups_lambda_0.02` | 15 | 2 | 21 | 62 | 88.2% | 25.3% |
| `medical_glm_kimi` | medqa | kimi_k3 | `cups_lambda_0.02` | 5 | 1 | 8 | 86 | 83.3% | 8.5% |
| `medical_glm_kimi` | medxpertqa_text | glm_5_2 | `cups_lambda_0.02` | 9 | 3 | 54 | 34 | 75.0% | 61.4% |
| `medical_glm_kimi` | medxpertqa_text | kimi_k3 | `cups_lambda_0.02` | 8 | 6 | 29 | 57 | 57.1% | 33.7% |
| `medical_glm_kimi` | pubmedqa | glm_5_2 | `cups_lambda_0.02` | 3 | 3 | 18 | 76 | 50.0% | 19.1% |
| `medical_glm_kimi` | pubmedqa | kimi_k3 | `cups_lambda_0.02` | 0 | 4 | 2 | 94 | 0.0% | 2.1% |
| `medical_glm_kimi` | medmcqa | glm_5_2 | `cups_lambda_0.05` | 7 | 1 | 21 | 71 | 87.5% | 22.8% |
| `medical_glm_kimi` | medmcqa | kimi_k3 | `cups_lambda_0.05` | 5 | 2 | 2 | 91 | 71.4% | 2.2% |

## Files

- `rescue_harm_by_dataset_model.csv`: direct vs each collaboration strategy.
- `oracle_opportunity_by_dataset_model.csv`: direct-first oracle rescue opportunity.
- `cups_router_fp_fn_by_scope_dataset_model.csv`: CUPS escalation diagnosis.
