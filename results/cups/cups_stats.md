# CUPS Statistical Package

CUPS was regenerated offline from existing `router_features.csv` files. No model/API calls are involved.

## Scope Summary

| scope | rows | dataset-models |
|---|---:|---|
| `medical_glm_kimi` | 800 | medmcqa-glm_5_2, medmcqa-kimi_k3, medqa-glm_5_2, medqa-kimi_k3, medxpertqa_text-glm_5_2, medxpertqa_text-kimi_k3, pubmedqa-glm_5_2, pubmedqa-kimi_k3 |
| `medical_glm_kimi` | 800 | medmcqa-glm_5_2, medmcqa-kimi_k3, medqa-glm_5_2, medqa-kimi_k3, medxpertqa_text-glm_5_2, medxpertqa_text-kimi_k3, pubmedqa-glm_5_2, pubmedqa-kimi_k3 |
| `medical_glm_kimi` | 800 | medmcqa-glm_5_2, medmcqa-kimi_k3, medqa-glm_5_2, medqa-kimi_k3, medxpertqa_text-glm_5_2, medxpertqa_text-kimi_k3, pubmedqa-glm_5_2, pubmedqa-kimi_k3 |
| `nonmedical_glm_kimi` | 400 | gsm8k-glm_5_2, gsm8k-kimi_k3, mmlu_pro-glm_5_2, mmlu_pro-kimi_k3 |
| `nonmedical_glm_kimi` | 400 | gsm8k-glm_5_2, gsm8k-kimi_k3, mmlu_pro-glm_5_2, mmlu_pro-kimi_k3 |
| `nonmedical_glm_kimi` | 400 | gsm8k-glm_5_2, gsm8k-kimi_k3, mmlu_pro-glm_5_2, mmlu_pro-kimi_k3 |
| `nonmedical_deepseek_flash` | 200 | gsm8k-deepseek_v4_flash, mmlu_pro-deepseek_v4_flash |
| `nonmedical_deepseek_flash` | 200 | gsm8k-deepseek_v4_flash, mmlu_pro-deepseek_v4_flash |
| `math500_core_deepseek_glm` | 200 | math500-deepseek_v4_flash, math500-glm_5_2 |
| `math500_core_deepseek_glm` | 200 | math500-deepseek_v4_flash, math500-glm_5_2 |
| `math500_core_deepseek_glm` | 200 | math500-deepseek_v4_flash, math500-glm_5_2 |
| `math500_minimax_m3` | 100 | math500-minimax_m3 |
| `math500_qwen_clean` | 100 | math500-qwen_3_7_max_preview |

## Best-Policy Tests

| scope | selection | policy | baseline | delta | 95% CI | p |
|---|---|---|---|---:|---:|---:|
| `medical_glm_kimi` | best_accuracy | `cups_lambda_0.02` | direct | +2.9 pp | [+1.1, +4.5] | 0.001 |
| `medical_glm_kimi` | best_accuracy | `cups_lambda_0.02` | cot | +0.6 pp | [-0.9, +2.1] | 0.522 |
| `medical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.02` | direct | +2.9 pp | [+1.1, +4.5] | 0.001 |
| `medical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.02` | cot | +0.6 pp | [-0.9, +2.1] | 0.522 |
| `medical_glm_kimi` | best_accuracy | `cups_lambda_0.05` | direct | +2.8 pp | [+1.0, +4.2] | 0.001 |
| `medical_glm_kimi` | best_accuracy | `cups_lambda_0.05` | cot | +0.5 pp | [-1.0, +2.0] | 0.618 |
| `medical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.05` | direct | +2.8 pp | [+1.0, +4.2] | 0.001 |
| `medical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.05` | cot | +0.5 pp | [-1.0, +2.0] | 0.618 |
| `medical_glm_kimi` | best_accuracy | `cups_lambda_0.05` | direct | +3.0 pp | [+1.5, +4.6] | <0.001 |
| `medical_glm_kimi` | best_accuracy | `cups_lambda_0.05` | cot | +0.8 pp | [-0.9, +2.4] | 0.451 |
| `medical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.05` | direct | +3.0 pp | [+1.5, +4.6] | <0.001 |
| `medical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.05` | cot | +0.8 pp | [-0.9, +2.4] | 0.451 |
| `nonmedical_glm_kimi` | best_accuracy | `cups_lambda_0.15` | direct | +4.5 pp | [+2.2, +7.0] | <0.001 |
| `nonmedical_glm_kimi` | best_accuracy | `cups_lambda_0.15` | cot | +3.0 pp | [+1.0, +5.0] | 0.004 |
| `nonmedical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.15` | direct | +4.5 pp | [+2.2, +7.0] | <0.001 |
| `nonmedical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.15` | cot | +3.0 pp | [+1.0, +5.0] | 0.004 |
| `nonmedical_glm_kimi` | best_accuracy | `cups_lambda_0.005` | direct | +4.8 pp | [+2.5, +7.0] | <0.001 |
| `nonmedical_glm_kimi` | best_accuracy | `cups_lambda_0.005` | cot | +3.2 pp | [+1.0, +5.5] | 0.007 |
| `nonmedical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.005` | direct | +4.8 pp | [+2.5, +7.0] | <0.001 |
| `nonmedical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.005` | cot | +3.2 pp | [+1.0, +5.5] | 0.007 |
| `nonmedical_glm_kimi` | best_accuracy | `cups_lambda_0.01` | direct | +4.8 pp | [+2.5, +7.2] | <0.001 |
| `nonmedical_glm_kimi` | best_accuracy | `cups_lambda_0.01` | cot | +3.2 pp | [+1.2, +5.2] | 0.002 |
| `nonmedical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.01` | direct | +4.8 pp | [+2.5, +7.2] | <0.001 |
| `nonmedical_glm_kimi` | best_vs_cot_cpc | `cups_lambda_0.01` | cot | +3.2 pp | [+1.2, +5.2] | 0.002 |
| `nonmedical_deepseek_flash` | best_accuracy | `cups_lambda_0.05` | direct | +9.0 pp | [+4.0, +14.0] | <0.001 |
| `nonmedical_deepseek_flash` | best_accuracy | `cups_lambda_0.05` | cot | +9.5 pp | [+4.5, +14.5] | <0.001 |
| `nonmedical_deepseek_flash` | best_vs_cot_cpc | `cups_lambda_0.05` | direct | +9.0 pp | [+4.0, +14.0] | <0.001 |
| `nonmedical_deepseek_flash` | best_vs_cot_cpc | `cups_lambda_0.05` | cot | +9.5 pp | [+4.5, +14.5] | <0.001 |
| `nonmedical_deepseek_flash` | best_accuracy | `cups_lambda_0` | direct | +9.5 pp | [+4.5, +15.0] | <0.001 |
| `nonmedical_deepseek_flash` | best_accuracy | `cups_lambda_0` | cot | +10.0 pp | [+5.0, +15.0] | <0.001 |
| `nonmedical_deepseek_flash` | best_vs_cot_cpc | `cups_lambda_0.2` | direct | +0.5 pp | [-1.0, +2.0] | 1.000 |
| `nonmedical_deepseek_flash` | best_vs_cot_cpc | `cups_lambda_0.2` | cot | +1.0 pp | [-3.0, +5.0] | 0.791 |
| `math500_core_deepseek_glm` | best_accuracy | `cups_lambda_0.02` | direct | +13.5 pp | [+9.0, +18.5] | <0.001 |
| `math500_core_deepseek_glm` | best_accuracy | `cups_lambda_0.02` | cot | +14.0 pp | [+8.5, +19.5] | <0.001 |
| `math500_core_deepseek_glm` | best_vs_cot_cpc | `cups_lambda_0.02` | direct | +13.5 pp | [+9.0, +18.5] | <0.001 |
| `math500_core_deepseek_glm` | best_vs_cot_cpc | `cups_lambda_0.02` | cot | +14.0 pp | [+8.5, +19.5] | <0.001 |
| `math500_core_deepseek_glm` | best_accuracy | `cups_lambda_0` | direct | +8.5 pp | [+3.5, +13.5] | 0.002 |
| `math500_core_deepseek_glm` | best_accuracy | `cups_lambda_0` | cot | +9.0 pp | [+4.0, +14.0] | 0.001 |
| `math500_core_deepseek_glm` | best_vs_cot_cpc | `cups_lambda_0.08` | direct | +0.5 pp | [+0.0, +1.5] | 1.000 |
| `math500_core_deepseek_glm` | best_vs_cot_cpc | `cups_lambda_0.08` | cot | +1.0 pp | [-4.5, +6.5] | 0.860 |
| `math500_core_deepseek_glm` | best_accuracy | `cups_lambda_0` | direct | +8.5 pp | [+3.5, +13.5] | 0.002 |
| `math500_core_deepseek_glm` | best_accuracy | `cups_lambda_0` | cot | +9.0 pp | [+4.0, +14.0] | 0.001 |
| `math500_core_deepseek_glm` | best_vs_cot_cpc | `cups_lambda_0.08` | direct | +0.5 pp | [+0.0, +1.5] | 1.000 |
| `math500_core_deepseek_glm` | best_vs_cot_cpc | `cups_lambda_0.08` | cot | +1.0 pp | [-4.5, +6.5] | 0.860 |
| `math500_minimax_m3` | best_accuracy | `cups_lambda_0.02` | direct | +5.0 pp | [+1.0, +9.0] | 0.062 |
| `math500_minimax_m3` | best_accuracy | `cups_lambda_0.02` | cot | +6.0 pp | [+0.0, +12.0] | 0.109 |
| `math500_minimax_m3` | best_vs_cot_cpc | `cups_lambda_0.02` | direct | +5.0 pp | [+1.0, +9.0] | 0.062 |
| `math500_minimax_m3` | best_vs_cot_cpc | `cups_lambda_0.02` | cot | +6.0 pp | [+0.0, +12.0] | 0.109 |
| `math500_qwen_clean` | best_accuracy | `cups_lambda_0` | direct | +1.0 pp | [+0.0, +3.0] | 1.000 |
| `math500_qwen_clean` | best_accuracy | `cups_lambda_0` | cot | +2.0 pp | [+0.0, +5.0] | 0.500 |
| `math500_qwen_clean` | best_vs_cot_cpc | `cups_lambda_0.15` | direct | +0.0 pp | [+0.0, +0.0] | 1.000 |
| `math500_qwen_clean` | best_vs_cot_cpc | `cups_lambda_0.15` | cot | +1.0 pp | [+0.0, +3.0] | 1.000 |

## Missing/Excluded CUPS Scopes

- `gsm8k;mmlu_pro-deepseek_v4_flash` (nonmedical_deepseek_flash): CUPS leave_one_model_out requires at least two groups; found 1.
- `math500-minimax_m3` (math500_minimax_m3): CUPS leave_one_dataset_model_out requires at least two groups; found 1.
- `math500-minimax_m3` (math500_minimax_m3): CUPS leave_one_model_out requires at least two groups; found 1.
- `math500-qwen_3_7_max_preview` (math500_qwen_clean): CUPS leave_one_dataset_model_out requires at least two groups; found 1.
- `math500-qwen_3_7_max_preview` (math500_qwen_clean): CUPS leave_one_model_out requires at least two groups; found 1.
- `math500-kimi_k3` (math500_kimi_k3): raw ceiling-control exists, but router_features.csv is absent; CUPS requires feature rows.

## Files

- `cups_best_policy_tests.csv`: scope-level CUPS vs direct/CoT paired tests.
- `cups_per_dataset_model_metrics.csv`: dataset-model paired tests.
- `cups_strategy_mix_by_scope.csv`: selected strategy distributions.
- `cups_curve_all_scopes.csv`: all CUPS and baseline curve rows.
- `generated_cups/*`: regenerated CUPS prediction/curve reports per scope.
