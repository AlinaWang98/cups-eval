# Cross-Domain Bootstrap CI Table

Generated from completed local runs only. No provider calls were made.

Bootstrap iterations: 1000; seed: 42.

| Scope | Method | Accuracy [95% CI] | Cost x Direct [95% CI] | Relative CPC [95% CI] |
|---|---|---:|---:|---:|
| gsm8k_glm_kimi | `direct` | 0.940 [0.905, 0.970] | 1.00 [1.00, 1.00] | 1.064 [1.031, 1.105] |
| gsm8k_glm_kimi | `cot` | 0.945 [0.910, 0.975] | 0.91 [0.84, 0.99] | 0.965 [0.879, 1.063] |
| gsm8k_glm_kimi | `planner_verifier` | 0.955 [0.920, 0.980] | 1.97 [1.86, 2.08] | 2.061 [1.930, 2.207] |
| gsm8k_glm_kimi | `debate_2_judge` | 0.955 [0.925, 0.980] | 3.12 [2.94, 3.29] | 3.262 [3.050, 3.492] |
| gsm8k_glm_kimi | `direct_first_oracle_any` | 0.985 [0.965, 1.000] | 1.04 [1.01, 1.07] | 1.058 [1.026, 1.094] |
| gsm8k_glm_kimi | `learned_rule_list` | 0.980 [0.960, 0.995] | 1.07 [1.04, 1.11] | 1.094 [1.044, 1.149] |
| mmlu_pro_glm_kimi | `direct` | 0.735 [0.680, 0.795] | 1.00 [1.00, 1.00] | 1.361 [1.258, 1.471] |
| mmlu_pro_glm_kimi | `cot` | 0.760 [0.700, 0.820] | 0.99 [0.95, 1.04] | 1.309 [1.194, 1.437] |
| mmlu_pro_glm_kimi | `planner_verifier` | 0.720 [0.655, 0.780] | 1.99 [1.92, 2.07] | 2.765 [2.529, 3.076] |
| mmlu_pro_glm_kimi | `debate_2_judge` | 0.740 [0.680, 0.800] | 3.05 [2.93, 3.17] | 4.116 [3.763, 4.491] |
| mmlu_pro_glm_kimi | `direct_first_oracle_any` | 0.825 [0.775, 0.875] | 1.16 [1.09, 1.25] | 1.411 [1.291, 1.541] |
| mmlu_pro_glm_kimi | `learned_rule_list` | 0.765 [0.705, 0.825] | 1.53 [1.45, 1.61] | 2.004 [1.784, 2.236] |
