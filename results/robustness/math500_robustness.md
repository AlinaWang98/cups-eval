# MATH-500 Robustness Pack

This pack is generated offline from existing run artifacts. No provider calls are made.

## Inclusion

| model | sample | raw rows | item rows | status |
|---|---|---:|---:|---|
| DeepSeek v4 flash | full500_anchor | 2500 | 2500 | included_complete |
| GLM-5.2 | clean100 | 400 | 400 | included_complete |
| MiniMax-M3 | clean100 | 400 | 400 | included_complete |
| Qwen3.7-Max-Preview | clean100 | 400 | 400 | included_complete |
| GPT-5.5 | frontier50_full5 | 250 | 250 | included_complete |
| claude-sonnet-5 | frontier50_full5 | 250 | 250 | included_complete |
| deepseek_v4_flash | aborted_20260805_scratch | 142 | 0 | excluded_aborted_incomplete |
| glm_5_2 | aborted_20260805_scratch | 4 | 0 | excluded_aborted_incomplete |
| qwen_3_7_max_preview | aborted_20260805_scratch | 4 | 0 | excluded_aborted_incomplete |

## Key Paired Results

| model | sample | comparison | n | delta | 95% CI | p | rescues | harms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek v4 flash | full500_anchor | self_consistency_3 | 500 | +0.000 | [-0.030, +0.030] | 1.000 | 28 | 28 |
| DeepSeek v4 flash | full500_anchor | planner_verifier | 500 | +0.114 | [+0.080, +0.150] | 0.000 | 73 | 16 |
| DeepSeek v4 flash | full500_anchor | debate_2_judge | 500 | +0.170 | [+0.132, +0.208] | 0.000 | 97 | 12 |
| GLM-5.2 | clean100 | planner_verifier | 100 | +0.120 | [+0.030, +0.210] | 0.023 | 18 | 6 |
| GLM-5.2 | clean100 | debate_2_judge | 100 | +0.130 | [+0.030, +0.220] | 0.015 | 19 | 6 |
| MiniMax-M3 | clean100 | planner_verifier | 100 | +0.040 | [+0.000, +0.090] | 0.219 | 5 | 1 |
| MiniMax-M3 | clean100 | debate_2_judge | 100 | +0.050 | [+0.000, +0.110] | 0.180 | 7 | 2 |
| Qwen3.7-Max-Preview | clean100 | planner_verifier | 100 | +0.010 | [-0.020, +0.050] | 1.000 | 2 | 1 |
| Qwen3.7-Max-Preview | clean100 | debate_2_judge | 100 | +0.000 | [-0.030, +0.030] | 1.000 | 1 | 1 |
| GPT-5.5 | frontier50_full5 | self_consistency_3 | 50 | +0.000 | [-0.060, +0.060] | 1.000 | 1 | 1 |
| GPT-5.5 | frontier50_full5 | planner_verifier | 50 | +0.000 | [-0.060, +0.060] | 1.000 | 1 | 1 |
| GPT-5.5 | frontier50_full5 | debate_2_judge | 50 | +0.040 | [+0.000, +0.100] | 0.500 | 2 | 0 |
| claude-sonnet-5 | frontier50_full5 | self_consistency_3 | 50 | +0.000 | [-0.120, +0.100] | 1.000 | 4 | 4 |
| claude-sonnet-5 | frontier50_full5 | planner_verifier | 50 | +0.160 | [+0.060, +0.260] | 0.008 | 8 | 0 |
| claude-sonnet-5 | frontier50_full5 | debate_2_judge | 50 | +0.140 | [+0.040, +0.240] | 0.039 | 8 | 1 |

## DeepSeek 500-Question Prefix Stability

| prefix n | strategy | accuracy | gain vs direct |
|---:|---|---:|---:|
| 50 | direct | 0.440 | 0.000 |
| 50 | planner_verifier | 0.460 | 0.020 |
| 50 | debate_2_judge | 0.600 | 0.160 |
| 50 | oracle_any | 0.700 | 0.260 |
| 100 | direct | 0.440 | 0.000 |
| 100 | planner_verifier | 0.510 | 0.070 |
| 100 | debate_2_judge | 0.570 | 0.130 |
| 100 | oracle_any | 0.670 | 0.230 |
| 200 | direct | 0.460 | 0.000 |
| 200 | planner_verifier | 0.560 | 0.100 |
| 200 | debate_2_judge | 0.590 | 0.130 |
| 200 | oracle_any | 0.695 | 0.235 |
| 500 | direct | 0.432 | 0.000 |
| 500 | planner_verifier | 0.546 | 0.114 |
| 500 | debate_2_judge | 0.602 | 0.170 |
| 500 | oracle_any | 0.688 | 0.256 |

## DeepSeek Cost Sensitivity

| strategy | accuracy | cost x direct | tokens x direct | relative CPC |
|---|---:|---:|---:|---:|
| direct | 0.432 | 1.000 | 1.000 | 1.000 |
| cot | 0.430 | 1.109 | 1.077 | 1.114 |
| self_consistency_3 | 0.432 | 3.011 | 3.026 | 3.011 |
| planner_verifier | 0.546 | 2.075 | 2.155 | 1.642 |
| debate_2_judge | 0.602 | 3.217 | 3.361 | 2.308 |

## Math Equivalence Audit

| model | rows | equivalent | not equivalent | unresolved |
|---|---:|---:|---:|---:|
| DeepSeek v4 flash | 2500 | 1291 | 801 | 408 |
| GLM-5.2 | 400 | 315 | 7 | 78 |
| MiniMax-M3 | 400 | 364 | 5 | 31 |
| Qwen3.7-Max-Preview | 400 | 376 | 6 | 18 |
| GPT-5.5 | 250 | 239 | 1 | 10 |
| claude-sonnet-5 | 250 | 218 | 23 | 9 |
