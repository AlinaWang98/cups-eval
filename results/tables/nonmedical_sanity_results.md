# Non-Medical Sanity Results: DeepSeek v4 Flash

Last updated: 2026-07-29

This run tested whether the cost-fair collaboration pattern observed on medical QA also appears on non-medical reasoning benchmarks.

## Scope

| Dataset | Model | Sample | Strategies | Status |
|---|---|---:|---|---|
| GSM8K test | deepseek_v4_flash | 100 | direct, cot, planner_verifier, debate_2_judge | complete |
| MMLU-Pro test | deepseek_v4_flash | 100 | direct, cot, planner_verifier, debate_2_judge | complete |

Completed raw rows:

- GSM8K: 400/400
- MMLU-Pro: 400/400

Parse success:

- GSM8K: 100%
- MMLU-Pro: 100%

Cost:

- Completed analysis runs: $0.194797
- Partial duplicate MMLU-Pro run from the interrupted serial batch: $0.001833
- Total provider spend including partial duplicate: $0.196630

The partial duplicate run is excluded from all analysis because it has no complete summary or router feature file.

## Main Results

| Dataset | Method | Accuracy | Cost USD | Cost x Direct | Notes |
|---|---|---:|---:|---:|---|
| GSM8K | direct | 0.690 | 0.007426 | 1.00 | baseline |
| GSM8K | cot | 0.680 | 0.007620 | 1.03 | no gain over direct |
| GSM8K | planner_verifier | 0.890 | 0.016331 | 2.20 | large gain, significant paired test |
| GSM8K | debate_2_judge | 0.880 | 0.025134 | 3.39 | large gain, expensive |
| GSM8K | oracle_router | 0.930 | 0.009378 | 1.26 | large selective headroom |
| MMLU-Pro | direct | 0.700 | 0.018735 | 1.00 | baseline |
| MMLU-Pro | cot | 0.700 | 0.018710 | 1.00 | matches direct |
| MMLU-Pro | planner_verifier | 0.710 | 0.039807 | 2.13 | tiny, noisy gain |
| MMLU-Pro | debate_2_judge | 0.670 | 0.061034 | 3.26 | harms accuracy |
| MMLU-Pro | oracle_router | 0.760 | 0.018706 | 1.00 | modest selective headroom |

## Pooled Non-Medical Router Analysis

Across 200 questions:

| Policy | Accuracy | Cost x | Cost per correct |
|---|---:|---:|---:|
| direct | 0.695 | 1.00 | $0.000188 |
| always_cot | 0.690 | 1.01 | $0.000191 |
| always_planner_verifier | 0.800 | 2.15 | $0.000351 |
| always_debate_2_judge | 0.775 | 3.29 | $0.000556 |
| direct_first_oracle_any | 0.845 | 1.20 | $0.000185 |
| learned_rule_list | 0.770 | 1.53 | $0.000261 |

Key router result:

- Oracle opportunity: 0.695 -> 0.845 (+15.0pp) at 1.20x direct cost.
- Learned rule-list: 0.695 -> 0.770 (+7.5pp) at 1.53x direct cost.
- Retained oracle gain: 50.0%.
- The learned router captures meaningful accuracy but still overpays relative to the oracle.

## Rescue/Harm

| Dataset | Strategy | Rescue | Harm | Rescue - Harm |
|---|---|---:|---:|---:|
| GSM8K | cot | 4 | 5 | -1pp |
| GSM8K | planner_verifier | 21 | 1 | +20pp |
| GSM8K | debate_2_judge | 20 | 1 | +19pp |
| MMLU-Pro | cot | 4 | 4 | 0pp |
| MMLU-Pro | planner_verifier | 5 | 4 | +1pp |
| MMLU-Pro | debate_2_judge | 3 | 6 | -3pp |

This is the strongest new finding: collaboration behavior is task-dependent. On GSM8K, planner/debate are highly rescuing; on MMLU-Pro, debate can be actively harmful.

## Interpretation

The non-medical sanity run changes the story.

Medical Phase 2 suggested that cheap CoT absorbs much of the observed collaboration gain. Non-medical DeepSeek flash does not uniformly replicate that. GSM8K shows a much stronger benefit from planner-verifier and debate, while MMLU-Pro shows only modest oracle headroom and weak or harmful always-on collaboration.

The stronger CUPS framing is therefore:

> Collaboration is not universally wasteful or universally beneficial. Cost-fair evaluation reveals task-dependent collaboration utility: cheap CoT is sufficient on some QA settings, planner/debate can strongly help on arithmetic reasoning, and always-on debate can harm or overpay on broad knowledge MCQA. Selective routing has real oracle headroom, but current simple routers still over-route.

## Implications for CUPS

Supported more strongly after this run:

- Cost-fair evaluation changes conclusions about multi-agent reasoning.
- Always-on collaboration is not a reliable default.
- Oracle opportunity exists beyond medical QA.
- Collaboration utility is strongly task-dependent.

Still not supported:

- A deployable router dominates direct or CoT on cost-normalized metrics across tasks.
- CoT always absorbs collaboration gains.
- Multi-agent collaboration is generally superior.

## Next Experimental Decision

Do not scale Kimi/GLM yet.

The next best zero/low-cost step is to produce a cross-domain main table:

- medical pooled GLM/Kimi;
- non-medical DeepSeek flash;
- per-dataset direct, CoT, planner, debate, oracle, learned router;
- cost per correct and bootstrap intervals.

Only after this table is stable should we decide whether to run GLM/Kimi on GSM8K/MMLU-Pro.

## Artifacts

- `runs/phase2_nonmedical_sanity_deepseek_flash/20260729T020951Z/runs/gsm8k/deepseek_v4_flash/20260729T020952Z/summary.md`
- `runs/phase2_nonmedical_sanity_deepseek_flash/20260729T020951Z/runs/mmlu_pro/deepseek_v4_flash/20260729T021858Z/summary.md`
- `runs/phase2_nonmedical_sanity_deepseek_flash/20260729T020951Z/analysis_offline_router_nonmedical/offline_router_report.md`
- `runs/phase2_nonmedical_sanity_deepseek_flash/20260729T020951Z/analysis_mechanistic_nonmedical/mechanistic_report.md`
- `runs/phase2_nonmedical_sanity_deepseek_flash/20260729T020951Z/analysis_cot_router_nonmedical/phase2_6_report.md`

