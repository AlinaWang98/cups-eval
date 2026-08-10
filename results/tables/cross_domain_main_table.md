# Cross-Domain Main Table

Generated from existing completed runs only. No provider calls were made while producing this report.

## Pooled Main Results

| Scope | Method | Accuracy | Cost x Direct | Relative CPC | Escalation |
|---|---|---:|---:|---:|---:|
| medical | `direct` | 0.706 | 1.00 | 1.416 | NA |
| medical | `cot` | 0.729 | 1.00 | 1.366 | NA |
| medical | `planner_verifier` | 0.730 | 2.01 | 2.752 | NA |
| medical | `debate_2_judge` | 0.733 | 3.03 | 4.132 | NA |
| medical | `direct_first_oracle_any` | 0.799 | 1.17 | 1.467 | 9.2% |
| medical | `learned_rule_list` | 0.731 | 1.42 | 1.944 | 31.8% |
| nonmedical | `direct` | 0.695 | 1.00 | 1.439 | NA |
| nonmedical | `cot` | 0.690 | 1.01 | 1.459 | NA |
| nonmedical | `planner_verifier` | 0.800 | 2.15 | 2.682 | NA |
| nonmedical | `debate_2_judge` | 0.775 | 3.29 | 4.250 | NA |
| nonmedical | `direct_first_oracle_any` | 0.845 | 1.20 | 1.417 | 15.0% |
| nonmedical | `learned_rule_list` | 0.770 | 1.53 | 1.993 | 38.5% |

## Key Contrasts

### Medical

- CoT vs direct: 0.022 accuracy delta; relative CPC 1.366 vs direct 1.416.
- Planner-verifier vs direct: 0.024 accuracy delta at 2.01x cost.
- Debate vs direct: 0.026 accuracy delta at 3.03x cost.
- Direct-first oracle headroom: 0.092 accuracy delta at 1.17x cost; relative CPC 1.467.
- Learned router: accuracy 0.731, cost 1.42x, retained oracle gain 0.270.

### Nonmedical

- CoT vs direct: -0.005 accuracy delta; relative CPC 1.459 vs direct 1.439.
- Planner-verifier vs direct: 0.105 accuracy delta at 2.15x cost.
- Debate vs direct: 0.080 accuracy delta at 3.29x cost.
- Direct-first oracle headroom: 0.150 accuracy delta at 1.20x cost; relative CPC 1.417.
- Learned router: accuracy 0.770, cost 1.53x, retained oracle gain 0.500.

## Updated CUPS Claim

The cross-domain table supports a stronger but more nuanced thesis:

> Collaboration is task-dependent. Under cost-fair evaluation, cheap CoT is a strong baseline in medical QA, planner/debate can be highly valuable on arithmetic reasoning, and debate can harm broad knowledge MCQA. Oracle routing exposes sparse opportunity across domains, but current deployable routers still overpay relative to that opportunity.

## What This Rules Out

- It rules out a simple claim that multi-agent collaboration is always better.
- It also rules out a simple claim that CoT always absorbs collaboration gains, because GSM8K shows large planner/debate gains.
- It still does not support a claim that the current router is a solved deployable method.

## Files

- `cross_domain_main_table.csv`
- `cross_domain_dataset_model_table.csv`
- `cross_domain_ci/cross_domain_ci_table.csv`
- `cross_domain_ci/cross_domain_ci_table.md`
