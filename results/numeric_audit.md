# Numeric Consistency Audit

This public audit checks that release-bundled result tables are internally consistent and path-safe.

| Check | Status | Detail |
|---|---:|---|
| Required public artifacts | PASS | 8 files found |
| Local path / token scan | PASS | no local user paths, sk-style keys, or bearer tokens found |
| Cross-domain main table | PASS | 12 strategy rows across medical/nonmedical scopes |
| CUPS item-OOF rows | PASS | medical_glm_kimi: acc=0.735, cost_x_direct=1.44; nonmedical_glm_kimi: acc=0.882, cost_x_direct=1.24; nonmedical_deepseek_flash: acc=0.785, cost_x_direct=1.90 |
| Rescue/harm source tables | PASS | DeepSeek n=100; GLM-5.2 n=100; MiniMax n=100; Qwen n=100 |
| Figure outputs | PASS | 4 PNG/PDF figure pairs found |

Summary: 6 PASS, 0 FAIL.
