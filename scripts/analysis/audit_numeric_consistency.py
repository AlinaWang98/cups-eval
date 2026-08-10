from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = RELEASE_ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"
SOURCE_TABLE_DIR = RESULTS_DIR / "source_tables"
CUPS_DIR = RESULTS_DIR / "cups"
FIGURE_DIR = RESULTS_DIR / "figures"
REPORT_PATH = RESULTS_DIR / "numeric_audit.md"

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".yaml", ".yml", ".txt"}
SKIP_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "runs"}
LOCAL_USER_PATH = "/" + "Users" + "/"
SENSITIVE_PATTERNS = [
    re.compile(re.escape(LOCAL_USER_PATH)),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{8,}", re.IGNORECASE),
]


@dataclass
class Check:
    name: str
    status: str
    detail: str


def pass_check(name: str, detail: str) -> Check:
    return Check(name, "PASS", detail)


def fail_check(name: str, detail: str) -> Check:
    return Check(name, "FAIL", detail)


def check_required_files() -> list[Check]:
    paths = [
        TABLE_DIR / "cross_domain_main_table.csv",
        TABLE_DIR / "math500_capability_gradient.csv",
        CUPS_DIR / "cups_best_policy_tests.csv",
        CUPS_DIR / "cups_curve_all_scopes.csv",
        SOURCE_TABLE_DIR / "paired_stats_math500_deepseek_v4_flash.csv",
        SOURCE_TABLE_DIR / "paired_stats_math500_glm_5_2.csv",
        SOURCE_TABLE_DIR / "paired_stats_math500_minimax_m3.csv",
        SOURCE_TABLE_DIR / "paired_stats_math500_qwen_3_7_max_preview.csv",
    ]
    missing = [path.relative_to(RELEASE_ROOT).as_posix() for path in paths if not path.exists()]
    if missing:
        return [fail_check("Required public artifacts", "missing: " + ", ".join(missing))]
    return [pass_check("Required public artifacts", f"{len(paths)} files found")]


def check_no_local_paths() -> Check:
    hits: list[str] = []
    for path in sorted(_iter_release_text_files()):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                hits.append(path.relative_to(RELEASE_ROOT).as_posix())
                break
    if hits:
        return fail_check("Local path / token scan", "hits: " + ", ".join(hits[:20]))
    return pass_check("Local path / token scan", "no local user paths, sk-style keys, or bearer tokens found")


def _iter_release_text_files() -> list[Path]:
    files: list[Path] = []
    for path in RELEASE_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name == ".env.example"):
            files.append(path)
    return files


def check_cross_domain_table() -> Check:
    df = pd.read_csv(TABLE_DIR / "cross_domain_main_table.csv")
    required_methods = {
        "direct",
        "cot",
        "planner_verifier",
        "debate_2_judge",
        "direct_first_oracle_any",
        "learned_rule_list",
    }
    problems = []
    for scope in ["medical", "nonmedical"]:
        methods = set(df[df["scope"] == scope]["method"])
        missing = sorted(required_methods - methods)
        if missing:
            problems.append(f"{scope} missing {missing}")
    if problems:
        return fail_check("Cross-domain main table", "; ".join(problems))
    return pass_check("Cross-domain main table", f"{len(df)} strategy rows across medical/nonmedical scopes")


def check_cups_rows() -> Check:
    df = pd.read_csv(CUPS_DIR / "cups_best_policy_tests.csv")
    scope_ids = ["medical_glm_kimi", "nonmedical_glm_kimi", "nonmedical_deepseek_flash"]
    rows = []
    for scope_id in scope_ids:
        sub = df[
            (df["scope_id"] == scope_id)
            & (df["evaluation_regime"] == "item_oof")
            & (df["selection_type"] == "best_accuracy")
            & (df["baseline"] == "cot")
        ]
        if sub.empty:
            return fail_check("CUPS item-OOF rows", f"missing best_accuracy/cot row for {scope_id}")
        row = sub.iloc[0]
        rows.append(
            f"{scope_id}: acc={float(row['selected_accuracy']):.3f}, "
            f"cost_x_direct={float(row['cups_cost_multiplier']):.2f}"
        )
    return pass_check("CUPS item-OOF rows", "; ".join(rows))


def check_paired_source_tables() -> Check:
    paths = {
        "DeepSeek": SOURCE_TABLE_DIR / "paired_stats_math500_deepseek_v4_flash.csv",
        "GLM-5.2": SOURCE_TABLE_DIR / "paired_stats_math500_glm_5_2.csv",
        "MiniMax": SOURCE_TABLE_DIR / "paired_stats_math500_minimax_m3.csv",
        "Qwen": SOURCE_TABLE_DIR / "paired_stats_math500_qwen_3_7_max_preview.csv",
    }
    required = {"cot", "planner_verifier", "debate_2_judge"}
    details = []
    for label, path in paths.items():
        df = pd.read_csv(path)
        missing = sorted(required - set(df["strategy"]))
        if missing:
            return fail_check("Rescue/harm source tables", f"{label} missing {missing}")
        details.append(f"{label} n={int(df['n'].max())}")
    return pass_check("Rescue/harm source tables", "; ".join(details))


def check_figure_files() -> Check:
    stems = [
        "figure_1_cost_accuracy_frontier",
        "figure_2_capability_decay",
        "figure_3_retained_oracle_gain",
        "figure_4_rescue_harm",
    ]
    missing = []
    for stem in stems:
        for suffix in [".png", ".pdf"]:
            path = FIGURE_DIR / f"{stem}{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                missing.append(path.relative_to(RELEASE_ROOT).as_posix())
    if missing:
        return fail_check("Figure outputs", "missing/empty: " + ", ".join(missing))
    return pass_check("Figure outputs", f"{len(stems)} PNG/PDF figure pairs found")


def main() -> None:
    checks: list[Check] = []
    checks.extend(check_required_files())
    checks.append(check_no_local_paths())
    if not any(check.status == "FAIL" for check in checks):
        checks.append(check_cross_domain_table())
        checks.append(check_cups_rows())
        checks.append(check_paired_source_tables())
        checks.append(check_figure_files())

    failures = [check for check in checks if check.status != "PASS"]
    lines = [
        "# Numeric Consistency Audit",
        "",
        "This public audit checks that release-bundled result tables are internally consistent and path-safe.",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for check in checks:
        lines.append(f"| {check.name} | {check.status} | {check.detail.replace('|', '/')} |")
    lines.extend(["", f"Summary: {len(checks) - len(failures)} PASS, {len(failures)} FAIL."])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH.relative_to(RELEASE_ROOT))
    print(f"{len(checks) - len(failures)} PASS, {len(failures)} FAIL")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
