from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    load_dotenv()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return resolve_llm_profile(config)


def load_dotenv(path: Path | str | None = None) -> None:
    dotenv_path = Path(path) if path is not None else Path.cwd() / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = _clean_env_value(value)
        os.environ.setdefault(key, value)


def resolve_llm_profile(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm")
    if not isinstance(llm, dict) or "profiles" not in llm:
        return config
    profiles = llm.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("llm.profiles must be a mapping")
    profile_env = str(llm.get("active_profile_env", "CUPS_MODEL_PROFILE"))
    active_profile = os.environ.get(profile_env) or str(
        llm.get("active_profile", "")
    )
    if active_profile not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles))
        raise ValueError(
            f"Unknown llm profile {active_profile!r}; available profiles: {available}"
        )

    defaults = llm.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError("llm.defaults must be a mapping")
    selected = profiles[active_profile]
    if not isinstance(selected, dict):
        raise ValueError(f"llm.profiles.{active_profile} must be a mapping")

    resolved_llm = dict(defaults)
    resolved_llm.update(selected)
    resolved_llm["profile"] = active_profile
    resolved_llm.pop("pricing", None)

    resolved_config = dict(config)
    resolved_config["llm"] = resolved_llm
    if "pricing" in selected:
        pricing = selected["pricing"]
        if not isinstance(pricing, dict):
            raise ValueError(f"llm.profiles.{active_profile}.pricing must be a mapping")
        resolved_config["pricing"] = pricing
    return resolved_config


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return cleaned


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, Any],
) -> float:
    input_rate = float(pricing.get("input_per_million_usd", 0.0))
    output_rate = float(pricing.get("output_per_million_usd", 0.0))
    return (prompt_tokens / 1_000_000.0 * input_rate) + (
        completion_tokens / 1_000_000.0 * output_rate
    )
