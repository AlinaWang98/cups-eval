import os
from pathlib import Path

import pytest
import yaml

from cups_eval.config import load_config
from cups_eval.llm import OpenAICompatibleLLMClient


def test_load_config_reads_dotenv_without_overriding_existing_environment(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EXISTING_KEY", "from-shell")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "CUPS_MODEL_PROFILE=glm_5_2",
                "GLM_API_KEY=from-dotenv",
                "EXISTING_KEY=from-dotenv",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  name: test
llm:
  active_profile: deepseek_v4_pro
  active_profile_env: CUPS_MODEL_PROFILE
  defaults:
    temperature: 0.2
    max_tokens: 128
  profiles:
    deepseek_v4_pro:
      provider: deepseek
      api_key_env: DEEPSEEK_API_KEY
      model: deepseek-v4-pro
      base_url: https://api.deepseek.com
    glm_5_2:
      provider: glm
      api_key_env: GLM_API_KEY
      model: glm-5.2
      base_url: https://open.bigmodel.cn/api/paas/v4/
      pricing:
        input_per_million_usd: 0.1
        output_per_million_usd: 0.2
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert os.environ["GLM_API_KEY"] == "from-dotenv"
    assert os.environ["EXISTING_KEY"] == "from-shell"
    assert config["llm"]["profile"] == "glm_5_2"
    assert config["llm"]["model"] == "glm-5.2"
    assert config["llm"]["temperature"] == 0.2
    assert config["pricing"]["input_per_million_usd"] == 0.1


def test_load_config_rejects_unknown_model_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("CUPS_MODEL_PROFILE", "missing")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  active_profile_env: CUPS_MODEL_PROFILE
  profiles:
    deepseek_v4_pro:
      provider: deepseek
      api_key_env: DEEPSEEK_API_KEY
      model: deepseek-v4-pro
      base_url: https://api.deepseek.com
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown llm profile"):
        load_config(config_path)


def test_openai_compatible_client_allows_base_url_env_override(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("TEST_API_KEY", "secret")
    monkeypatch.setenv("TEST_MODEL", "provider-model")
    monkeypatch.setenv("TEST_BASE_URL", "https://override.example/v1")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    client = OpenAICompatibleLLMClient.from_config(
        {
            "llm": {
                "api_key_env": "TEST_API_KEY",
                "model_env": "TEST_MODEL",
                "base_url_env": "TEST_BASE_URL",
                "base_url": "https://default.example/v1",
                "model": "default-model",
            }
        }
    )

    assert client.model == "provider-model"
    assert captured["api_key"] == "secret"
    assert captured["base_url"] == "https://override.example/v1"


def test_env_example_declares_every_model_env_used_by_profiles():
    config = yaml.safe_load(Path("configs/mvp.medmcqa.yaml").read_text())
    env_example = Path(".env.example").read_text()
    model_envs = {
        str(profile["model_env"])
        for profile in config["llm"]["profiles"].values()
    }

    missing = sorted(env_name for env_name in model_envs if f"{env_name}=" not in env_example)

    assert missing == []


def test_qwen_preview_profile_matches_dashscope_openai_compatible_settings():
    config = yaml.safe_load(Path("configs/mvp.medmcqa.yaml").read_text())
    env_example = Path(".env.example").read_text()

    profile = config["llm"]["profiles"]["qwen_3_7_max_preview"]

    assert profile["provider"] == "qwen"
    assert profile["api_key_env"] == "DASHSCOPE_API_KEY"
    assert profile["base_url_env"] == "QWEN_BASE_URL"
    assert profile["model_env"] == "QWEN_MODEL"
    assert profile["model"] == "qwen3.7-max-preview"
    assert profile["pricing"] == {
        "input_per_million_usd": 1.77,
        "output_per_million_usd": 5.32,
    }
    assert profile["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert "qwen_3_7_max_preview" in env_example
