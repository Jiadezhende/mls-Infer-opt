from __future__ import annotations

from mls_infer_opt.llm import LLMConfig


def test_config_reads_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'OPENAI_API_KEY="sk-test"',
                "OPENAI_BASE_URL=https://example.test/v1",
                "MLS_LLM_MODEL=gpt-test",
                "MLS_LLM_MAX_TOOL_ROUNDS=7",
            ]
        ),
        encoding="utf-8",
    )

    config = LLMConfig.from_env({}, env_file=env_file)
    assert config.api_key == "sk-test"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "gpt-test"
    assert config.max_tool_rounds == 7


def test_dotenv_overrides_exported_env(tmp_path):
    # 本服务 .env 对凭证逐键权威，环境变量压不过 .env 里设过的键。
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\nMLS_LLM_MODEL=from-file\n", encoding="utf-8")

    config = LLMConfig.from_env(
        {"OPENAI_API_KEY": "from-shell", "MLS_LLM_MODEL": "from-shell"},
        env_file=env_file,
    )
    assert config.api_key == "from-file"
    assert config.model == "from-file"


def test_exported_env_fills_keys_absent_from_dotenv(tmp_path):
    # .env 未设置（或留空）该键时，环境变量兜底。
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\nOPENAI_BASE_URL=\n", encoding="utf-8")

    config = LLMConfig.from_env(
        {"OPENAI_BASE_URL": "https://shell.test/v1", "MLS_LLM_MODEL": "from-shell"},
        env_file=env_file,
    )
    assert config.api_key == "from-file"  # .env 权威
    assert config.base_url == "https://shell.test/v1"  # .env 留空 → env 兜底
    assert config.model == "from-shell"  # .env 没这键 → env 兜底
