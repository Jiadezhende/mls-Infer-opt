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


def test_exported_env_overrides_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\nMLS_LLM_MODEL=from-file\n", encoding="utf-8")

    config = LLMConfig.from_env(
        {"OPENAI_API_KEY": "from-shell", "MLS_LLM_MODEL": "from-shell"},
        env_file=env_file,
    )
    assert config.api_key == "from-shell"
    assert config.model == "from-shell"
