from pathlib import Path

import pytest

from lucid_director.core.config import Config, ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_stub_config_loads() -> None:
    config = load_config(REPO_ROOT / "configs" / "stub.yaml")

    assert config.frame_source.type == "stub"
    assert config.frame_source.resolution == (320, 200)
    assert config.pipeline.resolution == (384, 240)
    assert config.pipeline.initial.strength == 0.5
    assert config.agent.budgets.min_fps == 20


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")) == Config()


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "pipeline:\n  initial:\n    stregth: 0.5\n")

    with pytest.raises(ConfigError, match="stregth"):
        load_config(path)


def test_out_of_range_value_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "pipeline:\n  initial:\n    strength: 1.5\n")

    with pytest.raises(ConfigError, match="strength"):
        load_config(path)


def test_resolution_must_be_multiple_of_8(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="multiple of 8"):
        load_config(write(tmp_path, "pipeline:\n  resolution: [384, 250]\n"))


def test_non_mapping_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, "- just\n- a list\n"))


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.yaml")


def test_options_pass_through_unvalidated(tmp_path: Path) -> None:
    path = write(tmp_path, "frame_source:\n  type: vizdoom\n  options: {scenario: basic}\n")

    assert load_config(path).frame_source.options == {"scenario": "basic"}
