import pytest

from llm.prepare_data import load_non_empty_text


def test_load_non_empty_text_rejects_empty_input(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        load_non_empty_text(path)
