from pathlib import Path

import csv

from llm.observation import Observation, append_summary_row, render_markdown


def observation() -> Observation:
    return Observation(
        output_path=Path("tracks/from-scratch/runs/observations/example.md"),
        checkpoint_path=Path("tracks/from-scratch/checkpoints/example.pt"),
        tokens_path=Path("tracks/from-scratch/data/processed/example.pt"),
        checkpoint_step=10,
        checkpoint_metadata={"parameter_count": 100},
        tokens_metadata={"vocab_size": 256},
        validation_loss=1.25,
        validation_perplexity=3.49,
        run_id="example",
        prompt="KING:",
        prompt_id="king",
        seed=1337,
        samples=2,
        max_new_tokens=20,
        temperature=1.0,
        top_k=10,
        sampling=True,
        generated_samples=["KING: first", "KING: second"],
    )


def test_render_markdown_includes_metrics_settings_and_samples() -> None:
    markdown = render_markdown(observation())

    assert "validation" in markdown.lower()
    assert "1.2500" in markdown
    assert "3.49" in markdown
    assert "KING:" in markdown
    assert "seed: 1337" in markdown
    assert "KING: first" in markdown
    assert "KING: second" in markdown


def test_append_summary_row_writes_header_once(tmp_path) -> None:
    path = tmp_path / "summary.csv"

    append_summary_row(path, observation())
    append_summary_row(path, observation())

    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("output_path,checkpoint_path,tokens_path")
    assert len(rows) == 2
    assert rows[0]["checkpoint_step"] == "10"
    assert rows[0]["run_id"] == "example"
    assert rows[0]["prompt_id"] == "king"
    assert rows[0]["prompt"] == "KING:"
