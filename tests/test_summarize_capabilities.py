import csv
import json
from pathlib import Path

import pytest

from llm.leverage import summarize_capabilities


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_count_capabilities_rejects_unknown_capability(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": "row-1", "capability": "legacy_label"}])

    with pytest.raises(ValueError, match="unknown capability"):
        summarize_capabilities.count_capabilities(path)


def test_summary_rows_counts_layers_and_reviewed_deficits(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.jsonl"
    reviewed = tmp_path / "reviewed.jsonl"
    evals = tmp_path / "evals.jsonl"
    write_jsonl(
        seeds,
        [
            {"id": "seed-1", "capability": "reasoning"},
            {"id": "seed-2", "capability": "tool_use"},
            {"id": "seed-3", "capability": "tool_use"},
        ],
    )
    write_jsonl(reviewed, [{"id": "reviewed-1", "capability": "reasoning"}])
    write_jsonl(
        evals,
        [
            {"id": "eval-1", "capability": "reasoning"},
            {"id": "eval-2", "capability": "coding"},
        ],
    )

    rows = summarize_capabilities.summary_rows(
        seeds_path=seeds,
        reviewed_path=reviewed,
        eval_paths=[evals],
        reviewed_targets={
            "instruction_following": 0,
            "reasoning": 2,
            "coding": 1,
            "knowledge_qa": 0,
            "summarization_transformation": 0,
            "tool_use": 3,
        },
    )

    by_capability = {row["capability"]: row for row in rows}
    assert by_capability["reasoning"] == {
        "capability": "reasoning",
        "seed_count": "1",
        "reviewed_count": "1",
        "eval_count": "1",
        "reviewed_target": "2",
        "reviewed_deficit": "1",
    }
    assert by_capability["tool_use"]["seed_count"] == "2"
    assert by_capability["coding"]["eval_count"] == "1"


def test_write_csv_writes_stable_columns(tmp_path: Path) -> None:
    output = tmp_path / "summary.csv"
    rows = [
        {
            "capability": "reasoning",
            "seed_count": "1",
            "reviewed_count": "1",
            "eval_count": "1",
            "reviewed_target": "2",
            "reviewed_deficit": "1",
        }
    ]

    summarize_capabilities.write_csv(output, rows)

    with output.open(newline="", encoding="utf-8") as input_file:
        assert list(csv.DictReader(input_file)) == rows


def test_provenance_rows_counts_capability_and_review_source(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed.jsonl"
    write_jsonl(
        reviewed,
        [
            {
                "id": "row-1",
                "capability": "coding",
                "review": {"source": "edited_candidate"},
            },
            {
                "id": "row-2",
                "capability": "coding",
                "review": {"source": "edited_candidate"},
            },
            {
                "id": "row-3",
                "capability": "reasoning",
                "review": {"source": "judge_accepted_candidate"},
            },
            {
                "id": "row-4",
                "capability": "tool_use",
                "review": {"notes": "old row"},
            },
        ],
    )

    assert summarize_capabilities.provenance_rows(reviewed) == [
        {"capability": "coding", "source": "edited_candidate", "count": "2"},
        {"capability": "reasoning", "source": "judge_accepted_candidate", "count": "1"},
        {"capability": "tool_use", "source": "historical_reviewed", "count": "1"},
    ]


def test_write_provenance_csv_writes_stable_columns(tmp_path: Path) -> None:
    output = tmp_path / "provenance.csv"
    rows = [{"capability": "coding", "source": "edited_candidate", "count": "2"}]

    summarize_capabilities.write_provenance_csv(output, rows)

    with output.open(newline="", encoding="utf-8") as input_file:
        assert list(csv.DictReader(input_file)) == rows
