import json
from pathlib import Path

import pytest

from llm.leverage.export_sft_dataset import export_dataset, export_rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def candidate_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "sft_test_001",
        "source_prompt_id": "lt_seed_test",
        "category": "resource_judgment",
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "Explain when to use hosted inference."},
            {"role": "assistant", "content": "Use hosted inference before spending GPU time."},
        ],
        "review": {
            "status": "accepted_candidate",
            "author": "tester",
            "notes": "Good minimal row.",
        },
    }
    row.update(overrides)
    return row


def test_export_rows_keeps_only_id_and_messages() -> None:
    row = candidate_row()

    exported = export_rows([(1, row)], include_id=True)

    assert exported == [{"id": "sft_test_001", "messages": row["messages"]}]


def test_export_rows_can_omit_ids() -> None:
    row = candidate_row()

    exported = export_rows([(1, row)], include_id=False)

    assert exported == [{"messages": row["messages"]}]


def test_export_dataset_writes_training_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "datasets" / "candidates.jsonl"
    output_path = tmp_path / "data" / "sft" / "train.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(input_path, [candidate_row()])

    count = export_dataset(input_path, output_path, eval_dir=eval_dir, include_id=True, overwrite=False)

    assert count == 1
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        json.dumps(
            {
                "id": "sft_test_001",
                "messages": candidate_row()["messages"],
            },
            ensure_ascii=False,
        )
    ]


def test_export_dataset_rejects_invalid_candidates(tmp_path: Path) -> None:
    input_path = tmp_path / "datasets" / "candidates.jsonl"
    output_path = tmp_path / "data" / "sft" / "train.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(input_path, [candidate_row(review={"status": "raw", "author": "tester", "notes": "No."})])

    with pytest.raises(ValueError, match="review.status"):
        export_dataset(input_path, output_path, eval_dir=eval_dir, include_id=True, overwrite=False)

    assert not output_path.exists()


def test_export_dataset_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "datasets" / "candidates.jsonl"
    output_path = tmp_path / "data" / "sft" / "train.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(input_path, [candidate_row()])
    output_path.parent.mkdir(parents=True)
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_dataset(input_path, output_path, eval_dir=eval_dir, include_id=True, overwrite=False)
