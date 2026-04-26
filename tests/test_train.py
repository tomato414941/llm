import csv

from llm.train import append_metrics_row


def test_append_metrics_row_writes_header_and_values(tmp_path) -> None:
    path = tmp_path / "metrics.csv"

    append_metrics_row(path, 0, {"train": 1.0, "val": 2.0})

    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    assert rows == [
        {
            "step": "0",
            "train_loss": "1.0",
            "val_loss": "2.0",
            "train_ppl": "2.718281828459045",
            "val_ppl": "7.38905609893065",
        }
    ]


def test_append_metrics_row_appends_without_rewriting_header(tmp_path) -> None:
    path = tmp_path / "metrics.csv"

    append_metrics_row(path, 0, {"train": 1.0, "val": 2.0})
    append_metrics_row(path, 1, {"train": 0.5, "val": 1.5})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "step,train_loss,val_loss,train_ppl,val_ppl"
    assert len(lines) == 3
