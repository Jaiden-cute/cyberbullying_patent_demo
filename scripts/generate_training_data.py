"""Emit independent scenario splits as CSV from explicit synthetic annotations."""

import argparse
import csv
import json
from pathlib import Path

from src.event_builder import format_event_summary
from src.schemas import MASKING_LABELS
from src.training_data import CSV_FIELDS, TRAINING_SOURCE, TrainingExample, split_statistics, validate_splits

ROOT = Path(__file__).resolve().parents[1]


def generate(source: Path, output: Path) -> dict:
    document = json.loads(source.read_text(encoding="utf-8"))
    if document["source"] != TRAINING_SOURCE:
        raise ValueError("the annotation source must declare synthetic_training_demo")
    splits = {"train": [], "test": []}
    for case in document["cases"]:
        summary = format_event_summary(
            case["event_time"], tuple(case["initiators"]), case["victim_id"], tuple(case["types"]), case["content"]
        )
        for index, row in enumerate(case["rows"], 1):
            unknown = set(row["labels"]) - set(MASKING_LABELS)
            if unknown:
                raise ValueError(f"unknown masking labels: {unknown}")
            labels = tuple(int(label in row["labels"]) for label in MASKING_LABELS)
            family = case["family"]
            splits[case["split"]].append(TrainingExample(f"{family}-{index:02}", family, summary, row["reply"], labels))
    validate_splits(splits["train"], splits["test"])
    output.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        with (output / f"synthetic_masking_{split}.csv").open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(CSV_FIELDS)
            writer.writerows(item.to_csv() for item in items)
    return {key: split_statistics(value) for key, value in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/synthetic_annotation_source.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(json.dumps(generate(args.source, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
