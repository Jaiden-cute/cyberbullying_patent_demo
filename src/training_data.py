"""Validation and leakage checks for the curated synthetic masking experiment."""

from collections import Counter
from dataclasses import dataclass
import csv
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path
from unicodedata import normalize

from .schemas import MASKING_LABELS

TRAINING_SOURCE = "synthetic_training_demo"
CSV_FIELDS = ("sample_id", "scenario_group", "data_source", "event_summary", "reply", *MASKING_LABELS)
NEAR_DUPLICATE_THRESHOLD = 0.86


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    scenario_group: str
    event_summary: str
    reply: str
    labels: tuple[int, ...]

    def to_csv(self) -> tuple:
        return self.sample_id, self.scenario_group, TRAINING_SOURCE, self.event_summary, self.reply, *self.labels


def _normalized(text: str) -> str:
    return "".join(char.casefold() for char in normalize("NFKC", text) if char.isalnum())


def read_examples(path: Path) -> list[TrainingExample]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"{path.name} has an invalid field order")
        examples = []
        for row in reader:
            if row.get(None) or not all(row[field] for field in CSV_FIELDS):
                raise ValueError(f"{path.name}:{reader.line_num} has missing or excess fields")
            if row["data_source"] != TRAINING_SOURCE or any(row[label] not in ("0", "1") for label in MASKING_LABELS):
                raise ValueError(f"{path.name}:{reader.line_num} requires the declared source and eight 0/1 labels")
            labels = tuple(int(row[label]) for label in MASKING_LABELS)
            examples.append(TrainingExample(row["sample_id"], row["scenario_group"], row["event_summary"], row["reply"], labels))
    return examples


def split_statistics(examples: list[TrainingExample]) -> dict:
    return {
        "rows": len(examples),
        "scenario_groups": len({item.scenario_group for item in examples}),
        "negative_rows": sum(sum(item.labels) == 0 for item in examples),
        "multi_label_rows": sum(sum(item.labels) > 1 for item in examples),
        "positives": dict(zip(MASKING_LABELS, map(sum, zip(*(item.labels for item in examples))), strict=True)),
    }


def validate_splits(train: list[TrainingExample], test: list[TrainingExample]) -> None:
    combined = train + test
    counts = Counter(item.sample_id for item in combined)
    if not train or not test or any(count != 1 for count in counts.values()):
        raise ValueError("both splits are required and sample IDs must be unique")
    for get_value in (
        lambda item: item.scenario_group,
        lambda item: _normalized(item.event_summary),
        lambda item: _normalized(item.reply),
    ):
        if {get_value(item) for item in train} & {get_value(item) for item in test}:
            raise ValueError("train/test leakage: repeated scenario, summary or reply")
    left_replies = [_normalized(item.reply) for item in train]
    right_replies = [_normalized(item.reply) for item in test]
    for left, right in product(left_replies, right_replies):
        if min(len(left), len(right)) >= 8 and SequenceMatcher(None, left, right, autojunk=False).ratio() >= NEAR_DUPLICATE_THRESHOLD:
            raise ValueError("train/test leakage: near-duplicate reply")
    for split in (train, test):
        stats = split_statistics(split)
        if not stats["negative_rows"] or not stats["multi_label_rows"] or not all(stats["positives"].values()):
            raise ValueError("each split needs all labels, negative examples and multi-label examples")
