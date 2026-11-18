"""Train and verify an eight-output linear head on the declared synthetic split."""

import argparse
import json
from pathlib import Path

from src.config import load_settings
from src.model_runtime import ROOT, HeadPayload, MaskHead, SentenceEncoder, artifact_version, file_sha256
from src.schemas import MASKING_LABELS
from src.training_data import TRAINING_SOURCE, read_examples, split_statistics, validate_splits


def train(config_path: Path, train_path: Path, test_path: Path, artifact_path: Path, report_path: Path) -> dict:
    import torch
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    train_rows, test_rows = read_examples(train_path), read_examples(test_path)
    validate_splits(train_rows, test_rows)
    config = load_settings(config_path).engineering_demo_default.models
    params = config.training
    torch.manual_seed(params.seed)
    encoder = SentenceEncoder.from_settings(config)

    def tensors(rows):
        summaries = list(dict.fromkeys(row.event_summary for row in rows))
        summary_vectors = dict(zip(summaries, encoder.encode_batch(summaries), strict=True))
        event = torch.tensor([summary_vectors[row.event_summary] for row in rows], dtype=torch.float32)
        reply = torch.tensor(encoder.encode_batch([row.reply for row in rows]), dtype=torch.float32)
        labels = torch.tensor([row.labels for row in rows], dtype=torch.float32)
        return event, reply, labels

    event, reply, labels = tensors(train_rows)
    joint = torch.cat((event, reply, torch.abs(event - reply)), dim=1)
    head = torch.nn.Linear(3 * encoder.dimension, len(MASKING_LABELS))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    with torch.no_grad():
        initial_loss = loss_fn(head(joint), labels).item()
    for _ in range(params.epochs):
        optimizer.zero_grad()
        loss = loss_fn(head(joint), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = loss_fn(head(joint), labels).item()
    if not final_loss < initial_loss:
        raise ValueError("training failed to reduce the fixed training-set loss")

    payload = {
        "artifact_format": "mask-head-linear-v1",
        "training_data_source": TRAINING_SOURCE,
        "label_order": list(MASKING_LABELS),
        "embedding_model_id": config.embedding_model_id,
        "embedding_revision": config.embedding_revision,
        "embedding_sha256": config.embedding_sha256,
        "embedding_dimension": encoder.dimension,
        "joint_representation": "concat(e,r,abs(e-r))",
        "training_csv_sha256": file_sha256(train_path),
        "training_rows": len(train_rows),
        "training_parameters": {**params.model_dump(), "loss": "BCEWithLogitsLoss", "optimizer": "AdamW"},
        "W": head.weight.detach().tolist(),
        "b": head.bias.detach().tolist(),
    }
    payload["model_version"] = artifact_version(payload)
    validated = HeadPayload.model_validate(payload)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    pending = artifact_path.with_suffix(".tmp")
    pending.write_text(validated.model_dump_json(indent=2), encoding="utf-8")
    loaded = MaskHead.load(pending, encoder)

    test_events, test_replies, test_labels = tensors(test_rows)
    probabilities = [
        loaded.predict_from_embeddings(left.tolist(), right.tolist())
        for left, right in zip(test_events, test_replies, strict=True)
    ]
    with torch.no_grad():
        test_joint = torch.cat((test_events, test_replies, torch.abs(test_events - test_replies)), dim=1)
        original = torch.sigmoid(head(test_joint))
    max_reload_delta = float(torch.max(torch.abs(original.double() - torch.tensor(probabilities, dtype=torch.float64))))
    if max_reload_delta > 1e-5:
        raise ValueError(f"saved head does not reproduce training probabilities: {max_reload_delta}")
    pending.replace(artifact_path)

    predicted = [[int(p >= params.evaluation_threshold) for p in row] for row in probabilities]
    truth = test_labels.int().tolist()
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, average=None, zero_division=0)
    report = {
        **loaded.provenance,
        "scope": "仅用于合成数据原型流程验证，未使用真实青少年样本。",
        "train": split_statistics(train_rows),
        "holdout": split_statistics(test_rows),
        "holdout_csv_sha256": file_sha256(test_path),
        "initial_training_bce": initial_loss,
        "final_training_bce": final_loss,
        "max_reloaded_probability_delta": max_reload_delta,
        "synthetic_holdout_micro_f1": f1_score(truth, predicted, average="micro", zero_division=0),
        "synthetic_holdout_macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        "per_label": {
            label: {"precision": float(p), "recall": float(r), "f1": float(score), "support": int(n)}
            for label, p, r, score, n in zip(MASKING_LABELS, precision, recall, f1, support, strict=True)
        },
        "prediction_cases": [
            {"sample_id": row.sample_id, "group": row.scenario_group, "labels": row.labels,
             "p": p, "predicted_labels": decisions}
            for row, p, decisions in zip(test_rows, probabilities, predicted, strict=True)
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--train", type=Path, default=ROOT / "data/synthetic_masking_train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/synthetic_masking_test.csv")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/mask_head_training.json")
    args = parser.parse_args()
    artifact = args.artifact or ROOT / load_settings(args.config).engineering_demo_default.models.mask_head_path
    report = train(args.config, args.train, args.test, artifact, args.report)
    print(json.dumps({key: val for key, val in report.items() if key != "prediction_cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
