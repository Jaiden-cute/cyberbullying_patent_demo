"""Fit candidate b0–b6 from a supplied CSV of six features and human link labels."""

import argparse
import csv
import json
from pathlib import Path

from src.config import load_settings
from src.model_runtime import ROOT, file_sha256
from src.reply_linker import association_logit

FEATURES = ("L", "Q", "S", "T", "D", "G")


def calibrate(path: Path, config_path: Path, epochs: int, learning_rate: float) -> dict:
    import torch

    if epochs < 1 or not 0 < learning_rate <= 1:
        raise ValueError("epochs must be positive and learning rate must be within (0, 1]")
    config = load_settings(config_path).engineering_demo_default
    priors = config.association_weights
    observations = []
    labels = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != (*FEATURES, "label"):
            raise ValueError("CSV must use fields L,Q,S,T,D,G,label in that order")
        for record in reader:
            values = {key: float(record[key]) for key in FEATURES}
            association_logit(priors, **values)
            if record["label"] not in ("0", "1"):
                raise ValueError(f"invalid human association label on CSV line {reader.line_num}")
            observations.append(tuple(values.values()))
            labels.append(float(record["label"]))
    if set(labels) != {0.0, 1.0}:
        raise ValueError("association calibration requires both linked and unlinked examples")
    X = torch.tensor(observations, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    layer = torch.nn.Linear(len(FEATURES), 1, dtype=torch.float64)
    with torch.no_grad():
        layer.weight[0].copy_(torch.tensor(priors.slopes, dtype=torch.float64))
        layer.bias[0].fill_(priors.b0)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(layer.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        criterion = loss(layer(X).flatten(), y)
        criterion.backward()
        optimizer.step()
    values = [layer.bias.detach().item(), *layer.weight.detach()[0].tolist()]
    return {
        "source": "offline_association_calibration",
        "input_sha256": file_sha256(path), "sample_count": len(labels),
        "epochs": epochs, "learning_rate": learning_rate,
        "association_weights": {f"b{index}": value for index, value in enumerate(values)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.labels, args.config, args.epochs, args.learning_rate), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
