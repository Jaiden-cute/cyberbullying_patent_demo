"""Download the pinned official Sentence-BERT snapshot used by the classifier."""

import argparse
from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen

from src.config import load_settings
from src.model_runtime import ROOT, file_sha256

FILES = (
    "modules.json", "config.json", "config_sentence_transformers.json", "sentence_bert_config.json",
    "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "1_Pooling/config.json", "model.safetensors",
)


def fetch(config_path: Path) -> Path:
    settings = load_settings(config_path).engineering_demo_default.models
    model_dir = ROOT / settings.embedding_path
    base = f"https://huggingface.co/{settings.embedding_model_id}/resolve/{settings.embedding_revision}"
    for name in FILES:
        path = model_dir / name
        is_weights = name == "model.safetensors"
        if path.is_file() and (not is_weights or file_sha256(path) == settings.embedding_sha256):
            continue
        pending = path.with_suffix(path.suffix + ".part")
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(f"{base}/{name}?download=true", headers={"User-Agent": "patent-prototype-sbert/1"})
        with urlopen(request, timeout=120) as source, pending.open("wb") as target:
            copyfileobj(source, target, length=1024 * 1024)
        if is_weights and file_sha256(pending) != settings.embedding_sha256:
            raise ValueError("downloaded Sentence-BERT weights failed SHA-256 verification")
        pending.replace(path)
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    try:
        print(fetch(args.config))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Sentence-BERT 下载失败：{error}\n")


if __name__ == "__main__":
    main()
