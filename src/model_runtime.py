"""Offline Sentence-BERT and versioned eight-output linear-head inference."""

from collections.abc import Sequence
from functools import lru_cache
from hashlib import file_digest, sha256
import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .config import ModelSettings
from .masking_classifier import MaskingResult, joint_representation, linear_probabilities, masking_score
from .schemas import Finite, Identifier, InputModel, MASKING_LABELS
from .training_data import TRAINING_SOURCE

ROOT = Path(__file__).resolve().parents[1]


class ModelUnavailable(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


class SentenceEncoder:
    def __init__(self, model_id: str, revision: str, sha256: str, model_path: str, threads: int):
        self.model_id = model_id
        self.revision = revision
        self.weights_sha256 = sha256
        path = Path(model_path)
        weight_path = path / "model.safetensors"
        if not weight_path.is_file():
            raise ModelUnavailable(f"缺少 Sentence-BERT 权重 {weight_path}")
        if file_sha256(weight_path) != self.weights_sha256:
            raise ModelUnavailable("Sentence-BERT 权重 SHA-256 与配置不符")
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(threads)
            self.model = SentenceTransformer(
                str(path), local_files_only=True, device="cpu", trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            if not self.dimension or self.dimension < 1:
                raise ValueError("encoder reported no usable dimension")
        except (ImportError, RuntimeError, ValueError, OSError) as error:
            raise ModelUnavailable(f"Sentence-BERT 加载失败 ({type(error).__name__}): {error}") from error

    @classmethod
    def from_settings(cls, settings: ModelSettings) -> "SentenceEncoder":
        return cls(settings.embedding_model_id, settings.embedding_revision, settings.embedding_sha256,
                   str(ROOT / settings.embedding_path), settings.cpu_threads)

    @property
    def provenance(self) -> dict:
        return {"model_id": self.model_id, "revision": self.revision, "weights_sha256": self.weights_sha256,
                "embedding_dimension": self.dimension}

    def encode_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        values = self.model.encode(
            list(texts), batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False
        )
        return [tuple(map(float, vector)) for vector in values]

    @lru_cache(maxsize=512)
    def embed(self, text: str) -> tuple[float, ...]:
        return self.encode_batch((text,))[0]


class HeadPayload(InputModel):
    artifact_format: Literal["mask-head-linear-v1"]
    model_version: Identifier
    training_data_source: Literal["synthetic_training_demo"]
    label_order: tuple[str, ...]
    embedding_model_id: Identifier
    embedding_revision: Identifier
    embedding_sha256: Identifier
    embedding_dimension: Annotated[int, Field(strict=True, gt=0)]
    joint_representation: Literal["concat(e,r,abs(e-r))"]
    training_csv_sha256: Identifier
    training_rows: Annotated[int, Field(strict=True, gt=0)]
    training_parameters: dict
    W: tuple[tuple[Finite, ...], ...]
    b: tuple[Finite, ...]

    @model_validator(mode="after")
    def validate_shape_and_version(self) -> Self:
        if self.label_order != MASKING_LABELS:
            raise ValueError("masking label order differs from the implementation")
        if len(self.W) != len(MASKING_LABELS) or len(self.b) != len(MASKING_LABELS):
            raise ValueError("the classifier must contain eight outputs")
        if any(len(row) != 3 * self.embedding_dimension for row in self.W):
            raise ValueError("classifier head width must be 3 × encoder dimension")
        if self.model_version != artifact_version(self.model_dump(mode="json", exclude={"model_version"})):
            raise ValueError("classifier parameter fingerprint does not match model_version")
        return self


def artifact_version(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return f"{TRAINING_SOURCE}-{sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


class MaskHead:
    def __init__(self, payload: HeadPayload):
        self.payload = payload

    @classmethod
    def load(cls, path: Path, encoder: SentenceEncoder) -> "MaskHead":
        payload = HeadPayload.model_validate_json(path.read_bytes())
        expected = (encoder.model_id, encoder.revision, encoder.weights_sha256, encoder.dimension)
        metadata = (payload.embedding_model_id, payload.embedding_revision,
                    payload.embedding_sha256, payload.embedding_dimension)
        if expected != metadata:
            raise ValueError("classifier and Sentence-BERT identity, revision, hash or dimension differ")
        return cls(payload)

    def predict_from_embeddings(self, event: Sequence[float], reply: Sequence[float]) -> tuple[float, ...]:
        joint = joint_representation(event, reply)
        return linear_probabilities(joint, self.payload.W, self.payload.b)

    def predict(
        self, encoder: SentenceEncoder, summary: str, reply: str, weights: tuple[float, ...]
    ) -> MaskingResult:
        probabilities = self.predict_from_embeddings(encoder.embed(summary), encoder.embed(reply))
        return MaskingResult(
            MASKING_LABELS, probabilities, None, masking_score(probabilities, weights),
            "trained_linear_head", "available", {}, self.payload.model_version, self.payload.training_data_source,
        )

    @property
    def provenance(self) -> dict:
        return self.payload.model_dump(mode="json", exclude={"W", "b"})
