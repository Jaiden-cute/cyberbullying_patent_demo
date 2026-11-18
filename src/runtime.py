"""Select the inference components for a reproducible offline or trained run."""

from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import dotenv_values

from .config import ModelSettings, Settings
from .event_builder import Event
from .llm_adapter import LLMUnavailable, QwenAdapter, RelationEvidence
from .masking_classifier import MaskingResult, evaluate_demo_masking
from .math_utils import cosine_similarity
from .model_runtime import ROOT, MaskHead, ModelUnavailable, SentenceEncoder
from .schemas import MASKING_LABELS, Message


@lru_cache(maxsize=2)
def _encoder(model_id, revision, sha256, path, threads) -> SentenceEncoder:
    # Only the immutable encoder identity affects the cached model.
    return SentenceEncoder(model_id, revision, sha256, path, threads)


@dataclass(frozen=True)
class Runtime:
    requested: str = "demo"
    strict: bool = False
    encoder: SentenceEncoder | None = None
    head: MaskHead | None = None
    qwen: QwenAdapter | None = None
    encoder_status: str = "demo_mode"
    head_status: str = "demo_mode"
    qwen_status: str = "demo_mode"
    qwen_requested: bool = False

    def similarity(self, left: str, right: str, norm_floor: float = 0) -> float | None:
        if self.encoder is None:
            return None
        return cosine_similarity(self.encoder.embed(left), self.encoder.embed(right), norm_floor=norm_floor)

    def relation(self, event: Event, reply: Message) -> RelationEvidence | None:
        if self.qwen is not None:
            return self.qwen.judge_relations(event, reply)
        if self.qwen_requested or self.strict:
            return RelationEvidence(0, 0, "llm_unavailable", "llm_unavailable", (self.qwen_status,))
        return None

    def masking(self, summary: str, text: str, weights: tuple[float, ...]) -> MaskingResult:
        if self.head is not None and self.encoder is not None:
            return self.head.predict(self.encoder, summary, text, weights)
        if self.strict:
            return MaskingResult(
                MASKING_LABELS, None, None, None, "unavailable", "trained_head_unavailable",
                {}, unavailable_reason=self.head_status,
            )
        return evaluate_demo_masking(text, weights)

    @property
    def model_sources(self) -> dict:
        result = {}
        if self.encoder is not None:
            result["sentence_bert"] = self.encoder.provenance
        if self.head is not None:
            result["trained_mask_head"] = self.head.provenance
        if self.qwen is not None:
            result["qwen_2_5_7b_instruct"] = {"model_id": self.qwen.served_id, "transport": "openai_compatible"}
        return result


def _load_qwen(config: ModelSettings) -> tuple[QwenAdapter | None, str, bool]:
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    endpoint = env.get("QWEN_BASE_URL")
    if not endpoint:
        return None, "qwen_endpoint_not_configured", False
    try:
        qwen = QwenAdapter(
            base_url=endpoint, served_id=env.get("QWEN_MODEL_ID") or config.llm_model_id,
            expected_model=config.llm_model_id, api_key=env.get("QWEN_API_KEY") or "",
            timeout=config.llm_timeout_seconds,
        )
        qwen.probe()
    except (LLMUnavailable, ValueError) as error:
        return None, str(error), True
    return qwen, "endpoint_available", True


def load_runtime(settings: Settings, *, requested: str, strict: bool = False, use_qwen: bool = True) -> Runtime:
    if requested not in ("demo", "trained"):
        raise ValueError("requested runtime must be demo or trained")
    if requested == "demo":
        return Runtime()
    defaults = settings.engineering_demo_default
    config = defaults.models
    encoder = head = None
    encoder_status = head_status = "unavailable"
    try:
        encoder = _encoder(config.embedding_model_id, config.embedding_revision, config.embedding_sha256,
                           str(ROOT / config.embedding_path), config.cpu_threads)
        encoder_status = "available"
        head = MaskHead.load(ROOT / config.mask_head_path, encoder)
        head_status = "available"
    except (ModelUnavailable, OSError, ValueError) as error:
        if encoder is None:
            encoder_status = str(error)
            head_status = "encoder_required"
        else:
            head_status = str(error)
    qwen, qwen_status, qwen_requested = _load_qwen(config) if use_qwen else (None, "disabled_for_run", False)
    return Runtime(
        requested, strict or defaults.strict_model_mode, encoder, head, qwen,
        encoder_status, head_status, qwen_status, qwen_requested,
    )
