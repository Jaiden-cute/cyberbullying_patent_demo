"""S4 response-level fusion and representative reply selection."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .config import FusionWeights
from .schemas import Message


@dataclass(frozen=True)
class HResult:
    H: float
    branch: Literal["multimodal", "text"]


def event_masking_score(A: float, M: float, N: float | None, weights: FusionWeights, kappa: float) -> HResult:
    if not all(0 <= value <= 1 for value in (A, M)) or (N is not None and not 0 <= N <= 1):
        raise ValueError("A, M and an available N must lie in [0, 1]")
    if not 0 < kappa < 1:
        raise ValueError("kappa must be strictly between 0 and 1")
    if N is None:
        return HResult(kappa * A * M, "text")
    alpha, beta, gamma = weights.alpha, weights.beta, weights.gamma
    H = A * (alpha * M + beta * N + gamma * M * N) / (alpha + beta + gamma)
    return HResult(H, "multimodal")


def representative_reply(results: Iterable[tuple[Message, HResult]]) -> Message | None:
    best = min(results, key=lambda item: (-item[1].H, item[0].utc_time, item[0].key), default=None)
    return best[0] if best is not None else None
