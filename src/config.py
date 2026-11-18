"""Loading and validation of patent example values and adjustable engineering defaults."""

from math import fsum, isclose
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, model_validator

from .schemas import Finite, Identifier, InputModel, UnitValue

Positive = Annotated[Finite, Field(gt=0)]
Nonnegative = Annotated[Finite, Field(ge=0)]
FourWeights = tuple[UnitValue, UnitValue, UnitValue, UnitValue]


def validate_unit_weights(weights: tuple[float, ...], name: str) -> None:
    if any(not 0 <= weight <= 1 for weight in weights) or not isclose(
        fsum(weights), 1.0, abs_tol=1e-12, rel_tol=0
    ):
        raise ValueError(f"{name} must contain nonnegative weights summing to 1")


class SeverityWeights(InputModel):
    targeting: UnitValue
    aggression: UnitValue
    repetition: UnitValue
    harm: UnitValue

    @property
    def values(self) -> FourWeights:
        return self.targeting, self.aggression, self.repetition, self.harm

    @model_validator(mode="after")
    def validate_sum(self) -> Self:
        validate_unit_weights(self.values, "severity_weights")
        return self


class AssociationWeights(InputModel):
    b0: Finite
    b1: Finite
    b2: Finite
    b3: Finite
    b4: Finite
    b5: Finite
    b6: Finite

    @property
    def slopes(self) -> tuple[float, ...]:
        return self.b1, self.b2, self.b3, self.b4, self.b5, self.b6


class FusionWeights(InputModel):
    alpha: Nonnegative
    beta: Nonnegative
    gamma: Nonnegative

    @model_validator(mode="after")
    def validate_sum(self) -> Self:
        if self.alpha + self.beta + self.gamma <= 0:
            raise ValueError("alpha + beta + gamma must be positive")
        return self


class PatentExample(InputModel):
    attack_similarity_tau_s: UnitValue
    time_tau_minutes: Annotated[Finite, Field(ge=5, le=30)]
    association_tau_A: UnitValue
    masking_class_count: Literal[8]
    baseline_min_valid_observations: Annotated[int, Field(strict=True, ge=3)]
    fusion_weights: FusionWeights
    history_tau_p: Positive
    history_mask_tau_H: UnitValue
    min_events_for_cross_event: Annotated[int, Field(strict=True, ge=3)]
    risk_thresholds: tuple[UnitValue, UnitValue, UnitValue]

    @model_validator(mode="after")
    def ordered_risk_thresholds(self) -> Self:
        general, important, high = self.risk_thresholds
        if not general < important < high:
            raise ValueError("risk_thresholds must be strictly increasing")
        return self


class FeatureConfig(InputModel):
    modality: Literal["audio", "video"]
    unit: Identifier
    direction: Literal[-1, 1]
    weight: Nonnegative


class TrainingSettings(InputModel):
    seed: Annotated[int, Field(strict=True, ge=0)]
    epochs: Annotated[int, Field(strict=True, ge=1)]
    learning_rate: Positive
    weight_decay: Nonnegative
    evaluation_threshold: UnitValue


class ModelSettings(InputModel):
    embedding_model_id: Identifier
    embedding_revision: Identifier
    embedding_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    embedding_path: Identifier
    mask_head_path: Identifier
    llm_model_id: Literal["Qwen2.5-7B-Instruct"]
    llm_timeout_seconds: Positive
    cpu_threads: Annotated[int, Field(strict=True, ge=1)]
    training: TrainingSettings


class EngineeringDefaults(InputModel):
    first_event_window_minutes: Positive
    second_history_window_days: Positive
    event_topic_similarity_threshold: UnitValue
    candidate_window_minutes: Positive
    history_tau_p_unit: Literal["days"]
    repetition_basis: Literal["within_event", "similar_attacks"]
    similar_attack_window_days: Positive
    severity_weights: SeverityWeights
    association_weights: AssociationWeights
    demo_topic_continuity_threshold: UnitValue
    masking_weights: Annotated[tuple[UnitValue, ...], Field(min_length=8, max_length=8)]
    kappa_text: Annotated[Finite, Field(gt=0, lt=1)]
    history_n0: Positive
    risk_weights: FourWeights
    epsilon: Positive
    cosine_norm_floor: Nonnegative
    strict_model_mode: bool
    models: ModelSettings
    baseline_window_minutes: Positive
    baseline_std_floor: Positive
    nonverbal_features: dict[Identifier, FeatureConfig]
    safety_phrases: dict[Identifier, tuple[Identifier, ...]]

    @model_validator(mode="after")
    def validate_windows_and_weights(self) -> Self:
        if self.second_history_window_days * 24 * 60 <= self.first_event_window_minutes:
            raise ValueError("the second history window must exceed the first event window")
        validate_unit_weights(self.masking_weights, "masking_weights")
        validate_unit_weights(self.risk_weights, "risk_weights")
        return self


class Settings(InputModel):
    patent_example: PatentExample
    engineering_demo_default: EngineeringDefaults


def load_settings(path: Path) -> Settings:
    return Settings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
