from pathlib import Path

from pydantic import TypeAdapter
import pytest

from src.config import load_settings
from src.schemas import Message, ReplyFeatureTests

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    return load_settings(ROOT / "config.yaml")


@pytest.fixture
def messages():
    path = ROOT / "data" / "synthetic_messages.json"
    return TypeAdapter(list[Message]).validate_json(path.read_bytes())


@pytest.fixture
def features():
    path = ROOT / "data" / "synthetic_reference_features.json"
    return TypeAdapter(list[ReplyFeatureTests]).validate_json(path.read_bytes())
