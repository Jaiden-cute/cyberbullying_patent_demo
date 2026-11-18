"""Validate local media references and define the real extractor integration point."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .event_builder import Event
from .schemas import Message, ReplyMediaFeatures

AUDIO_SUFFIXES = {".wav", ".mp3"}
VIDEO_SUFFIXES = {".mp4"}


@dataclass(frozen=True)
class LocalMedia:
    audio: Path | None
    video: Path | None


class MediaFeatureExtractor(Protocol):
    def extract_reply(
        self, event: Event, reply: Message, current: LocalMedia,
        references: Sequence[tuple[Message, LocalMedia]],
    ) -> ReplyMediaFeatures | None:
        """Return synchronized observations with provenance from the validated paths."""
        ...


def resolve_media(message: Message, permitted_dir: Path | None) -> LocalMedia:
    if message.audio_path is None and message.video_path is None:
        return LocalMedia(None, None)
    if permitted_dir is None:
        raise ValueError("allowed_media_directory_not_configured")
    if message.consent_status != "authorized":
        raise ValueError("media_requires_authorized_consent")
    if message.media_start is None or message.media_end is None:
        raise ValueError("media_requires_synchronization_window")
    root = permitted_dir.resolve()
    paths = []
    for name, suffixes in ((message.audio_path, AUDIO_SUFFIXES), (message.video_path, VIDEO_SUFFIXES)):
        if name is None:
            paths.append(None)
            continue
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("media_outside_allowed_directory")
        if candidate.suffix.lower() not in suffixes:
            raise ValueError("unsupported_media_format")
        if not candidate.is_file():
            raise ValueError("media_file_missing")
        paths.append(candidate)
    return LocalMedia(*paths)
