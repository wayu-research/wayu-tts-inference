"""The contract between a released checkpoint and the frontend that feeds it.

A released checkpoint is a set of weights over a *fixed* token inventory: the
phoneme string handed to the model has to use the exact symbols the weights were
trained on, or the audio is wrong in ways nothing downstream can repair.  So the
inventory travels with the weights, in ``config.json``, and never lives in this
package:

* ``vocab`` / ``n_token`` / the architecture blocks -- read straight by
  the upstream model class, so a released directory loads unmodified.
* the ``thai`` block -- what :mod:`wayu_tts.g2p` needs to land inside that
  vocab: how a tone is written, which IPA symbols to fold, the voices shipped
  alongside (with their calibrated speaking rates), and the sample rate the
  decoder produces.

Nothing is defaulted.  A checkpoint that changes its tone inventory ships a new
``config.json`` and this code keeps working.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.pth"
VOICES_DIR = "voices"


@dataclass(frozen=True)
class WayuTTSConfig:
    """Everything ``config.json`` says, parsed once."""

    architecture: Mapping[str, Any]
    """The architecture block, passed through to the model class unchanged."""

    vocab: Mapping[str, int]
    """Phoneme symbol -> embedding id."""

    tone_tokens: Mapping[str, str]
    """tltk tone digit -> the vocab symbol that carries it."""

    ipa_fixups: Mapping[str, str]
    """IPA symbols tltk spells differently from the vocab, folded on the way in."""

    sample_rate: int
    voices: tuple[str, ...]

    voice_speeds: Mapping[str, float] = field(default_factory=dict)
    """Per-voice duration scale that lands the voice on its teacher's speaking rate.

    Measured at export time against the teacher audio the voice was distilled from;
    a voice not listed serves at 1.0.  The user-facing ``speed`` multiplies this.
    """

    def voice_speed(self, voice: str) -> float:
        return float(self.voice_speeds.get(voice, 1.0))

    @property
    def context_length(self) -> int:
        """Maximum token count per forward pass, boundary tokens included."""
        return int(self.architecture["plbert"]["max_position_embeddings"])

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> WayuTTSConfig:
        try:
            thai = raw["thai"]
            return cls(
                architecture=raw,
                vocab=raw["vocab"],
                tone_tokens=thai["tone_tokens"],
                ipa_fixups=thai["ipa_fixups"],
                sample_rate=int(thai["sample_rate"]),
                voices=tuple(thai["voices"]),
                voice_speeds=thai.get("voice_speeds", {}),
            )
        except KeyError as exc:  # a stock upstream config has no `thai` block
            raise ValueError(f"config.json is missing {exc}; "
                             "is this a Wayu-Paxa-TTS-Edge model directory?") from exc

    @classmethod
    def from_file(cls, path: str | Path) -> WayuTTSConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
