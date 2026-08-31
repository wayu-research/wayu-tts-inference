"""Speaker style vectors, bucketed by sentence length.

The model conditions on a 256-d style vector, split down the middle: the first half
steers the decoder's timbre, the second half steers the prosody predictor -- and
therefore the *duration* of every token.  That second half is why a voice cannot
be one fixed vector.  A style taken from a four-second reference asks the
predictor to speak every sentence at a four-second sentence's rate, which reads
about 1.25x too fast once the sentence is long.

So a voice ships as one vector per token count, and synthesis looks up the
bucket matching the sentence it is about to say -- the same shape the upstream
voice packs use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import VOICES_DIR

if TYPE_CHECKING:
    import torch

VOICE_SUFFIX = ".pt"


@dataclass(frozen=True)
class Voice:
    """One speaker: `styles[n]` is the style vector for an n+1 token sentence."""

    name: str
    styles: torch.Tensor  # (buckets, 1, 256)

    def style(self, token_count: int) -> torch.Tensor:
        """The style vector for a sentence of `token_count` tokens.

        Every bucket is filled at export time, buckets past the longest sentence
        the speaker was recorded saying by reusing the longest style available.
        That is extrapolation, and it is where very long sentences start to drift.
        """
        index = min(max(token_count - 1, 0), len(self.styles) - 1)
        return self.styles[index]

    def to(self, device) -> Voice:
        return Voice(self.name, self.styles.to(device))


def voice_path(model_dir: str | Path, name: str) -> Path:
    return Path(model_dir) / VOICES_DIR / f"{name}{VOICE_SUFFIX}"


def load_voice(model_dir: str | Path, name: str, device="cpu") -> Voice:
    import torch

    path = voice_path(model_dir, name)
    if not path.exists():
        available = ", ".join(available_voices(model_dir)) or "none"
        raise FileNotFoundError(f"no voice {name!r} in {path.parent} (available: {available})")
    return Voice(name, torch.load(path, map_location=device, weights_only=True))


def available_voices(model_dir: str | Path) -> list[str]:
    directory = Path(model_dir) / VOICES_DIR
    return sorted(p.stem for p in directory.glob(f"*{VOICE_SUFFIX}")) if directory.is_dir() else []
