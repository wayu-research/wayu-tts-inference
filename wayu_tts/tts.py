"""The one thing most callers want: Thai text in, a waveform out.

    >>> tts = ThaiTTS.from_pretrained("wayu-ai/wayu-paxa-tts-edge")
    >>> audio = tts("สวัสดีครับ วันนี้อากาศดีมาก", voice="m_young_clear")
    >>> tts.save("hello.wav", audio)

Text longer than the model's context window is split at sentence boundaries and
the pieces are joined with a short silence, so there is no length limit to work
around at the call site.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import CONFIG_NAME, WEIGHTS_NAME, WayuTTSConfig
from .g2p import ThaiG2P
from .voices import Voice, available_voices, load_voice

if TYPE_CHECKING:
    import torch

#: Break here first; these carry a pause anyway, so the seam is inaudible.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
JOIN_SILENCE_SECONDS = 0.25


@dataclass
class ThaiTTS:
    """A loaded checkpoint, its frontend, and its voices."""

    model: torch.nn.Module
    g2p: ThaiG2P
    config: WayuTTSConfig
    model_dir: Path
    device: str = "cpu"
    _loaded: dict[str, Voice] = field(default_factory=dict, repr=False)

    @classmethod
    def from_pretrained(cls, source: str | Path, device: str | None = None,
                        **g2p_kwargs) -> ThaiTTS:
        """Load from a local directory or a Hugging Face repo id."""
        import torch
        from kokoro import KModel

        model_dir = resolve_model_dir(source)
        config = WayuTTSConfig.from_file(model_dir / CONFIG_NAME)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = KModel(config=str(model_dir / CONFIG_NAME),
                       model=str(model_dir / WEIGHTS_NAME)).to(device).eval()
        return cls(model=model, g2p=ThaiG2P(config, **g2p_kwargs),
                   config=config, model_dir=model_dir, device=device)

    @property
    def voices(self) -> list[str]:
        return available_voices(self.model_dir)

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def load_voice(self, voice: str | Voice) -> Voice:
        """A voice by name, read from disk once and kept."""
        if isinstance(voice, Voice):
            return voice.to(self.device)
        if voice not in self._loaded:
            self._loaded[voice] = load_voice(self.model_dir, voice, device=self.device)
        return self._loaded[voice]

    def __call__(self, text: str, voice: str | Voice, speed: float = 1.0,
                 seed: int | None = None) -> np.ndarray:
        """Synthesize `text`, returning float32 audio at :attr:`sample_rate`.

        `speed` multiplies the voice's calibrated rate (``config.json``'s
        ``voice_speeds``), so 1.0 is "this voice at its teacher's pace", not
        "whatever the predictor happens to emit".

        Two renders of the same text differ slightly: the vocoder excites its
        harmonic source with Gaussian noise, so the noise floor is redrawn every
        call (correlation ~0.99, no structural difference).  Pass `seed` when
        renders have to be reproducible -- scoring a system twice, or diffing two
        checkpoints on identical input.
        """
        if seed is not None:
            import torch

            torch.manual_seed(seed)
        speaker = self.load_voice(voice)
        speed = speed * self.config.voice_speed(speaker.name)
        clips = [self._synthesize(phonemes, speaker, speed) for phonemes in self.phonemize(text)]
        if not clips:
            return np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(JOIN_SILENCE_SECONDS * self.sample_rate), dtype=np.float32)
        return np.concatenate([c for clip in clips for c in (clip, silence)][:-1])

    def phonemize(self, text: str) -> list[str]:
        """The phoneme chunks this text will be synthesized as, in order."""
        budget = self.config.context_length - 2  # the model pads with two boundary tokens
        count = self.g2p.count_tokens
        return [chunk
                for sentence in _SENTENCE_END.split(text.strip()) if sentence.strip()
                for chunk in _fit(self.g2p(sentence), budget, count) if count(chunk)]

    def _synthesize(self, phonemes: str, voice: Voice, speed: float) -> np.ndarray:
        import torch

        style = voice.style(self.g2p.count_tokens(phonemes))
        with torch.no_grad():
            audio = self.model(phonemes, style, speed=speed)
        return audio.detach().cpu().numpy().astype(np.float32)

    def save(self, path: str | Path, audio: np.ndarray) -> Path:
        import soundfile as sf

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, self.sample_rate)
        return path


def resolve_model_dir(source: str | Path) -> Path:
    """A local directory as-is, anything else as a Hugging Face repo id."""
    path = Path(source)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(source)))


def _fit(phonemes: str, budget: int, count: Callable[[str], int]) -> Iterator[str]:
    """Pack space-separated phoneme words into chunks of at most `budget` tokens.

    A single word over budget is emitted alone and will be rejected by the model
    rather than silently truncated -- that is a frontend bug worth seeing.
    """
    chunk: list[str] = []
    size = 0
    for word in phonemes.split(" "):
        cost = count(word) + (1 if chunk else 0)  # the joining space is a token too
        if chunk and size + cost > budget:
            yield " ".join(chunk)
            chunk, size, cost = [], 0, count(word)
        chunk.append(word)
        size += cost
    if chunk:
        yield " ".join(chunk)
