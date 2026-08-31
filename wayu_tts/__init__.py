"""Wayu-Paxa-TTS-Edge: a fixed-voice Thai text-to-speech model and its frontend.

    from wayu_tts import ThaiTTS

    tts = ThaiTTS.from_pretrained("wayu-ai/wayu-paxa-tts-edge")
    tts.save("hello.wav", tts("สวัสดีครับ", voice="m_young_clear"))

The frontend is usable on its own -- it is an ordinary text -> IPA function::

    from wayu_tts import ThaiG2P, WayuTTSConfig

    g2p = ThaiG2P(WayuTTSConfig.from_file("model/config.json"))
    g2p("เรียก Grab ไปทำงาน")
"""

from .config import WayuTTSConfig
from .g2p import MisakiEnglish, Phonemizer, ThaiG2P, TltkThai, Trace, encode
from .normalize import normalize
from .tts import ThaiTTS, resolve_model_dir
from .voices import Voice, available_voices, load_voice

__version__ = "0.1.0"

__all__ = [
    "MisakiEnglish",
    "Phonemizer",
    "ThaiG2P",
    "WayuTTSConfig",
    "ThaiTTS",
    "TltkThai",
    "Trace",
    "Voice",
    "available_voices",
    "encode",
    "load_voice",
    "resolve_model_dir",
    "normalize",
]
