"""Command line: synthesize a line of Thai, or inspect what a model ships.

    python -m wayu_tts "สวัสดีครับ วันนี้อากาศดี" --voice m_young_clear --out hello.wav
    python -m wayu_tts --list-voices
    python -m wayu_tts "ราคา 1,250 บาท" --phonemes
    cat article.txt | python -m wayu_tts --out article.wav
"""

from __future__ import annotations

import argparse
import sys

from .config import CONFIG_NAME, WayuTTSConfig
from .g2p import ThaiG2P
from .tts import ThaiTTS, resolve_model_dir
from .voices import available_voices

DEFAULT_MODEL = "wayu-ai/wayu-paxa-tts-edge"
DEFAULT_VOICE = "f_young_clear"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wayu-tts", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="Thai text to speak; omit to read stdin")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="local model directory or Hugging Face repo id")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--out", default="out.wav")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="duration scale on top of the voice's calibrated rate; "
                             ">1 is faster")
    parser.add_argument("--device", help="cuda, cpu, ... (default: cuda when available)")
    parser.add_argument("--seed", type=int,
                        help="make generation reproducible (otherwise the vocoder's noise "
                             "source is redrawn every call)")
    parser.add_argument("--list-voices", action="store_true",
                        help="print the model's voices and exit")
    parser.add_argument("--phonemes", action="store_true",
                        help="print what the model would be fed, and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = resolve_model_dir(args.model)

    # Both of these answer from the model directory alone -- no weights, no torch.
    if args.list_voices:
        print("\n".join(available_voices(model_dir)))
        return 0
    if args.phonemes:
        config = WayuTTSConfig.from_file(model_dir / CONFIG_NAME)
        print(ThaiG2P(config)(_read_text(args)))
        return 0

    tts = ThaiTTS.from_pretrained(model_dir, device=args.device)
    audio = tts(_read_text(args), voice=args.voice, speed=args.speed, seed=args.seed)
    path = tts.save(args.out, audio)
    print(f"{path}  ({len(audio) / tts.sample_rate:.2f}s, {args.voice})")
    return 0


def _read_text(args: argparse.Namespace) -> str:
    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        raise SystemExit("no text given")
    return text
