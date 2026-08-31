"""Thai (and embedded English) text -> one phoneme string.

The model uses a single IPA token inventory shared across every language it speaks,
so a bilingual sentence does not need a bilingual model -- only a frontend that
routes each span to the right G2P and lands both inside that one inventory::

    เรียก Grab ไปทำงาน
      ├ เรียก      -> tltk      -> rîːak
      ├ Grab       -> misaki-en -> ɡɹˈæb
      └ ไปทำงาน    -> tltk      -> pai tʰam ŋaːn

Thai tones are the one place the two disagree.  tltk writes a tone as a digit
suffix (``diː1``); the vocab has no digits, so each digit becomes the tone symbol
the checkpoint was trained on -- a mapping that ships in ``config.json``, not
here, because it is a property of the weights.

Both branches are ordinary callables, so either can be replaced::

    ThaiG2P(config, english=my_english_g2p)   # or english=None to drop Latin
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .config import WayuTTSConfig
from .normalize import normalize

#: Thai run | Latin phrase | whitespace | any single other character.
#:
#: The Latin branch matches whole *phrases* (interior spaces and hyphens) on
#: purpose: an English G2P weights function words and homographs by their
#: neighbours ("read", "live", "to"), and feeding it one word at a time throws
#: that away.  It still has to end on a letter, so a run cannot swallow the
#: space in front of the next Thai word.
_SPAN = re.compile(r"([฀-๿]+)|([A-Za-z][A-Za-z'\- ]*[A-Za-z]|[A-Za-z])|(\s+)|(.)")

_TLTK_TAG = re.compile(r"<[^>]*>")

#: Distinguishes "I did not choose" from an explicit `english=None`, which means
#: "this deployment does not speak English -- report Latin spans as lost".
_DEFAULT = object()
_TLTK_SYLLABLE_BREAK = ".+"


class Phonemizer(Protocol):
    """Anything that turns one span of text into the model's phoneme symbols."""

    def __call__(self, text: str) -> str: ...


@dataclass
class Trace:
    """A phonemization with its losses kept, for auditing a corpus."""

    phonemes: str
    dropped_words: list[tuple[str, str]] = field(default_factory=list)
    """(language, surface) spans that produced no phonemes at all."""

    dropped_symbols: set[str] = field(default_factory=set)
    """Characters normalization discarded as unspeakable."""

    @property
    def is_clean(self) -> bool:
        return not self.dropped_words and not self.dropped_symbols


class MisakiEnglish:
    """The upstream English G2P, so English is phonemized the way the model learnt it.

    Imported lazily: a Thai-only deployment never pays for it.  When espeak-ng is
    installed misaki falls back to it for words no dictionary lists (brand names,
    surnames), which is most of the English a Thai corpus actually contains.
    """

    def __init__(self, british: bool = False) -> None:
        self._british = british
        self._g2p: Callable | None = None

    def _load(self) -> Callable:
        from misaki import en

        try:
            from misaki.espeak import EspeakFallback

            fallback = EspeakFallback(british=self._british)
        except Exception:  # espeak-ng absent: unlisted words phonemize to nothing
            fallback = None
        return en.G2P(trf=False, british=self._british, fallback=fallback)

    def __call__(self, text: str) -> str:
        if self._g2p is None:
            self._g2p = self._load()
        phonemes, _ = self._g2p(text)
        return phonemes or ""


class TltkThai:
    """Thai G2P through `tltk`, with a word-by-word retry.

    tltk does not raise on a spelling it cannot segment -- it returns an empty
    string, or a partial conversion ending in a dangling syllable break.  Both
    mean a word was lost, so both are detected and retried one word at a time,
    where a single bad token can no longer take its neighbours down with it.
    """

    def __call__(self, text: str) -> str:
        ipa = self._convert(text)
        if _is_complete(ipa):
            return ipa
        words = _tokenize(text)
        if len(words) > 1:
            recovered = [ipa for word in words if _is_complete(ipa := self._convert(word))]
            if recovered:
                return " ".join(recovered)
        return ipa if _is_complete(ipa) else ""

    @staticmethod
    def _convert(text: str) -> str:
        import tltk.nlp

        try:
            return tltk.nlp.th2ipa(text)
        except Exception:
            return ""


def _is_complete(ipa: str) -> bool:
    """tltk marks a truncated conversion with a trailing syllable break."""
    core = _TLTK_TAG.sub("", ipa).strip()
    return bool(core) and not core.endswith(".")


def _tokenize(text: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    return [word for word in word_tokenize(text) if word.strip()]


class ThaiG2P:
    """Normalize, segment by script, phonemize each span, filter to the vocab."""

    def __init__(
        self,
        config: WayuTTSConfig,
        *,
        thai: Phonemizer | None = None,
        english: Phonemizer | None = _DEFAULT,
    ) -> None:
        self.config = config
        self.thai = thai or TltkThai()
        self.english = MisakiEnglish() if english is _DEFAULT else english

    def __call__(self, text: str) -> str:
        return self.trace(text).phonemes

    def trace(self, text: str) -> Trace:
        """Phonemize, keeping every span that was lost on the way."""
        result = Trace(phonemes="")
        normalized = normalize(text, dropped=result.dropped_symbols)
        result.phonemes = self._render(normalized, result.dropped_words)
        return result

    def _render(self, text: str, lost: list[tuple[str, str]]) -> str:
        pieces = []
        for thai, latin, space, other in _SPAN.findall(text):
            if thai:
                pieces.append(self._span("th", thai, self._thai_symbols, lost))
            elif latin and self.english is None:
                lost.append(("en", latin))  # no English G2P: say so, never drop in silence
            elif latin:
                pieces.append(self._span("en", latin, self._english_symbols, lost))
            elif space:
                pieces.append(" ")
            elif other in self.config.vocab:
                pieces.append(other)
        return re.sub(r" +", " ", "".join(pieces)).strip()

    def _span(self, language: str, surface: str, convert: Callable[[str], str],
              lost: list[tuple[str, str]]) -> str:
        symbols = convert(surface)
        if surface.strip() and not symbols.strip():
            lost.append((language, surface))
        return symbols

    def _thai_symbols(self, text: str) -> str:
        """tltk IPA -> vocab symbols: fold spellings, digits become tones."""
        raw = _TLTK_TAG.sub(" ", self.thai(text))
        out: list[str] = []
        for char in raw:
            if char.isdigit():
                # A digit is a tone only when it follows a phoneme.  One at a
                # span boundary is a leaked number, and emitting a tone there
                # would put a pitch contour on nothing.
                if out and out[-1] != " ":
                    out.append(self.config.tone_tokens.get(char, ""))
            elif char in _TLTK_SYLLABLE_BREAK:
                continue
            elif char.isspace():
                if out and out[-1] != " ":
                    out.append(" ")
            else:
                symbol = self.config.ipa_fixups.get(char, char)
                if symbol in self.config.vocab:
                    out.append(symbol)
        return "".join(out)

    def _english_symbols(self, text: str) -> str:
        """misaki already emits the model's symbols; the filter only catches its
        "unpronounceable" marker for a word no dictionary and no fallback knew."""
        symbols = self.english(text)
        kept = "".join(c for c in symbols if c == " " or c in self.config.vocab)
        return re.sub(r" +", " ", kept).strip()

    def count_tokens(self, phonemes: str) -> int:
        """Tokens the model will actually see -- unknown symbols are dropped."""
        return sum(1 for char in phonemes if char in self.config.vocab)


def encode(phonemes: str, vocab: Mapping[str, int]) -> list[int]:
    """Phoneme string -> embedding ids, skipping anything out of vocabulary."""
    return [vocab[char] for char in phonemes if char in vocab]
