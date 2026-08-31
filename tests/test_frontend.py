"""Frontend tests: normalization and G2P routing, no model download required.

The vocabulary here is a fixture, not the real one -- the real inventory ships
with the weights.  These assert *structure*: that numbers become words, that
tones stop being digits, that each script reaches the right G2P, and that
nothing is lost silently.
"""

import pytest

from wayu_tts import ThaiG2P, WayuTTSConfig, normalize
from wayu_tts.tts import _fit

IPA = "aeiouɛɔəɤɯʉæʌpbtdkgʔmnŋlrwjszfvhcʰɡɹʃʒθðːˈˌ"
TONES = "→˩↘↗↓"
PUNCTUATION = ' !"(),.:;?—“”…'


@pytest.fixture
def config() -> WayuTTSConfig:
    symbols = IPA + TONES + PUNCTUATION
    return WayuTTSConfig(
        architecture={"plbert": {"max_position_embeddings": 512}},
        vocab={symbol: index for index, symbol in enumerate(dict.fromkeys(symbols))},
        tone_tokens={"1": "→", "2": "˩", "3": "↘", "4": "↗", "5": "↓"},
        ipa_fixups={"ᴐ": "ɔ"},
        sample_rate=24000,
        voices=("f_young_clear",),
    )


@pytest.mark.parametrize("text, expected", [
    ("มี 3 คน", "มี สาม คน"),
    ("ราคา 1,250 บาท", "ราคา หนึ่งพันสองร้อยห้าสิบ บาท"),
    ("ลด 25%", "ลด ยี่สิบห้า เปอร์เซ็นต์"),
    ("฿500", "ห้าร้อย บาท"),
    ("36.5 องศา", "สามสิบหกจุดห้า องศา"),
    ("เมื่อ ม.ค.", "เมื่อ มกราคม"),
    ("เด็ก ๆ", "เด็กเด็ก"),
    ("ขำมาก 555", "ขำมาก ห้าห้าห้า"),       # internet laughter, not a quantity
    ("ราคา 555 บาท", "ราคา ห้าร้อยห้าสิบห้า บาท"),  # ... unless a unit follows
    ("จ้าา", "จ้า"),                          # held vowel/tone mark
    ("ว้อทททท", "ว้อท"),                       # held consonant, 3+ only
    ("ธรรม นกกระจอก", "ธรรม นกกระจอก"),        # a doubled consonant is ordinary spelling
])
def test_normalize_speaks_every_symbol(text, expected):
    assert normalize(text) == expected


def test_normalize_records_what_it_drops():
    dropped = set()
    assert "😀" not in normalize("สวัสดี 😀", dropped=dropped)
    assert dropped == {"😀"}


def test_thai_tones_are_symbols_not_digits(config):
    phonemes = ThaiG2P(config, english=None)("สวัสดีครับ")
    assert phonemes
    assert not any(char.isdigit() for char in phonemes)
    assert any(tone in phonemes for tone in TONES)


def test_each_script_reaches_its_own_g2p(config):
    g2p = ThaiG2P(config, thai=lambda text: "aa", english=lambda text: "ee")
    assert g2p("สวัสดี Grab ครับ") == "aa ee aa"


def test_latin_is_dropped_and_reported_without_an_english_g2p(config):
    trace = ThaiG2P(config, english=None).trace("เรียก Grab")
    assert trace.dropped_words == [("en", "Grab")]


def test_a_lost_span_is_recorded(config):
    trace = ThaiG2P(config, thai=lambda text: "", english=None).trace("สวัสดี")
    assert trace.dropped_words == [("th", "สวัสดี")]
    assert not trace.is_clean


def test_out_of_vocabulary_symbols_never_reach_the_model(config):
    phonemes = ThaiG2P(config, thai=lambda text: "aXbYc", english=None)("ก")
    assert phonemes == "abc"


@pytest.mark.parametrize("phonemes, budget, expected", [
    ("a b c", 10, ["a b c"]),
    ("a b c", 3, ["a b", "c"]),
    ("a b c", 1, ["a", "b", "c"]),
    ("abcdef", 2, ["abcdef"]),  # one oversized word is surfaced, not truncated
])
def test_chunks_stay_within_the_context_window(phonemes, budget, expected):
    assert list(_fit(phonemes, budget, len)) == expected
