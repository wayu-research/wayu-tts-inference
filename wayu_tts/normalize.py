"""Turn written Thai into *speakable* Thai, before the G2P sees it.

tltk converts spelled-out Thai.  It has nothing to say about ``2,500``, ``฿``,
``25%`` or ``ๆ``, and silently drops what it cannot read -- so anything
meaning-bearing that is not a Thai letter is rewritten into words here first,
and everything else is folded to punctuation the model knows or dropped.

The tables are deliberately short and flat.  Extend them: they are ordinary
dicts, and a Thai deployment usually wants a few domain abbreviations of its
own.

    >>> normalize("ราคา 1,250 บาท (ลด 25%)")
    'ราคา หนึ่งพันสองร้อยห้าสิบ บาท (ลด ยี่สิบห้า เปอร์เซ็นต์ )'
"""

from __future__ import annotations

import re
import unicodedata

from pythainlp.util import num_to_thaiword, thai_digit_to_arabic_digit

try:  # renamed in pythainlp 5.0.5; the old name is deprecated
    from pythainlp.util import expand_maiyamok
except ImportError:  # pragma: no cover
    from pythainlp.util import maiyamok as expand_maiyamok

#: Written forms that are always read out in full.
ABBREVIATIONS = {
    "ม.ค.": "มกราคม", "ก.พ.": "กุมภาพันธ์", "มี.ค.": "มีนาคม", "เม.ย.": "เมษายน",
    "พ.ค.": "พฤษภาคม", "มิ.ย.": "มิถุนายน", "ก.ค.": "กรกฎาคม", "ส.ค.": "สิงหาคม",
    "ก.ย.": "กันยายน", "ต.ค.": "ตุลาคม", "พ.ย.": "พฤศจิกายน", "ธ.ค.": "ธันวาคม",
    "พ.ศ.": "พุทธศักราช", "ค.ศ.": "คริสต์ศักราช", "กทม.": "กรุงเทพมหานคร",
    "กรุงเทพฯ": "กรุงเทพมหานคร", "ฯลฯ": "และอื่น ๆ", "ดร.": "ด็อกเตอร์",
    "น.ส.": "นางสาว", "โทร.": "โทรศัพท์", "จ.": "จังหวัด", "อ.": "อำเภอ", "ต.": "ตำบล",
}

#: Symbols with a spoken Thai reading.
SYMBOLS = {
    "%": "เปอร์เซ็นต์", "+": "บวก", "=": "เท่ากับ", "≈": "ประมาณ", "~": "ประมาณ",
    "≠": "ไม่เท่ากับ", ">": "มากกว่า", "<": "น้อยกว่า", "×": "คูณ", "÷": "หาร",
    "±": "บวกลบ", "°": "องศา", "&": "และ", "@": "แอท", "#": "แฮชแท็ก", "/": "ทับ",
}

#: Currency signs, spoken *after* the amount they precede.
CURRENCIES = {"฿": "บาท", "$": "ดอลลาร์", "€": "ยูโร", "£": "ปอนด์", "¥": "เยน"}

#: Punctuation the model has tokens for; everything else is folded or dropped.
KEEP_PUNCTUATION = set(' !"(),.:;?—“”…')

#: Variants folded onto a token that exists.  A space means "pause, no word".
PUNCTUATION_FOLD = {
    "–": "—", "―": "—", "‒": "—", "-": " ", "_": " ", "*": " ", "|": " ", "\\": " ",
    "[": "(", "]": ")", "{": "(", "}": ")", "（": "(", "）": ")",
    "«": '"', "»": '"', "„": '"', "″": '"', "'": "", "‘": "", "’": "", "`": "",
    "、": ",", "，": ",", "。": ".", "．": ".", "‼": "!", "⁉": "?", "‽": "?",
    "•": " ", "·": " ", "→": " ", "←": " ", "⇒": " ",
}

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_RANGE = re.compile(r"(\d)\s*[-–—]\s*(\d)")
_AMOUNT = re.compile("([" + "".join(map(re.escape, CURRENCIES)) + r"])\s*(\d[\d,]*(?:\.\d+)?)")
# Thai internet laughter: 555 is "ห้าห้าห้า", not "five hundred and fifty-five" -- unless a
# unit noun follows, which makes it a real quantity ("555 บาท" is a price). The ambiguity
# is irreducible: "ได้ 555" could be either, and laughter is far more common online.
_QUANTIFIED = ("บาท", "คน", "ครั้ง", "ชิ้น", "ปี", "เดือน", "วัน", "เมตร", "กิโล", "กรัม",
               "ลิตร", "ชั่วโมง", "นาที", "อัน", "ตัว", "หน้า", "ล้าน", "แสน", "พัน")
_LAUGHTER = re.compile(r"(?<!\d)5{3,}(?!\d)(?!\s*(?:" + "|".join(_QUANTIFIED) + r"))")
# A vowel or tone mark typed several times over (จ้าา); tltk returns nothing for those.
_STUTTER = re.compile(r"([ะ-ฺ็-๎])\1+")
# The same for a held consonant (ว้อทททท).  Three or more, because a doubled consonant is
# ordinary across a syllable boundary -- ธรรม, นกกระจอก, กรรมกร are all correctly spelled.
_STUTTER_CONSONANT = re.compile(r"([ก-ฮ])\1{2,}")


def _number_to_words(match: re.Match) -> str:
    digits = match.group(0).replace(",", "")
    whole, _, fraction = digits.partition(".")
    try:
        spoken = num_to_thaiword(int(whole)) if whole else ""
        if fraction:
            spoken += "จุด" + "".join(num_to_thaiword(int(d)) for d in fraction)
    except (ValueError, OverflowError):  # beyond what Thai number words cover
        return digits
    return spoken or digits


def _fold_latin_accents(text: str) -> str:
    """café -> cafe, so the English G2P sees a word it can read.

    Only Latin letters are decomposed.  Thai vowels and tone marks are combining
    characters too, and stripping those would destroy the word.
    """
    folded = []
    for char in text:
        if char.isascii() or not char.isalpha():
            folded.append(char)
            continue
        decomposed = unicodedata.normalize("NFKD", char)
        is_latin = unicodedata.name(char, "").startswith("LATIN")
        folded.append("".join(c for c in decomposed if not unicodedata.combining(c))
                      if is_latin else char)
    return "".join(folded)


def _is_speakable(char: str) -> bool:
    """Thai, ASCII letters, and the punctuation the model has tokens for."""
    return (char in KEEP_PUNCTUATION
            or (char.isascii() and char.isalpha())
            or "฀" <= char <= "๿")


def normalize(text: str, dropped: set[str] | None = None) -> str:
    """Rewrite `text` so every meaning-bearing character is a spoken word.

    Pass a set as `dropped` to collect the characters that were discarded --
    useful when auditing a corpus for symbols worth adding to the tables above.
    """
    text = thai_digit_to_arabic_digit(text)
    text = _LAUGHTER.sub(lambda m: "ห้า" * len(m.group()), text)
    text = _STUTTER.sub(lambda m: m.group(1), text)
    text = _STUTTER_CONSONANT.sub(lambda m: m.group(1), text)
    text = _fold_latin_accents(text)

    for written, spoken in sorted(ABBREVIATIONS.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(written, spoken)

    text = _RANGE.sub(r"\1 ถึง \2", text)
    text = _AMOUNT.sub(lambda m: f" {m.group(2)} {CURRENCIES[m.group(1)]} ", text)
    for sign, unit in CURRENCIES.items():
        text = text.replace(sign, f" {unit} ")
    for symbol, spoken in SYMBOLS.items():
        text = text.replace(symbol, f" {spoken} ")

    text = _NUMBER.sub(_number_to_words, text)
    text = "".join(expand_maiyamok(text))  # ๆ -> repeat the previous word
    for variant, canonical in PUNCTUATION_FOLD.items():
        text = text.replace(variant, canonical)

    kept = []
    for char in text:
        if _is_speakable(char):
            kept.append(char)
        elif char.isspace():
            kept.append(" ")
        elif dropped is not None:
            dropped.add(char)
    return re.sub(r"\s+", " ", "".join(kept)).strip()
