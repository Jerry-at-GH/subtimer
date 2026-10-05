import re
import unicodedata

import pyopenjtalk

_ALPHA_TO_KATA = {
    "A": "エー",
    "B": "ビー",
    "C": "シー",
    "D": "ディー",
    "E": "イー",
    "F": "エフ",
    "G": "ジー",
    "H": "エイチ",
    "I": "アイ",
    "J": "ジェー",
    "K": "ケー",
    "L": "エル",
    "M": "エム",
    "N": "エヌ",
    "O": "オー",
    "P": "ピー",
    "Q": "キュー",
    "R": "アール",
    "S": "エス",
    "T": "ティー",
    "U": "ユー",
    "V": "ブイ",
    "W": "ダブリュー",
    "X": "エックス",
    "Y": "ワイ",
    "Z": "ゼット",
}
_DROP_CHARS = str.maketrans("", "", "、。？！,.!?「」『』（）()［］[]{}・…:;\"'`")
_CLEAN_RE = re.compile(r"[・]+")


class JapaneseKanaConverter:
    def text_to_kana(self, text: str) -> str:
        text = _CLEAN_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        katakana = pyopenjtalk.g2p(text, kana=True)
        if not katakana:
            return ""

        katakana = unicodedata.normalize("NFKC", katakana)
        katakana = "".join(_ALPHA_TO_KATA.get(ch.upper(), ch) for ch in katakana)
        katakana = katakana.translate(_DROP_CHARS)
        katakana = "".join(ch for ch in katakana if self._is_kana(ch))

        if not katakana:
            return ""

        hiragana = self._kata_to_hira(katakana)

        return " ".join(list(hiragana))

    def _kata_to_hira(self, text: str) -> str:
        result = []
        for ch in text:
            cp = ord(ch)
            if 0x30A1 <= cp <= 0x30F6:
                result.append(chr(cp - 0x60))
            elif ch == "ー":
                result.append("ー")
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def _is_kana(ch: str) -> bool:
        cp = ord(ch)
        return (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF) or ch == "ー"


KANA = [
    "あ", "い", "う", "え", "お",
    "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の",
    "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も",
    "や", "ゆ", "よ",
    "ら", "り", "る", "れ", "ろ",
    "わ", "を", "ん",

    "が", "ぎ", "ぐ", "げ", "ご",
    "ざ", "じ", "ず", "ぜ", "ぞ",
    "だ", "ぢ", "づ", "で", "ど",
    "ば", "び", "ぶ", "べ", "ぼ",

    "ぱ", "ぴ", "ぷ", "ぺ", "ぽ",

    "ぁ", "ぃ", "ぅ", "ぇ", "ぉ",
    "っ", "ゃ", "ゅ", "ょ", "ゎ",

    "ー",
]


BLANK_TOKEN = "<blank>"
BLANK_IDX = 0


class KanaVocab:
    def __init__(self):
        self.kana = list(KANA)

        self.stoi = {BLANK_TOKEN: BLANK_IDX}
        for i, k in enumerate(KANA):
            self.stoi[k] = i + 1
        self.itos = {v: k for k, v in self.stoi.items()}
        self.size = len(self.stoi)

    def encode(self, kana_str: str) -> list[int]:
        if not kana_str.strip():
            return []
        tokens = kana_str.split()
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, indices: list[int], collapse: bool = True) -> str:
        result = []
        prev = None
        for idx in indices:
            if collapse:
                if idx == BLANK_IDX:
                    prev = idx
                    continue
                if idx == prev:
                    continue
            token = self.itos.get(idx)
            if token and token != BLANK_TOKEN:
                result.append(token)
            prev = idx
        return "".join(result)


PHONEMES = [
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
]


class PhonemeVocab:
    def __init__(self):
        self.phonemes = list(PHONEMES)

        self.stoi = {BLANK_TOKEN: BLANK_IDX}
        for i, ph in enumerate(PHONEMES):
            self.stoi[ph] = i + 1
        self.itos = {v: k for k, v in self.stoi.items()}
        self.size = len(self.stoi)

    def encode(self, phoneme_str: str) -> list[int]:
        if not phoneme_str.strip():
            return []
        return [self.stoi[ph] for ph in phoneme_str.split() if ph in self.stoi]

    def decode(self, indices: list[int], collapse: bool = True) -> str:
        result = []
        prev = None
        for idx in indices:
            if collapse:
                if idx == BLANK_IDX:
                    prev = idx
                    continue
                if idx == prev:
                    continue
            token = self.itos.get(idx)
            if token and token != BLANK_TOKEN:
                result.append(token)
            prev = idx
        return " ".join(result)
