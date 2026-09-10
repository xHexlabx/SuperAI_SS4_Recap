"""Small shared helpers: logging, Thai numerals, Thai name normalisation."""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"

# คำนำหน้าชื่อที่โผล่ในเอกสารจดทะเบียน — ตัดทิ้งก่อนเทียบชื่อเสมอ
TITLES = (
    "นางสาว", "น.ส.", "นาย", "นาง", "ดร.", "ดร", "ว่าที่ร้อยตรี", "ว่าที่ ร.ต.",
    "พลเอก", "พลตรี", "พันเอก", "พันตรี", "ร้อยเอก", "ร้อยโท", "ร้อยตรี",
    "ม.ล.", "ม.ร.ว.", "หม่อมหลวง", "หม่อมราชวงศ์", "คุณหญิง", "ท่านผู้หญิง",
    "Mr.", "Mrs.", "Miss", "Ms.",
)

_THAI_DIGITS = {
    "ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "ยี่": 2, "สาม": 3, "สี่": 4,
    "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9, "สิบ": 10,
}


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    # httpx logs one INFO line per request — 373 of those would bury the progress bar
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger("legal_act")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def thai_number(text: str) -> int | None:
    """'สอง' -> 2, 'สิบเอ็ด' -> 11, '3' -> 3. Returns None when it is not a number word."""
    text = text.strip()
    if not text:
        return None
    if re.fullmatch(r"\d+", text):
        return int(text)
    if text in _THAI_DIGITS:
        return _THAI_DIGITS[text]
    # สิบ / ยี่สิบ / สามสิบเอ็ด ...
    m = re.fullmatch(r"(ยี่|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)?สิบ(เอ็ด|[ก-๙]+)?", text)
    if m:
        tens = _THAI_DIGITS.get(m.group(1) or "หนึ่ง", 1)
        ones = _THAI_DIGITS.get(m.group(2) or "ศูนย์", 0)
        return tens * 10 + ones
    return None


def normalise_name(name: str) -> str:
    """Key used to compare two Thai person names: no title, no spaces, no punctuation.

    The competition data writes the same person three ways — 'นายอธิบดี คุ่ยประเสริฐ' in the
    clause, 'อธิบดี คุ่ยประเสริฐ' in `question`, and fname/lname columns in committee.csv —
    so every comparison goes through this.
    """
    s = str(name).strip()
    for t in TITLES:
        if s.startswith(t):
            s = s[len(t):]
            break
    s = re.sub(r"[\s​\.\,\(\)\[\]\"'`]+", "", s)
    return s


def best_match(name: str, pool: dict[str, str], cutoff: float = 0.75) -> str | None:
    """Snap `name` onto the closest key of `pool` (already-normalised -> display name).

    Exact first, then difflib. Used to pull an LLM-transcribed name back onto a real director,
    which is safe here because every signer in the data is a director of that company.
    """
    key = normalise_name(name)
    if not key:
        return None
    if key in pool:
        return pool[key]
    hit = difflib.get_close_matches(key, list(pool), n=1, cutoff=cutoff)
    return pool[hit[0]] if hit else None
