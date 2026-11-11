"""Explicit in-text rating detection and scrubbing for IMDb reviews.

Design goal: precision over recall. Every pattern is anchored so that dates
("9/11", "3/10/2005"), "24/7", fractions ("1/2 hour", "1/10 of the budget"),
"top 10", "one out of ten people" and actor "stars" are not matched.

Public API
----------
find_ratings(text)     -> list[Match]  (non-overlapping, left to right)
has_rating(text)       -> bool
scrub(text)            -> text with every matched rating expression deleted
invert_x10(text)       -> text with every "x/10" / "x out of 10" replaced by 10-x
clean_html(text)       -> replaces "<br />" tags with a space

Patterns (11):
  x10_slash     7/10, 7 / 10, 7.5/10, 10/10, "Rating: 7/10"
  x10_outof     7 out of 10, seven out of ten, 8 out of 10 stars
  x100_slash    85/100
  x5_slash      3/5, 3.5/5, 4/5 stars
  x5_outof      3 out of 5, three out of five stars, 3 out of 4 stars, 2 stars out of 5
  star_symbols  ***/****, ** 1/2 out of ****, *** out of *****
  digit_stars   3 stars, 3.5 stars, 2 1/2 stars, 4-star  (not "the 2 stars of")
  verb_word_stars  "I give it two stars", "rated this movie one star"
  verb_number   "I give it a 7", "I'd rate this film a 3."
  label_number  "Rating: 7", "My score = 6.5", "Verdict: 8"
  letter_grade  "Grade: B+", "I give it a C-." (case-sensitive letters)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BR = re.compile(r"<br\s*/?>", re.I)


def clean_html(text: str) -> str:
    return _BR.sub(" ", text)


WORD2NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NUM2WORD = {v: k for k, v in WORD2NUM.items()}

NUM10 = r"(?:10(?:\.0)?|[0-9](?:\.[0-9]{1,2})?)"          # 0 .. 10
WORD10 = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)"
NUM5 = r"(?:5(?:\.0)?|[0-4](?:\.[05])?)"                  # 0 .. 5 in halves
WORD5 = r"(?:zero|one|two|three|four|five)"
HALF = r"(?:\s*(?:1/2|½)|\s+and\s+a\s+half)"
# Optional label prefix so that "My rating: 7/10" is scrubbed as a whole.
PREFIX = (r"(?:\b(?:my\s+)?(?:overall\s+|final\s+|personal\s+)?"
          r"(?:rating|score|grade|verdict|vote)\s*[:=\-]?\s*)?")
# Things that follow "x out of y" when it is a proportion, not a rating.
NOT_PROPORTION = (r"(?!\s+(?:of|people|persons|times|viewers|men|women|kids|children|"
                  r"americans|critics|films|movies|episodes|segments|stories|cases|"
                  r"actors|scenes|jokes|songs|users|reviewers|members|doctors)\b)")
STARS_SUFFIX = r"(?:\s*-?\s*(?:stars?|points?|pts)\b)?"
NOT_FRACTION_AFTER = r"(?![\d/]|\.\d|\s*(?:th|ths|st|of)\b|th)"

_I = re.IGNORECASE

PATTERNS: dict[str, re.Pattern] = {
    "x10_slash": re.compile(
        PREFIX + rf"(?<![\w/])(?<!\d\.){NUM10}\s*/\s*10{NOT_FRACTION_AFTER}" + STARS_SUFFIX, _I),
    "x10_outof": re.compile(
        PREFIX + rf"(?<![\w])(?<!\d\.)(?:{NUM10}|{WORD10}){HALF}?(?:\s*(?:stars?|points?))?\s*[(\[]?\s*out\s+of\s+"
        rf"(?:a\s+possible\s+)?(?:10|ten)\b{NOT_PROPORTION}\s*[)\]]?" + STARS_SUFFIX, _I),
    "x100_slash": re.compile(
        PREFIX + rf"(?<![\w/])(?<!\d\.)(?:100|[1-9]?[0-9])(?:\s*/\s*100{NOT_FRACTION_AFTER}|\s+out\s+of\s+100\b{NOT_PROPORTION})"
        + STARS_SUFFIX, _I),
    "x5_slash": re.compile(
        PREFIX + rf"(?<![\w/])(?<!\d\.){NUM5}\s*/\s*5{NOT_FRACTION_AFTER}" + STARS_SUFFIX, _I),
    "x5_outof": re.compile(
        PREFIX + rf"(?<![\w])(?<!\d\.)(?:{NUM5}|{WORD5}){HALF}?\s*(?:stars?\s*)?[(\[]?\s*out\s+of\s+"
        rf"(?:a\s+possible\s+)?(?:5|five|4|four)\b{NOT_PROPORTION}\s*[)\]]?" + STARS_SUFFIX, _I),
    "star_symbols": re.compile(
        PREFIX + r"\*{1,10}" + HALF + r"?\s*[(\[]?\s*(?:/|out\s+of|of)\s*(?:\*{3,10}|(?:4|four|5|five|10|ten)\b)\s*[)\]]?"
        + STARS_SUFFIX, _I),
    "digit_stars": re.compile(
        PREFIX + r"(?<!the )(?<!its )(?<![\w/])(?<!\d\.)(?:10|[0-9](?:\.5)?)" + HALF
        + r"?\s*-?\s*stars?\b(?!\s+(?:of|in|and|are|were|who|from|on|as)\b)", _I),
    "verb_word_stars": re.compile(
        r"\b(?:give|gave|giving|gives|rate|rated|rating|award|awarded|deserves?|earns?)\s+"
        r"(?:it|this(?:\s+(?:movie|film|one|flick|show))?)?\s*(?:a\s+|an\s+)?"
        r"(?:solid\s+|generous\s+|measly\s+|mere\s+)?(?:zero|one|two|three|four|five|half\s+a|a\s+half)"
        r"(?:\s+and\s+a\s+half)?\s*-?\s*stars?\b", _I),
    "verb_number": re.compile(
        r"\b(?:give|gave|giving|gives|rate|rated|rating|award|awarded|score|scored)\s+"
        r"(?:it|this(?:\s+(?:movie|film|one|flick|show|piece))?)\s+(?:a\s+|an\s+)?"
        r"(?:solid\s+|generous\s+|strong\s+|measly\s+|low\s+|high\s+|mere\s+)?"
        rf"(?:{NUM10}(?![\w/%]|\.\d)(?!\s*(?:/|out\b|stars?\b|%|hours?\b|minutes?\b|times?\b|thumbs?\b|"
        r"more\b|chances?\b|tries\b|years?\b|of\b|or\b|to\b|-))"
        rf"|{WORD10}(?=\s*(?:[.,!;:)]|$)))", _I),
    "label_number": re.compile(
        r"\b(?:my\s+)?(?:overall\s+|final\s+|personal\s+)?(?:(?:rating|score|verdict|vote)\s*[:=]|rating\s*[,\-]?)\s*"
        rf"{NUM10}(?![\w/%]|\.\d)(?!\s*(?:/|out\b|stars?\b|times?\b|points?\b))", _I),
    "letter_grade": re.compile(
        r"(?:\b(?i:(?:my\s+)?(?:overall\s+|final\s+)?(?:grade|rating|score|verdict))\s*[:=\-]\s*"
        r"|\b(?i:give|gave|giving|gives|rate|rated|grade|graded)\s+"
        r"(?i:it|this(?:\s+(?:movie|film|one))?)\s+(?i:an?\s+))"
        r"[A-F][+-]?(?![\w+-])(?=\s*(?:[.,!;:)]|$|\s+(?i:for|from|on|overall|and|but)\b))"),
}


@dataclass
class Match:
    pattern: str
    start: int
    end: int
    text: str


def find_ratings(text: str) -> list[Match]:
    """All non-overlapping rating matches. Earlier start wins; ties -> longer."""
    cands = []
    for name, pat in PATTERNS.items():
        for m in pat.finditer(text):
            if m.end() > m.start():
                cands.append(Match(name, m.start(), m.end(), m.group(0)))
    cands.sort(key=lambda m: (m.start, -(m.end - m.start)))
    out, last_end = [], -1
    for m in cands:
        if m.start >= last_end:
            out.append(m)
            last_end = m.end
    return out


def has_rating(text: str) -> bool:
    return any(p.search(text) for p in PATTERNS.values())


_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def scrub(text: str) -> str:
    """Delete every matched rating expression, then tidy the seams."""
    ms = find_ratings(text)
    if not ms:
        return text
    pieces, pos = [], 0
    for m in ms:
        pieces.append(text[pos:m.start])
        pos = m.end
        # drop a trailing "." / "!" if the text before already ended a sentence
        before = "".join(pieces).rstrip()
        rest = text[pos:]
        stripped = rest.lstrip()
        if stripped[:1] in {".", "!"} and (not before or before[-1] in ".!?:;,-"):
            pos += len(rest) - len(stripped) + 1
    pieces.append(text[pos:])
    out = "".join(pieces)
    out = _EMPTY_BRACKETS.sub("", out)
    out = re.sub(r"\s+([.,!?;:])(?=\s|$)", r"\1", out) if out != text else out
    out = _MULTISPACE.sub(" ", out).strip()
    return out


# --------------------------------------------------------------------- scores
_NUM_RE = re.compile(r"\d+(?:\.\d+)?|" + WORD10, _I)


def _first_number(s: str):
    m = _NUM_RE.search(s)
    if not m:
        return None
    t = m.group(0).lower()
    return float(WORD2NUM[t]) if t in WORD2NUM else float(t)


def x10_score(m: Match):
    """Numeric score of an x/10 match (x10_slash / x10_outof), else None."""
    if m.pattern not in ("x10_slash", "x10_outof"):
        return None
    # strip an optional label prefix ("rating: ") before reading the number
    core = PATTERNS[m.pattern].pattern  # noqa: F841 (documentation only)
    body = re.sub(r"^(?:my\s+)?(?:overall\s+|final\s+|personal\s+)?(?:rating|score|grade|verdict|vote)"
                  r"\s*[:=\-]?\s*", "", m.text, flags=_I)
    return _first_number(body)


def last_x10_score(text: str):
    """Score of the LAST x/10 expression in the review (the reviewer's verdict)."""
    vals = [x10_score(m) for m in find_ratings(text)]
    vals = [v for v in vals if v is not None]
    return vals[-1] if vals else None


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def invert_x10(text: str) -> str:
    """Replace every x/10 or 'x out of 10' rating by (10-x); other text unchanged."""
    ms = [m for m in find_ratings(text) if m.pattern in ("x10_slash", "x10_outof")]
    if not ms:
        return text
    out, pos = [], 0
    for m in ms:
        out.append(text[pos:m.start])
        seg = m.text
        # find the score token (skip a label prefix)
        pre = re.match(r"^(?:my\s+)?(?:overall\s+|final\s+|personal\s+)?(?:rating|score|grade|verdict|vote)"
                       r"\s*[:=\-]?\s*", seg, flags=_I)
        off = pre.end() if pre else 0
        nm = _NUM_RE.search(seg, off)
        tok = nm.group(0)
        if tok.lower() in WORD2NUM:
            new = NUM2WORD[10 - WORD2NUM[tok.lower()]]
            if tok[0].isupper():
                new = new.capitalize()
        else:
            new = _fmt(round(10 - float(tok), 2))
        out.append(seg[:nm.start()] + new + seg[nm.end():])
        pos = m.end
    out.append(text[pos:])
    return "".join(out)
