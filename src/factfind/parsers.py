"""Per-kind value parsers.

Every parser takes the raw text a reader produced for a value region and the
Field it is supposed to fill, and returns a ParseResult. A parser either
returns a canonical, validated value or it refuses. Refusing is always
preferred to guessing: a refusal becomes "unsure" downstream, a wrong value
becomes a wrong-fill, and the brief weights those very differently.

Parsers also return `candidates`: the top alternatives considered, so the
pipeline can show top-3 for an unsure field.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field as dc_field
from datetime import date, datetime
from typing import Any, Callable

from rapidfuzz import fuzz, process as fuzz_process

from .knowledge import Field, normalise_label

# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass
class ParseResult:
    """Outcome of reading one value region for one field."""
    value: Any = None
    # 0..1, how well the raw text fits this kind's grammar. Not a model score.
    parse_score: float = 0.0
    valid: bool = False
    # (candidate_value, score) best first, for the unsure top-3.
    candidates: list[tuple[Any, float]] = dc_field(default_factory=list)
    # Human-readable reason a value was refused, for error analysis.
    reason: str = ""
    raw: str = ""

    @property
    def filled(self) -> bool:
        return self.valid and self.value not in (None, "")

    def top_candidates(self, n: int = 3) -> list[Any]:
        return [c for c, _ in self.candidates[:n]]


# --------------------------------------------------------------------------
# Shared text helpers
# --------------------------------------------------------------------------

# Characters OCR habitually confuses, applied only where the kind says the
# answer must be digits. Doing this globally would corrupt names.
DIGIT_CONFUSIONS = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8",
    "G": "6", "b": "6",
    "T": "7",
    "g": "9", "q": "9",
})

# The reverse, for NI numbers where a stray digit sits in a letter slot.
LETTER_CONFUSIONS = str.maketrans({
    "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A",
})

NOISE_PREFIX = re.compile(
    r"^\s*(?:[-:•*>•–—]+\s*)+", re.UNICODE
)
NOISE_SUFFIX = re.compile(r"[\s\.,;:\-_]+$")

# Tokens that mean "the writer left this blank" -- must never become a value.
BLANK_TOKENS = {
    "", "-", "--", "---", "n/a", "na", "n a", "nil", "none", "tbc", "tbd",
    "?", "??", "???", "x", "xx", "not known", "unknown", "not applicable",
    "not stated", "blank", "to follow", "see above", "as above", "same",
    ".", "..", "...", "/", "//", "_", "__", "___",
}


def clean_raw(text: str) -> str:
    """Strip leading bullets/colons and trailing punctuation."""
    if text is None:
        return ""
    text = str(text).replace(" ", " ")
    text = NOISE_PREFIX.sub("", text)
    text = NOISE_SUFFIX.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def is_blank(text: str) -> bool:
    return normalise_label(text) in BLANK_TOKENS or not clean_raw(text)


def digits_only(text: str, translate: bool = True) -> str:
    t = text.translate(DIGIT_CONFUSIONS) if translate else text
    return re.sub(r"\D", "", t)


# --------------------------------------------------------------------------
# Choice / yes-no
# --------------------------------------------------------------------------

YES_FORMS = {
    "y", "yes", "yeah", "yep", "true", "t", "1", "tick", "ticked", "x",
    "correct", "affirmative", "has", "does", "in place", "yes please",
}
NO_FORMS = {
    "n", "no", "nope", "false", "f", "0", "none", "nil", "not", "never",
    "negative", "no none", "doesnt", "does not", "has not", "not in place",
}
# "x" is genuinely ambiguous: a tick in a box, or a crossing-out. Treated as
# yes only when nothing else matches, and always with a reduced score.
AMBIGUOUS_YES = {"x", "tick", "ticked"}


def parse_yes_no(text: str, fld: Field) -> ParseResult:
    raw = clean_raw(text)
    if is_blank(raw) and normalise_label(raw) not in (YES_FORMS | NO_FORMS):
        return ParseResult(raw=raw, reason="blank")
    norm = normalise_label(raw)

    if norm in NO_FORMS:
        return ParseResult(value=False, parse_score=1.0, valid=True, raw=raw,
                           candidates=[(False, 1.0), (True, 0.0)])
    if norm in YES_FORMS:
        score = 0.55 if norm in AMBIGUOUS_YES else 1.0
        return ParseResult(value=True, parse_score=score, valid=True, raw=raw,
                           candidates=[(True, score), (False, 1 - score)])

    # A phrase: look for a leading yes/no token.
    first = norm.split(" ")[0] if norm else ""
    if first in NO_FORMS:
        return ParseResult(value=False, parse_score=0.8, valid=True, raw=raw,
                           candidates=[(False, 0.8), (True, 0.2)])
    if first in YES_FORMS:
        return ParseResult(value=True, parse_score=0.8, valid=True, raw=raw,
                           candidates=[(True, 0.8), (False, 0.2)])

    # Fuzzy, but only against short forms, and only at high similarity.
    best_yes = max((fuzz.ratio(norm, f) for f in YES_FORMS if len(f) > 2), default=0)
    best_no = max((fuzz.ratio(norm, f) for f in NO_FORMS if len(f) > 2), default=0)
    if max(best_yes, best_no) >= 85:
        val = best_yes >= best_no
        score = max(best_yes, best_no) / 100.0 * 0.8
        return ParseResult(value=val, parse_score=score, valid=True, raw=raw,
                           candidates=[(val, score), (not val, 1 - score)])

    return ParseResult(raw=raw, reason=f"not a yes/no value: {raw!r}",
                       candidates=[(True, 0.0), (False, 0.0)])


def parse_choice(text: str, fld: Field) -> ParseResult:
    """Snap to an allowed choice. Never emits a value outside the list unless
    the field sets allowOther, and then only with a reduced score."""
    raw = clean_raw(text)
    if is_blank(raw):
        return ParseResult(raw=raw, reason="blank")
    norm = normalise_label(raw)

    choices = fld.resolved_choices()
    if not choices:
        return ParseResult(raw=raw, reason="choice field with no choices")

    # surface form -> canonical value
    surface_to_value: dict[str, str] = {}
    for ch in choices:
        for form in ch.all_surface_forms():
            surface_to_value.setdefault(form, ch.value)

    # 1. exact surface match
    if norm in surface_to_value:
        val = surface_to_value[norm]
        others = [(v, 0.0) for v in dict.fromkeys(surface_to_value.values()) if v != val][:2]
        return ParseResult(value=val, parse_score=1.0, valid=True, raw=raw,
                           candidates=[(val, 1.0)] + others)

    # 2. fuzzy against every surface form
    matches = fuzz_process.extract(
        norm, list(surface_to_value.keys()), scorer=fuzz.WRatio, limit=12
    )
    ranked: list[tuple[str, float]] = []
    seen: set[str] = set()
    for form, score, _ in matches:
        val = surface_to_value[form]
        if val in seen:
            continue
        seen.add(val)
        # Very short surface forms ("m", "s", "d") fuzzy-match almost anything.
        penalty = 0.75 if len(form) <= 2 and len(norm) > 3 else 1.0
        ranked.append((val, score / 100.0 * penalty))
    ranked.sort(key=lambda p: -p[1])

    if ranked and ranked[0][1] >= 0.86:
        # Require a margin over the runner-up: two near-equal choices mean the
        # reader cannot actually tell them apart.
        margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
        score = ranked[0][1] * (1.0 if margin >= 0.08 else 0.7)
        return ParseResult(value=ranked[0][0], parse_score=score, valid=True,
                           raw=raw, candidates=ranked[:5])

    if fld.allow_other:
        # Keep the text for the adviser to check, but score it low so the
        # confidence model and thresholds can route it to "unsure".
        return ParseResult(value=raw, parse_score=0.35, valid=True, raw=raw,
                           reason="no choice matched; kept as other",
                           candidates=[(raw, 0.35)] + ranked[:4])

    return ParseResult(raw=raw, reason=f"no allowed choice matches {raw!r}",
                       candidates=ranked[:5])


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

MONTHS = {}
for _i in range(1, 13):
    MONTHS[calendar.month_name[_i].lower()] = _i
    MONTHS[calendar.month_abbr[_i].lower()] = _i
MONTHS.update({"sept": 9, "septmber": 9, "febuary": 2, "jan": 1, "dec": 12})

_DATE_SEP = r"[\/\-\.\s]"
DATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 01/02/1980, 1-2-80
    (re.compile(rf"^(\d{{1,2}}){_DATE_SEP}(\d{{1,2}}){_DATE_SEP}(\d{{2}}|\d{{4}})$"), "dmy"),
    # 1980-02-01 (ISO)
    (re.compile(rf"^(\d{{4}}){_DATE_SEP}(\d{{1,2}}){_DATE_SEP}(\d{{1,2}})$"), "ymd"),
    # 1st Feb 1980 / 1 February 1980
    (re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?\s+(\d{2}|\d{4})$"), "dMy"),
    # Feb 1 1980 / February 1st, 1980
    (re.compile(r"^([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2}|\d{4})$"), "Mdy"),
    # 010280 / 01021980
    (re.compile(r"^(\d{2})(\d{2})(\d{2}|\d{4})$"), "dmy"),
]


def _expand_year(y: int, today: date) -> int:
    """Two-digit year -> four. A fact-find date is a birth/issue/expiry date,
    so a year that would land in the near future is read as last century."""
    if y >= 100:
        return y
    century = today.year // 100 * 100
    candidate = century + y
    # Allow up to 10 years ahead (passport/licence expiry), else go back 100.
    return candidate if candidate <= today.year + 10 else candidate - 100


def parse_date(text: str, fld: Field, today: date | None = None) -> ParseResult:
    """Parse a date. Ambiguous day/month (both <= 12) is reported through the
    candidate list, and the score is cut, because UK vs US order is a
    classic silent wrong-fill."""
    today = today or date.today()
    raw = clean_raw(text)
    if is_blank(raw):
        return ParseResult(raw=raw, reason="blank")

    norm = raw.lower().strip()
    norm = re.sub(r"[,]", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    # OCR often reads 'O' for '0' inside an otherwise numeric date.
    if re.fullmatch(r"[\dOolISZB\/\-\.\s]+", norm):
        norm = norm.translate(DIGIT_CONFUSIONS).lower()
        norm = re.sub(r"\s+", " ", norm).strip()

    for pattern, order in DATE_PATTERNS:
        m = pattern.match(norm)
        if not m:
            continue
        a, b, c = m.groups()
        try:
            if order == "dmy":
                d_, mo, y = int(a), int(b), _expand_year(int(c), today)
            elif order == "ymd":
                y, mo, d_ = int(a), int(b), int(c)
            elif order == "dMy":
                d_ = int(a)
                mo = MONTHS.get(b.lower().rstrip("."), 0)
                y = _expand_year(int(c), today)
            else:  # Mdy
                mo = MONTHS.get(a.lower().rstrip("."), 0)
                d_ = int(b)
                y = _expand_year(int(c), today)
        except (ValueError, TypeError):
            continue
        if not mo:
            continue

        candidates: list[tuple[Any, float]] = []
        primary = _safe_date(y, mo, d_)
        if primary is None:
            # 13/05 read as d=13,m=5 failed? try swapping before giving up.
            swapped = _safe_date(y, d_, mo)
            if swapped is None:
                return ParseResult(raw=raw, reason=f"not a real date: {raw!r}")
            primary, d_, mo = swapped, mo, d_

        score = 1.0
        ambiguous = order in ("dmy", "ymd") and d_ <= 12 and mo <= 12 and d_ != mo
        if ambiguous and order == "dmy":
            alt = _safe_date(y, d_, mo)  # the US reading
            if alt is not None:
                # UK convention wins (this is a UK fact find) but not blindly.
                score = 0.72
                candidates = [(primary.isoformat(), score), (alt.isoformat(), 0.28)]

        if not candidates:
            candidates = [(primary.isoformat(), score)]

        res = ParseResult(value=primary.isoformat(), parse_score=score,
                          valid=True, raw=raw, candidates=candidates)
        return _apply_date_rules(res, fld, primary, today)

    return ParseResult(raw=raw, reason=f"unparseable date: {raw!r}")


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _apply_date_rules(res: ParseResult, fld: Field, value: date, today: date) -> ParseResult:
    """minAge and basic sanity. A date failing these is refused, not filled."""
    if value.year < 1900:
        return ParseResult(raw=res.raw, candidates=res.candidates,
                           reason=f"year {value.year} before 1900")
    if fld.min_age is not None:
        age = today.year - value.year - ((today.month, today.day) < (value.month, value.day))
        if age < fld.min_age:
            # The knowledge file says: younger than this is likely someone
            # else's date. Mark unsure rather than fill.
            return ParseResult(raw=res.raw, candidates=res.candidates,
                               reason=f"age {age} below minAge {fld.min_age}")
        if age > 120:
            return ParseResult(raw=res.raw, candidates=res.candidates,
                               reason=f"implausible age {age}")
    # Birth dates cannot be in the future; expiry dates can.
    is_birth = "birth" in fld.key.lower() or fld.min_age is not None
    if is_birth and value > today:
        return ParseResult(raw=res.raw, candidates=res.candidates,
                           reason="birth date in the future")
    return res


# --------------------------------------------------------------------------
# Numbers and money
# --------------------------------------------------------------------------

MONEY_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "bn": 1_000_000_000}
# Written frequencies that turn an amount into a different amount if ignored.
PER_PERIOD = re.compile(
    r"\b(p\s*/?\s*m|per\s+month|pm|monthly|a\s+month|p\s*/?\s*w|per\s+week|pw|weekly|"
    r"a\s+week|p\s*/?\s*a|per\s+annum|pa|per\s+year|annually|yearly|a\s+year|"
    r"per\s+quarter|quarterly)\b",
    re.IGNORECASE,
)


def _extract_number_with_repair(raw: str) -> tuple[float | None, float, str]:
    """Parse a number, retrying with OCR digit-confusions if it has no digits.

    Handwritten "55" is routinely read as "SS", "20" as "2O", "12" as "l2".
    The knowledge file says these fields take digits, so a letter in a
    digit-only field is a misread rather than a value. The repaired reading is
    scored down so the confidence model can still route it to unsure.
    """
    value, score, note = _extract_number(raw)
    if value is not None:
        return value, score, note
    if not raw or re.search(r"\d", raw):
        return value, score, note          # it had digits; it is just invalid
    # No digits at all: the whole thing may be letters standing in for digits.
    repaired = raw.translate(DIGIT_CONFUSIONS)
    if not re.search(r"\d", repaired):
        return value, score, note
    value, score, _note = _extract_number(repaired)
    if value is None:
        return None, 0.0, note
    return value, score * 0.65, (note + " digit-repair").strip()


def _extract_number(raw: str) -> tuple[float | None, float, str]:
    """Return (value, score, note). Handles 1,234.56 / 1.234,56 / 45k / £1 200."""
    text = raw.strip()
    text = re.sub(r"[£$€]", "", text)
    text = text.replace(" ", " ")

    mult = 1.0
    note = ""
    m = re.search(r"(?i)(\d)\s*(k|m|bn)\b", text)
    if m:
        mult = MONEY_MULTIPLIERS[m.group(2).lower()]
        text = re.sub(r"(?i)\s*(k|m|bn)\b", "", text)
        note = f"x{int(mult)}"

    # Keep only the numeric core.
    core = re.search(r"-?[\d.,\s]+", text)
    if not core:
        return None, 0.0, "no digits"
    s = core.group(0).strip()
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)  # "1 200" -> "1200"
    if not re.search(r"\d", s):
        return None, 0.0, "no digits"

    score = 1.0
    has_comma, has_dot = "," in s, "." in s
    if has_comma and has_dot:
        # Whichever appears last is the decimal separator.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma:
        parts = s.split(",")
        if len(parts[-1]) == 3 and all(len(p) <= 3 for p in parts[:-1]):
            s = s.replace(",", "")          # thousands separator
        elif len(parts[-1]) in (1, 2):
            s = s.replace(",", ".")         # european decimal
        else:
            s = s.replace(",", "")
            score = 0.8
    elif has_dot:
        parts = s.split(".")
        if len(parts) > 2:
            s = s.replace(".", "")          # 1.234.567
        elif len(parts[-1]) == 3 and len(parts[0]) <= 3:
            # 1.200 is ambiguous: 1200 or 1.2? Thousands is far more common
            # for money on a fact find, but flag the doubt.
            s = s.replace(".", "")
            score = 0.7
            note = (note + " ambiguous-thousands").strip()
    try:
        val = float(s) * mult
    except ValueError:
        return None, 0.0, "unparseable number"
    return val, score, note


def parse_number(text: str, fld: Field) -> ParseResult:
    raw = clean_raw(text)
    if is_blank(raw):
        return ParseResult(raw=raw, reason="blank")
    val, score, note = _extract_number_with_repair(raw)
    if val is None:
        return ParseResult(raw=raw, reason=note)

    if val != int(val) and fld.kind == "number":
        # Ages, counts and terms are whole numbers; a decimal means misread.
        if abs(val - round(val)) < 1e-9:
            val = float(round(val))
        else:
            score *= 0.6
    out: float | int = int(round(val)) if float(val).is_integer() else val

    if fld.minimum is not None and out < fld.minimum:
        return ParseResult(raw=raw, candidates=[(out, score)],
                           reason=f"{out} below min {fld.minimum}")
    if fld.maximum is not None and out > fld.maximum:
        return ParseResult(raw=raw, candidates=[(out, score)],
                           reason=f"{out} above max {fld.maximum}")
    if note:
        score *= 0.9
    return ParseResult(value=out, parse_score=score, valid=True, raw=raw,
                       candidates=[(out, score)], reason=note)


def parse_money(text: str, fld: Field) -> ParseResult:
    raw = clean_raw(text)
    if is_blank(raw):
        return ParseResult(raw=raw, reason="blank")
    val, score, note = _extract_number_with_repair(raw)
    if val is None:
        return ParseResult(raw=raw, reason=note)
    if val < 0:
        return ParseResult(raw=raw, reason="negative amount")
    if val > 100_000_000:
        return ParseResult(raw=raw, candidates=[(val, score)],
                           reason=f"implausible amount {val}")

    period = PER_PERIOD.search(raw)
    candidates: list[tuple[Any, float]] = []
    if period:
        # The amount carries a period the field does not model. Monthly vs
        # annual salary is a 12x silent error, so refuse to fill it blind.
        note = (note + f" period:{period.group(0).strip()}").strip()
        score *= 0.5
        candidates = [(round(val, 2), score)]
        annual = round(val * 12, 2) if re.search(r"(?i)month|pm|p/m", period.group(0)) else None
        if annual:
            candidates.append((annual, score * 0.8))

    out = round(val, 2)
    if not candidates:
        candidates = [(out, score)]
    return ParseResult(value=out, parse_score=score, valid=True, raw=raw,
                       candidates=candidates, reason=note)


# --------------------------------------------------------------------------
# Identity-bearing strings: these must be exact or refused
# --------------------------------------------------------------------------

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}$")
COMMON_TLDS = {"com", "co.uk", "org", "net", "org.uk", "gov.uk", "ac.uk", "io",
               "me", "uk", "email", "outlook.com", "gmail.com"}
COMMON_DOMAINS = ["gmail.com", "googlemail.com", "hotmail.com", "hotmail.co.uk",
                  "outlook.com", "yahoo.com", "yahoo.co.uk", "icloud.com",
                  "live.co.uk", "btinternet.com", "sky.com", "aol.com",
                  "msn.com", "me.com", "protonmail.com", "virginmedia.com"]


def parse_email(text: str, fld: Field) -> ParseResult:
    """Zero silent errors required. Only a string that is already a valid
    address is accepted; near-misses become candidates, never values."""
    raw = clean_raw(text)
    if is_blank(raw):
        return ParseResult(raw=raw, reason="blank")

    s = raw.replace(" ", "").strip().strip(".,;:")
    # Spoken/written forms.
    s = re.sub(r"(?i)\s*\(?at\)?\s*@", "@", s)
    s = re.sub(r"(?i)^(e\s*-?\s*mail|email)\s*:?", "", s)
    s = s.replace("’", "").replace("'", "")
    if s.count("@") == 0:
        m = re.search(r"(?i)(.+?)\s*\(?\s*at\s*\)?\s*(.+\..+)", raw)
        if m:
            s = f"{m.group(1).strip()}@{m.group(2).strip()}".replace(" ", "")
    s = s.lower()

    if EMAIL_RE.match(s):
        local, _, domain = s.partition("@")
        score = 1.0
        cands: list[tuple[Any, float]] = [(s, 1.0)]
        if domain not in COMMON_DOMAINS:
            near = fuzz_process.extractOne(domain, COMMON_DOMAINS, scorer=fuzz.ratio)
            if near and 80 <= near[1] < 100:
                # e.g. "gmial.com": plausible OCR slip. Offer the fix as a
                # candidate and drop the score; do not silently rewrite it.
                score = 0.6
                cands.append((f"{local}@{near[0]}", 0.55))
        return ParseResult(value=s, parse_score=score, valid=True, raw=raw,
                           candidates=cands)

    # Try one conservative repair so a near-miss can at least be offered.
    repaired = re.sub(r"\.{2,}", ".", s)
    repaired = re.sub(r"(?i)@+", "@", repaired)
    if EMAIL_RE.match(repaired):
        return ParseResult(raw=raw, candidates=[(repaired, 0.5)],
                           reason="email only valid after repair; left unsure")
    return ParseResult(raw=raw, reason=f"not a valid email: {raw!r}")


# UK numbers; the pipeline is a UK fact find.
def parse_phone(text: str, fld: Field) -> ParseResult:
    """Zero silent errors required. Digit count and prefix must be right."""
    raw = clean_raw(text)
    if is_blank(raw):
        return ParseResult(raw=raw, reason="blank")

    s = raw.strip()
    had_plus = s.lstrip().startswith("+")
    d = digits_only(s, translate=True)
    if not d:
        return ParseResult(raw=raw, reason="no digits")

    # Normalise international forms to a national 0-leading number.
    if d.startswith("0044"):
        d = "0" + d[4:]
    elif had_plus and d.startswith("44"):
        d = "0" + d[2:]
    elif d.startswith("44") and len(d) == 12:
        d = "0" + d[2:]

    if len(d) == 10 and d[0] in "1237" and not d.startswith("0"):
        d = "0" + d          # leading zero dropped by the writer
        score_penalty = 0.85
    else:
        score_penalty = 1.0

    if len(d) != 11 or not d.startswith("0"):
        return ParseResult(raw=raw, candidates=[(d, 0.3)],
                           reason=f"UK number must be 11 digits starting 0, got {len(d)}")
    if d.startswith("00"):
        return ParseResult(raw=raw, reason="invalid prefix 00")

    is_mobile = d.startswith("07")
    wants_mobile = "mobile" in fld.key.lower() or "mobile" in fld.label.lower()
    score = 1.0 * score_penalty
    if wants_mobile and not is_mobile:
        # A landline in the mobile field is a real mismatch worth flagging.
        score *= 0.6
    # How much digit-confusion repair did we need? Heavy repair = less trust.
    untranslated = re.sub(r"\D", "", s)
    if len(untranslated) < len(re.sub(r"\D", "", s.translate(DIGIT_CONFUSIONS))):
        score *= 0.85

    return ParseResult(value=d, parse_score=score, valid=True, raw=raw,
                       candidates=[(d, score)])


# QQ123456C -- 2 letters, 6 digits, 1 letter A-D. Excluded first letters and
# pairs per HMRC rules.
NI_RE = re.compile(r"^(?!BG|GB|NK|KN|TN|NT|ZZ)[ABCEGHJ-PRSTW-Z][ABCEGHJ-NPRSTW-Z]\d{6}[A-D]$")


def parse_ni_number(text: str, fld: Field) -> ParseResult:
    """Zero silent errors required. Must match the HMRC pattern exactly."""
    raw = clean_raw(text)
    if is_blank(raw):
        return ParseResult(raw=raw, reason="blank")

    s = re.sub(r"[\s\-\.]", "", raw).upper()
    s = re.sub(r"^(NI|NINO|NINUMBER|NATIONALINSURANCE(NUMBER)?)[:]?", "", s)

    if NI_RE.match(s):
        return ParseResult(value=s, parse_score=1.0, valid=True, raw=raw,
                           candidates=[(s, 1.0)])

    # One structured repair attempt: letters in letter slots, digits in digit
    # slots. Offered as a candidate only -- never filled silently.
    if len(s) == 9:
        fixed = (
            s[0:2].translate(LETTER_CONFUSIONS)
            + s[2:8].translate(DIGIT_CONFUSIONS)
            + s[8].translate(LETTER_CONFUSIONS)
        )
        if NI_RE.match(fixed):
            return ParseResult(raw=raw, candidates=[(fixed, 0.5)],
                               reason="NI number only valid after character repair; left unsure")
    return ParseResult(raw=raw, reason=f"does not match NI number pattern: {raw!r}")


# --------------------------------------------------------------------------
# Names and free text
# --------------------------------------------------------------------------

# Lower-case connectors inside a surname ("van der Berg"). "o" is deliberately
# absent: a standalone "o" is nearly always a misread O'Brien, not a particle.
NAME_PARTICLES = {"de", "la", "van", "von", "der", "den", "del", "di", "da",
                  "al", "bin", "ibn", "le", "dos", "das"}
TITLE_WORDS = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "lady", "rev"}
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z'’\-\.\s]{0,78}$")


def titlecase_name(s: str) -> str:
    """Title-case respecting McDonald, O'Brien, van der Berg, hyphenates."""
    out_parts = []
    for word in s.split():
        low = word.lower()
        if low in NAME_PARTICLES and out_parts:
            out_parts.append(low)
            continue
        piece = "-".join(_cap_token(p) for p in word.split("-"))
        out_parts.append(piece)
    return " ".join(out_parts)


def _cap_token(tok: str) -> str:
    if not tok:
        return tok
    low = tok.lower()
    if low.startswith("mc") and len(low) > 2:
        return "Mc" + low[2:].capitalize()
    if low.startswith("o'") and len(low) > 2:
        return "O'" + low[2:].capitalize()
    if "'" in low:
        a, _, b = low.partition("'")
        return a.capitalize() + "'" + b.capitalize()
    return low.capitalize()


def parse_name(text: str, fld: Field) -> ParseResult:
    raw = clean_raw(text)
    if is_blank(raw):
        return ParseResult(raw=raw, reason="blank")

    s = re.sub(r"[^A-Za-z'’\-\.\s]", " ", raw)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if not s:
        return ParseResult(raw=raw, reason="no letters in name")
    if not NAME_RE.match(s):
        return ParseResult(raw=raw, reason=f"not name-like: {raw!r}")

    tokens = [t for t in s.split() if t]
    # Strip a leading title; it belongs in the title field, not the name.
    if tokens and normalise_label(tokens[0]).rstrip(".") in TITLE_WORDS:
        tokens = tokens[1:]
    if not tokens:
        return ParseResult(raw=raw, reason="only a title, no name")

    score = 1.0
    if len(raw) != len(s):
        score *= 0.9          # characters were dropped to make it name-like
    if any(len(t) == 1 and not t.endswith(".") for t in tokens):
        score *= 0.9          # bare initials

    if fld.kind == "fullName":
        if len(tokens) < 2:
            # A "full name" with one token is usually a half-read line.
            score *= 0.55
        value = titlecase_name(" ".join(tokens))
    else:
        if len(tokens) > 1 and fld.key in ("firstName", "lastName", "middleName"):
            # Multiple tokens in a single-part name field: keep, but doubt it.
            score *= 0.8
        value = titlecase_name(" ".join(tokens))

    if len(value) < 2:
        return ParseResult(raw=raw, reason="name too short")
    return ParseResult(value=value, parse_score=score, valid=True, raw=raw,
                       candidates=[(value, score)])


def parse_text(text: str, fld: Field) -> ParseResult:
    """Free text, optionally snapped to the field's vocabulary.

    vocabulary is explicitly NOT a list of allowed values -- it only helps read
    messy handwriting -- so a close vocabulary hit replaces the text, a distant
    one does not.
    """
    raw = clean_raw(text)
    if is_blank(raw):
        return ParseResult(raw=raw, reason="blank")

    s = re.sub(r"\s+", " ", raw).strip()
    if len(s) > 500:
        s = s[:500]

    candidates: list[tuple[Any, float]] = []
    score = 0.85  # free text is never fully certain

    if fld.vocabulary:
        norm = normalise_label(s)
        matches = fuzz_process.extract(
            norm, [normalise_label(v) for v in fld.vocabulary],
            scorer=fuzz.WRatio, limit=5,
        )
        vocab_by_norm = {normalise_label(v): v for v in fld.vocabulary}
        ranked = [(vocab_by_norm[f], sc / 100.0) for f, sc, _ in matches if f in vocab_by_norm]
        if ranked:
            best, best_score = ranked[0]
            if best_score >= 0.92:
                value = best if best_score >= 0.97 else titlecase_first(best)
                return ParseResult(value=best, parse_score=min(1.0, best_score),
                                   valid=True, raw=raw, candidates=ranked[:5])
            candidates = ranked[:5]
            if best_score >= 0.80:
                score = 0.7   # close to a known word but not close enough

    value = s
    if not candidates:
        candidates = [(value, score)]
    else:
        candidates = [(value, score)] + candidates[:4]
    return ParseResult(value=value, parse_score=score, valid=True, raw=raw,
                       candidates=candidates)


def titlecase_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

PARSERS: dict[str, Callable[[str, Field], ParseResult]] = {
    "text": parse_text,
    "name": parse_name,
    "fullName": parse_name,
    "date": parse_date,
    "choice": parse_choice,
    "yesNo": parse_yes_no,
    "number": parse_number,
    "money": parse_money,
    "email": parse_email,
    "phone": parse_phone,
    "niNumber": parse_ni_number,
}


def parse_value(text: str, fld: Field) -> ParseResult:
    """Parse `text` as a value for `fld` according to its kind."""
    parser = PARSERS.get(fld.kind)
    if parser is None:
        raise KeyError(f"no parser for kind {fld.kind!r} (field {fld.qualified_key})")
    result = parser(text, fld)
    result.raw = result.raw or clean_raw(text)
    return result


__all__ = ["ParseResult", "parse_value", "PARSERS", "clean_raw", "is_blank",
           "titlecase_name", "normalise_label"]
