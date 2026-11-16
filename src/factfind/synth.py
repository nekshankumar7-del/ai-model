"""Synthetic fact-find pages generated from the knowledge file.

Why this exists: the brief allows synthetic data for pre-training and for
developing the pipeline, with final selection on real data. It also means the
whole pipeline can be built, tested and demonstrated before a single real
client note exists -- and it never touches real client data.

Every value is drawn to be valid for its field according to the knowledge
file, so the generator needs no per-field code and keeps working when you add
a field.

A "writer" is a reproducible rendering persona: font, size, slant, baseline
jitter, ink colour, spacing. Writers and clients are both ids in the label
file's _meta, which is what the splitter groups on.
"""
from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .knowledge import Knowledge, Field, Section, normalise_label
from .labels import write_label_file

FONT_DIR = Path(r"C:\Windows\Fonts")
# Open Font Licence handwriting faces fetched by scripts/08_fetch_fonts.py.
# These matter more than they sound: the Windows script fonts are mostly
# print-style, so a model trained only on them never sees joined cursive --
# which is exactly what the real adviser notes are written in.
DOWNLOADED_FONT_DIR = Path(__file__).resolve().parents[2] / "data" / "fonts"

# Fonts that read as handwriting, and the printed fonts used for form labels.
WINDOWS_HAND_FONTS = ["Inkfree.ttf", "comic.ttf", "MISTRAL.TTF", "PRISTINA.TTF",
                      "FRSCRIPT.TTF", "Gabriola.ttf", "JUICE___.TTF",
                      "VINERITC.TTF", "segoepr.ttf", "segoesc.ttf",
                      "comici.ttf", "SCRIPTBL.TTF"]


def _handwriting_fonts() -> list[str]:
    """Every handwriting font available, downloaded ones first."""
    downloaded = []
    if DOWNLOADED_FONT_DIR.exists():
        downloaded = [str(p) for p in sorted(DOWNLOADED_FONT_DIR.glob("*.ttf"))]
    return downloaded + WINDOWS_HAND_FONTS


HAND_FONTS = _handwriting_fonts()
PRINT_FONTS = ["arial.ttf", "calibri.ttf", "georgia.ttf", "times.ttf",
               "verdana.ttf", "tahoma.ttf", "corbel.ttf", "constan.ttf",
               "cambria.ttc", "trebuc.ttf"]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
    "Nancy", "Matthew", "Lisa", "Anthony", "Margaret", "Mark", "Sandra",
    "Aisha", "Omar", "Priya", "Rajesh", "Wei", "Mei", "Olumide", "Chidi",
    "Tomasz", "Agnieszka", "Sofia", "Mateo", "Hannah", "Ethan", "Grace",
    "Oliver", "Amelia", "Harry", "Isla", "Jack", "Ava", "Leo", "Mia",
    "Freya", "Noah", "Ruby", "Arthur", "Ivy", "Theo", "Poppy", "Rory",
]
LAST_NAMES = [
    "Smith", "Jones", "Taylor", "Brown", "Williams", "Wilson", "Johnson",
    "Davies", "Robinson", "Wright", "Thompson", "Evans", "Walker", "White",
    "Roberts", "Green", "Hall", "Wood", "Jackson", "Clarke", "Patel",
    "Khan", "Singh", "Ahmed", "Ali", "Begum", "Hussain", "Kaur", "Rahman",
    "O'Brien", "McDonald", "van der Berg", "Nowak", "Kowalski", "Okafor",
    "Adeyemi", "Nguyen", "Chen", "Ferreira", "Silva", "Murphy", "Kelly",
]
STREETS = ["High Street", "Station Road", "Church Lane", "Victoria Road",
           "Mill Lane", "Kings Road", "Queens Avenue", "Park View"]
COMPANIES = [
    "Barclays", "Tesco Stores Ltd", "NHS Trust", "BT Group", "Sainsbury's",
    "Royal Mail", "Lloyds Banking Group", "Amazon UK", "Vodafone",
    "Greggs plc", "Aviva", "Network Rail", "Hargreaves Lansdown",
    "Jones & Partners LLP", "Bright Futures Ltd", "Northgate Logistics",
]
EMAIL_DOMAINS = ["gmail.com", "hotmail.co.uk", "outlook.com", "yahoo.co.uk",
                 "btinternet.com", "icloud.com", "sky.com", "live.co.uk"]
NOTE_WORDS = ["reviewed", "client", "confirmed", "to follow up", "see policy",
              "discussed", "happy with", "needs review", "sent docs"]


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

@dataclass
class Writer:
    """A reproducible rendering persona. Groups the splitter never breaks."""
    id: str
    hand_font: str
    print_font: str
    base_size: int
    slant: float             # shear, radians-ish
    jitter: float            # per-character baseline wobble in px
    ink: tuple[int, int, int]
    spacing: float           # multiplier on line height
    pressure: float          # stroke thickness proxy

    @staticmethod
    def sample(rng: random.Random, index: int) -> "Writer":
        ink_choices = [(18, 18, 28), (12, 24, 86), (28, 28, 30), (10, 40, 120),
                       (40, 20, 20), (5, 55, 40)]
        return Writer(
            id=f"writer{index:02d}",
            hand_font=rng.choice(HAND_FONTS),
            print_font=rng.choice(PRINT_FONTS),
            base_size=rng.randint(26, 40),
            slant=rng.uniform(-0.18, 0.18),
            jitter=rng.uniform(0.0, 2.2),
            ink=rng.choice(ink_choices),
            spacing=rng.uniform(1.45, 2.1),
            pressure=rng.uniform(0.85, 1.35),
        )

    def font(self, handwritten: bool, size: int | None = None) -> ImageFont.FreeTypeFont:
        name = self.hand_font if handwritten else self.print_font
        return _load_font(name, size or self.base_size)


_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    key = (name, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    for candidate in (FONT_DIR / name, Path(name)):
        try:
            font = ImageFont.truetype(str(candidate), size)
            _FONT_CACHE[key] = font
            return font
        except OSError:
            continue
    # Last resort: any font that loads, so generation never dies on a missing
    # file. Rendering quality drops, which is honest rather than fatal.
    for fallback in PRINT_FONTS + HAND_FONTS:
        try:
            font = ImageFont.truetype(str(FONT_DIR / fallback), size)
            _FONT_CACHE[key] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def available_writers(n: int, seed: int = 7) -> list[Writer]:
    rng = random.Random(seed)
    return [Writer.sample(rng, i) for i in range(n)]


# --------------------------------------------------------------------------
# Value generation -- driven entirely by the knowledge file
# --------------------------------------------------------------------------

def _random_date(rng: random.Random, start_year: int, end_year: int) -> date:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _random_ni(rng: random.Random) -> str:
    first = "ABCEGHJKLMNOPRSTWXYZ"
    second = "ABCEGHJKLMNPRSTWXYZ"
    while True:
        pair = rng.choice(first) + rng.choice(second)
        if pair not in {"BG", "GB", "NK", "KN", "TN", "NT", "ZZ"}:
            break
    return pair + "".join(rng.choice(string.digits) for _ in range(6)) + rng.choice("ABCD")


def _random_phone(rng: random.Random, mobile: bool = True) -> str:
    if mobile:
        return "07" + rng.choice(["7", "8", "9"]) + "".join(rng.choice(string.digits) for _ in range(8))
    area = rng.choice(["0113", "0121", "0131", "0141", "0151", "0161", "0117", "0191"])
    return area + "".join(rng.choice(string.digits) for _ in range(7))


def generate_value(fld: Field, rng: random.Random, context: dict[str, Any]) -> Any:
    """Draw one plausible, valid value for a field, using its kind and rules."""
    kind = fld.kind
    key = fld.key

    if kind == "choice":
        choices = fld.allowed_values()
        if not choices:
            return None
        # Keep "Other" rare so the class balance is not silly.
        pool = [c for c in choices if c != "Other"] or choices
        return rng.choice(pool) if rng.random() > 0.03 else rng.choice(choices)

    if kind == "yesNo":
        return rng.random() < 0.5

    if kind == "date":
        if fld.min_age:
            today = date.today()
            age = rng.randint(max(fld.min_age, 18), 72)
            born = today.replace(year=today.year - age) - timedelta(days=rng.randint(0, 364))
            return born.isoformat()
        if "dateOfBirth" in key:
            # Dependant: a child, mostly.
            today = date.today()
            age = rng.randint(0, 24)
            return (today.replace(year=today.year - age) - timedelta(days=rng.randint(0, 364))).isoformat()
        if "expiry" in key.lower() or "validto" in key.lower().replace(" ", ""):
            return _random_date(rng, date.today().year + 1, date.today().year + 9).isoformat()
        if "issue" in key.lower() or "validfrom" in key.lower():
            return _random_date(rng, date.today().year - 9, date.today().year - 1).isoformat()
        if "start" in key.lower():
            return _random_date(rng, 2005, date.today().year).isoformat()
        return _random_date(rng, 1990, date.today().year).isoformat()

    if kind == "number":
        lo = int(fld.minimum) if fld.minimum is not None else 0
        hi = int(fld.maximum) if fld.maximum is not None else 100
        if key == "retirementAge":
            return rng.choice([55, 57, 60, 62, 65, 66, 67, 68, 70])
        if key == "alcoholUnitsPerWeek":
            return rng.choice([0, 0, 2, 4, 6, 8, 10, 12, 14, 18, 21, 28])
        if key == "dependencyTermYears":
            return rng.randint(1, 21)
        hi = min(hi, lo + 100)
        return rng.randint(lo, hi)

    if kind == "money":
        if "salary" in key.lower() or "income" in key.lower():
            return float(rng.randrange(18_000, 140_000, 500))
        if key in {"property", "mortgageMainResidence", "mortgageOtherProperty"}:
            return float(rng.randrange(90_000, 850_000, 5_000))
        if key in {"pensionFunds", "cashDeposits", "isa", "shares",
                   "investmentBonds", "businessAssets", "allOtherAssets",
                   "totalValueOfBusiness", "businessTurnover"}:
            return float(rng.randrange(0, 400_000, 1_000))
        if "bonus" in key.lower() or "commission" in key.lower() or "overtime" in key.lower():
            return float(rng.randrange(500, 20_000, 250))
        # Monthly household amounts.
        return float(rng.randrange(20, 1_400, 5))

    if kind == "email":
        first = context.get("firstName", rng.choice(FIRST_NAMES))
        last = context.get("lastName", rng.choice(LAST_NAMES))
        local = rng.choice([
            f"{first}.{last}", f"{first[0]}{last}", f"{first}{rng.randint(1, 99)}",
            f"{first}_{last}",
        ])
        local = "".join(c for c in local.lower() if c.isalnum() or c in "._")
        return f"{local}@{rng.choice(EMAIL_DOMAINS)}"

    if kind == "phone":
        return _random_phone(rng, mobile="mobile" in key.lower())

    if kind == "niNumber":
        return _random_ni(rng)

    if kind in {"name", "fullName"}:
        if key == "firstName":
            return context.get("firstName") or rng.choice(FIRST_NAMES)
        if key == "lastName":
            return context.get("lastName") or rng.choice(LAST_NAMES)
        if key == "middleName":
            return rng.choice(FIRST_NAMES)
        if key == "fullName":
            return f"{context.get('firstName', rng.choice(FIRST_NAMES))} " \
                   f"{context.get('lastName', rng.choice(LAST_NAMES))}"
        return f"{rng.choice(FIRST_NAMES)} {context.get('lastName', rng.choice(LAST_NAMES))}"

    # text
    if fld.vocabulary:
        return rng.choice(list(fld.vocabulary))
    if "company" in key.lower() or "employer" in key.lower():
        return rng.choice(COMPANIES)
    if "address" in key.lower():
        return f"{rng.randint(1, 180)} {rng.choice(STREETS)}"
    if "detail" in key.lower() or "information" in key.lower() or "note" in key.lower():
        return " ".join(rng.sample(NOTE_WORDS, rng.randint(2, 4)))
    if "height" in key.lower():
        return f"{rng.randint(5, 6)}ft {rng.randint(0, 11)}in"
    if "number" in key.lower():
        return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(9))
    return rng.choice(NOTE_WORDS)


def _only_if_satisfied(fld: Field, values: dict[str, Any]) -> bool:
    if not fld.only_if:
        return True
    dep_key = fld.only_if.get("key")
    dep_value = values.get(dep_key)
    if "filled" in fld.only_if:
        return (dep_value not in (None, "")) == bool(fld.only_if["filled"])
    return dep_value == fld.only_if.get("equals")


def generate_section_values(section: Section, rng: random.Random,
                            context: dict[str, Any], fill_rate: float) -> dict[str, Any]:
    """Fill a section. Respects onlyIf, and leaves fields out at random so the
    pipeline learns that most pages are partial."""
    values: dict[str, Any] = {}
    # Required fields and fields others depend on come first.
    ordered = sorted(section.fields, key=lambda f: (f.only_if is not None, not f.required))
    for fld in ordered:
        if fld.only_if is not None and not _only_if_satisfied(fld, values):
            continue
        keep = fld.required or rng.random() < fill_rate
        if not keep:
            continue
        value = generate_value(fld, rng, context)
        if value is None:
            continue
        values[fld.key] = value
        if fld.key in {"firstName", "lastName"}:
            context[fld.key] = value
    # Second pass: a dependent field whose condition only became true now.
    for fld in section.fields:
        if fld.key in values or fld.only_if is None:
            continue
        if _only_if_satisfied(fld, values) and fld.required:
            value = generate_value(fld, rng, context)
            if value is not None:
                values[fld.key] = value
    return values


@dataclass
class ClientRecord:
    client_id: str
    case_type: str                       # 'mortgage' | 'protection' | 'general'
    sections: dict[str, Any]             # section id -> values (or list for dependants)

    def flat_labels(self) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for section_id, payload in self.sections.items():
            if isinstance(payload, list):
                for i, row in enumerate(payload):
                    for k, v in row.items():
                        flat[f"{section_id}[{i}].{k}"] = v
            else:
                for k, v in payload.items():
                    flat[f"{section_id}.{k}"] = v
        return flat


def make_client(kn: Knowledge, rng: random.Random, index: int,
                sections_per_page: int = 3, fill_rate: float = 0.62) -> ClientRecord:
    """Build one client's answers across a few sections."""
    case_type = rng.choice(["mortgage", "protection", "general"])
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    context = {"firstName": first, "lastName": last}

    usable = [s for s in kn.sections if s.when is None or s.when == case_type]
    # Personal details appear on nearly every note.
    personal = kn.default_section
    chosen = [personal]
    others = [s for s in usable if s.id != personal.id]
    rng.shuffle(others)
    chosen += others[: max(0, sections_per_page - 1)]

    sections: dict[str, Any] = {}
    for section in chosen:
        if section.is_repeating:
            rows = []
            for _ in range(rng.randint(1, 3)):
                row_ctx = dict(context)
                rows.append(generate_section_values(section, rng, row_ctx, fill_rate=0.95))
            sections[section.id] = rows
        else:
            fields = [f for f in section.fields if f.when is None or f.when == case_type]
            sub = Section(id=section.id, title=section.title, headings=section.headings,
                          fields=tuple(fields), is_default=section.is_default,
                          when=section.when, save=section.save)
            sections[section.id] = generate_section_values(sub, rng, context, fill_rate)

    return ClientRecord(client_id=f"client{index:04d}", case_type=case_type,
                        sections=sections)


# --------------------------------------------------------------------------
# Surface forms: how a value is actually written on paper
# --------------------------------------------------------------------------

def surface_form(fld: Field, value: Any, rng: random.Random) -> str:
    """Render a stored value the way a person would write it."""
    kind = fld.kind
    if kind == "yesNo":
        return rng.choice(["Yes", "Y", "yes", "YES"]) if value else \
               rng.choice(["No", "N", "no", "NO"])
    if kind == "date":
        try:
            y, m, d = (int(p) for p in str(value).split("-"))
        except ValueError:
            return str(value)
        style = rng.random()
        if style < 0.45:
            return f"{d:02d}/{m:02d}/{y}"
        if style < 0.60:
            return f"{d}/{m}/{str(y)[2:]}"
        if style < 0.72:
            return f"{d:02d}-{m:02d}-{y}"
        if style < 0.86:
            import calendar
            return f"{d} {calendar.month_abbr[m]} {y}"
        import calendar
        return f"{d}{_ordinal(d)} {calendar.month_name[m]} {y}"
    if kind == "money":
        num = float(value)
        style = rng.random()
        if num >= 1000 and style < 0.22:
            return f"{num / 1000:g}k"
        body = f"{num:,.0f}" if num == int(num) else f"{num:,.2f}"
        if style < 0.55:
            return "£" + body
        if style < 0.8:
            return body
        return body.replace(",", "")
    if kind == "number":
        return str(value)
    if kind == "choice":
        for choice in fld.resolved_choices():
            if choice.value == value:
                options = [choice.label, choice.value, *choice.aliases]
                options = [o for o in options if o]
                # Favour the human-readable label over the stored value.
                weights = [3 if o == choice.label else 1 for o in options]
                return rng.choices(options, weights=weights, k=1)[0]
        return str(value)
    if kind == "phone":
        s = str(value)
        style = rng.random()
        if style < 0.35:
            return f"{s[:5]} {s[5:]}"
        if style < 0.5:
            return f"{s[:4]} {s[4:7]} {s[7:]}"
        if style < 0.6:
            return "+44 " + s[1:]
        return s
    if kind == "niNumber":
        s = str(value)
        style = rng.random()
        if style < 0.4:
            return f"{s[:2]} {s[2:4]} {s[4:6]} {s[6:8]} {s[8:]}"
        if style < 0.55:
            return f"{s[:2]} {s[2:8]} {s[8:]}"
        return s
    return str(value)


def _ordinal(d: int) -> str:
    if 11 <= d <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")


def label_form(fld: Field, rng: random.Random) -> str:
    """Pick a label a writer would put in front of the value."""
    options = [fld.label, *fld.aliases]
    weights = [4] + [1] * len(fld.aliases)
    chosen = rng.choices(options, weights=weights, k=1)[0]
    style = rng.random()
    if style < 0.12:
        return chosen.upper()
    if style < 0.3:
        return chosen.title()
    if style < 0.4:
        return chosen.lower()
    return chosen


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

PAGE_W, PAGE_H = 1240, 1754        # A4 at 150 dpi


def _paper(rng: random.Random, ruled: bool) -> Image.Image:
    base = rng.randint(238, 253)
    tint = rng.randint(-4, 6)
    img = Image.new("RGB", (PAGE_W, PAGE_H),
                    (base, base, min(255, base + tint)))
    draw = ImageDraw.Draw(img)
    if ruled:
        gap = rng.randint(46, 62)
        colour = (rng.randint(170, 205), rng.randint(185, 215), rng.randint(205, 235))
        y = rng.randint(90, 140)
        while y < PAGE_H - 40:
            draw.line([(60, y), (PAGE_W - 60, y)], fill=colour, width=1)
            y += gap
        if rng.random() < 0.5:
            mx = rng.randint(90, 130)
            draw.line([(mx, 40), (mx, PAGE_H - 40)],
                      fill=(rng.randint(205, 230), 160, 160), width=1)
    return img


def _draw_hand_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                    font: ImageFont.FreeTypeFont, writer: Writer,
                    rng: random.Random, ink: tuple[int, int, int] | None = None) -> int:
    """Draw text character by character with wobble, returning the end x."""
    x, y = xy
    colour = ink or writer.ink
    for ch in text:
        dy = int(rng.uniform(-writer.jitter, writer.jitter))
        dx = int(rng.uniform(-0.6, 0.9))
        jcolour = tuple(
            max(0, min(255, c + rng.randint(-14, 14))) for c in colour
        )
        draw.text((x + dx, y + dy), ch, font=font, fill=jcolour)
        advance = draw.textlength(ch, font=font)
        x += advance + rng.uniform(-0.4, 1.0) * writer.pressure
    return int(x)


@dataclass
class RenderedPage:
    image: np.ndarray
    labels: dict[str, Any]
    meta: dict[str, Any]


def render_record(record: ClientRecord, writer: Writer, kn: Knowledge,
                  rng: random.Random, layout: str | None = None) -> RenderedPage:
    """Render one client's answers as a page in one of four layouts."""
    layout = layout or rng.choice(["label_value", "two_column", "narrative", "printed_form"])
    ruled = layout in {"label_value", "narrative"} and rng.random() < 0.55
    img = _paper(rng, ruled)
    draw = ImageDraw.Draw(img)

    handwritten = layout != "printed_form"
    body_font = writer.font(handwritten)
    label_font = writer.font(handwritten if layout != "printed_form" else False,
                             size=writer.base_size - 2)
    head_font = writer.font(False, size=writer.base_size + 8)

    emitted: dict[str, Any] = {}
    y = rng.randint(60, 110)
    left = rng.randint(70, 140)
    line_h = int(writer.base_size * writer.spacing)

    def new_page_guard(needed: int) -> bool:
        return y + needed < PAGE_H - 80

    for section_id, payload in record.sections.items():
        section = kn.section(section_id)
        if not new_page_guard(line_h * 3):
            break

        # Heading
        if rng.random() < 0.9:
            heading = rng.choice(list(section.headings) + [section.title])
            heading = heading.title() if rng.random() < 0.7 else heading.upper()
            draw.text((left, y), heading, font=head_font, fill=writer.ink)
            if rng.random() < 0.45:
                width = int(draw.textlength(heading, font=head_font))
                draw.line([(left, y + head_font.size + 6),
                           (left + width, y + head_font.size + 6)],
                          fill=writer.ink, width=2)
            y += int(line_h * 1.4)

        rows = payload if isinstance(payload, list) else [payload]
        for row_index, values in enumerate(rows):
            if isinstance(payload, list) and len(rows) > 1:
                tag = f"{section.title[:-1] if section.title.endswith('s') else section.title} {row_index + 1}"
                if new_page_guard(line_h):
                    draw.text((left, y), tag, font=label_font, fill=writer.ink)
                    y += line_h

            items = list(values.items())
            if layout == "narrative":
                emitted.update(_render_narrative(
                    draw, items, section, kn, writer, rng, left, y, body_font,
                    section_id, row_index if isinstance(payload, list) else None,
                ))
                y += line_h * max(2, math.ceil(len(items) / 2))
                continue

            for key, value in items:
                if not new_page_guard(line_h * 2):
                    break
                try:
                    fld = kn.field(f"{section_id}.{key}")
                except KeyError:
                    continue
                # Real adviser notes separate label from value with a dash far
                # more often than a colon, so the generated pages match that.
                separator = rng.choices(
                    [" -", ":", "", " –", " -"],
                    weights=[40, 25, 12, 8, 15], k=1,
                )[0]
                label_text = label_form(fld, rng) + separator
                value_text = surface_form(fld, value, rng)

                if layout == "two_column":
                    col = row_index % 1
                    x_label = left + col * (PAGE_W // 2 - 80)
                else:
                    x_label = left

                draw.text((x_label, y), label_text, font=label_font,
                          fill=writer.ink if handwritten else (30, 30, 36))
                label_w = draw.textlength(label_text, font=label_font)
                x_value = x_label + label_w + rng.randint(14, 48)

                if layout == "printed_form":
                    # Printed label, handwritten answer on a rule.
                    box_w = rng.randint(260, 420)
                    draw.line([(x_value - 6, y + body_font.size + 8),
                               (x_value + box_w, y + body_font.size + 8)],
                              fill=(120, 120, 140), width=1)
                    hand = writer.font(True)
                    _draw_hand_text(draw, (int(x_value), y - 2), value_text,
                                    hand, writer, rng)
                else:
                    _draw_hand_text(draw, (int(x_value), y), value_text,
                                    body_font, writer, rng)

                flat_key = (f"{section_id}[{row_index}].{key}"
                            if isinstance(payload, list) else f"{section_id}.{key}")
                emitted[flat_key] = value
                y += line_h
            y += int(line_h * 0.4)
        y += int(line_h * 0.5)

    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    meta = {
        "writer": writer.id, "client": record.client_id, "layout": layout,
        "case_type": record.case_type, "synthetic": True,
        "font": writer.hand_font if handwritten else writer.print_font,
    }
    return RenderedPage(image=arr, labels=emitted, meta=meta)


def _render_narrative(draw, items, section, kn, writer, rng, left, y,
                      font, section_id, row_index) -> dict[str, Any]:
    """Write the answers as prose: 'John Smith, dob 14/03/1982, is married.'"""
    emitted: dict[str, Any] = {}
    x = left
    line_h = int(writer.base_size * writer.spacing)
    cur_y = y
    for key, value in items:
        try:
            fld = kn.field(f"{section_id}.{key}")
        except KeyError:
            continue
        connector = rng.choice([", ", ". ", "; ", " and "])
        phrase = f"{label_form(fld, rng).lower()} {surface_form(fld, value, rng)}"
        width = draw.textlength(phrase + connector, font=font)
        if x + width > PAGE_W - 80:
            x = left
            cur_y += line_h
            if cur_y > PAGE_H - 100:
                break
        x = _draw_hand_text(draw, (x, cur_y), phrase + connector, font, writer, rng)
        flat_key = (f"{section_id}[{row_index}].{key}" if row_index is not None
                    else f"{section_id}.{key}")
        emitted[flat_key] = value
    return emitted


# --------------------------------------------------------------------------
# Augmentation (train split only -- callers enforce that)
# --------------------------------------------------------------------------

def augment(img: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    """Photo-of-a-page degradations: perspective, blur, shadow, noise, JPEG."""
    out = img.copy()
    h, w = out.shape[:2]

    if rng.random() < 0.6 * strength:
        m = int(min(h, w) * rng.uniform(0.004, 0.028) * strength)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [rng.randint(0, m), rng.randint(0, m)],
            [w - rng.randint(0, m), rng.randint(0, m)],
            [w - rng.randint(0, m), h - rng.randint(0, m)],
            [rng.randint(0, m), h - rng.randint(0, m)],
        ])
        out = cv2.warpPerspective(out, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.35 * strength:
        angle = rng.uniform(-3.5, 3.5) * strength
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.5 * strength:
        k = rng.choice([3, 3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.4, 1.3))

    if rng.random() < 0.5 * strength:
        # Soft directional shadow, like a hand or phone over the page.
        shadow = np.zeros((h, w), np.float32)
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        radius = rng.randint(int(w * 0.3), int(w * 0.95))
        cv2.circle(shadow, (cx, cy), radius, 1.0, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), radius * 0.35)
        factor = 1.0 - shadow * rng.uniform(0.12, 0.42) * strength
        out = np.clip(out.astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)

    if rng.random() < 0.45 * strength:
        gain = rng.uniform(0.80, 1.18)
        bias = rng.uniform(-22, 16)
        out = np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    if rng.random() < 0.55 * strength:
        sigma = rng.uniform(2, 11) * strength
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.5 * strength:
        quality = int(rng.uniform(32, 85))
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if rng.random() < 0.25 * strength:
        scale = rng.uniform(0.62, 0.92)
        small = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------

def generate_dataset(kn: Knowledge, out_dir: Path | str, n_pages: int = 300,
                     n_writers: int = 12, seed: int = 11,
                     augment_fraction: float = 0.65) -> list[dict[str, Any]]:
    """Write n_pages images + label files. Returns the manifest rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    writers = available_writers(n_writers, seed=seed)

    manifest: list[dict[str, Any]] = []
    # A client is written up by one writer, which is what makes writer and
    # client sensible grouping keys that mostly coincide -- as in real notes.
    n_clients = max(1, int(n_pages * 0.8))
    clients = [make_client(kn, rng, i) for i in range(n_clients)]

    for i in range(n_pages):
        record = clients[i % n_clients]
        writer = writers[hash(record.client_id) % len(writers)]
        page = render_record(record, writer, kn, rng)
        if not page.labels:
            continue
        image = page.image
        if rng.random() < augment_fraction:
            image = augment(image, rng, strength=rng.uniform(0.5, 1.2))
            page.meta["augmented"] = True
        else:
            page.meta["augmented"] = False

        stem = f"{record.client_id}_{writer.id}_p{i:04d}"
        img_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(img_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        write_label_file(out_dir / f"{stem}.json", page.labels, page.meta)
        manifest.append({"image": str(img_path), "label": str(out_dir / f"{stem}.json"),
                         **page.meta, "n_fields": len(page.labels)})
    return manifest


__all__ = ["Writer", "ClientRecord", "make_client", "render_record", "augment",
           "generate_dataset", "available_writers", "surface_form", "label_form",
           "generate_value"]
