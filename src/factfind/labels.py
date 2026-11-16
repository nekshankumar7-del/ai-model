"""Label files: read, normalise and compare.

A label file sits next to its image and names the true value of every field
visible on that page. Two shapes are accepted:

  flat   {"personal-details.fullName": "John Smith",
          "dependants[0].name": "Amy Smith"}

  nested {"personal-details": {"fullName": "John Smith"},
          "dependants": [{"name": "Amy Smith"}]}

Both normalise to flat dotted keys, which is what the pipeline and the
evaluator speak. A `_meta` block carries writer / client / layout ids used for
leakage-free splitting; it is never treated as a field.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .knowledge import Knowledge, Field, normalise_label
from .parsers import titlecase_name

META_KEY = "_meta"
INDEXED = re.compile(r"^(?P<section>[^\[\].]+)\[(?P<index>\d+)\]\.(?P<field>.+)$")
PLAIN = re.compile(r"^(?P<section>[^\[\].]+)\.(?P<field>.+)$")


def flatten_labels(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn either shape into flat dotted keys."""
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if key == META_KEY:
            continue
        if isinstance(value, dict) and not INDEXED.match(key) and "." not in key:
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, list) and "." not in key:
            for i, entry in enumerate(value):
                if isinstance(entry, dict):
                    for sub_key, sub_value in entry.items():
                        flat[f"{key}[{i}].{sub_key}"] = sub_value
        else:
            flat[key] = value
    # Drop nulls: an absent field and a null field mean the same thing here.
    return {k: v for k, v in flat.items() if v is not None and v != ""}


def split_key(flat_key: str) -> tuple[str, int | None, str]:
    """'dependants[0].name' -> ('dependants', 0, 'name')."""
    m = INDEXED.match(flat_key)
    if m:
        return m.group("section"), int(m.group("index")), m.group("field")
    m = PLAIN.match(flat_key)
    if m:
        return m.group("section"), None, m.group("field")
    raise ValueError(f"label key is not section.field: {flat_key!r}")


def qualified_key(flat_key: str) -> str:
    """Drop the row index so the key names a Field: 'dependants.name'."""
    section, _index, field = split_key(flat_key)
    return f"{section}.{field}"


def load_label_file(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (flat labels, meta)."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = dict(raw.get(META_KEY, {}) or {})
    return flatten_labels(raw), meta


def validate_against_knowledge(flat: dict[str, Any], kn: Knowledge) -> list[str]:
    """Report label keys that do not exist in the knowledge file."""
    problems = []
    known = set(kn.field_keys)
    for key in flat:
        try:
            qk = qualified_key(key)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if qk not in known:
            problems.append(f"label key {key!r} has no field {qk!r} in the knowledge file")
    return problems


# --------------------------------------------------------------------------
# Comparison -- what counts as "the same value"
# --------------------------------------------------------------------------

def canonical(value: Any, fld: Field) -> Any:
    """Put a value in the one form used for equality checks.

    Both the label and the prediction go through this, so "45000", "45,000"
    and "£45,000.00" compare equal for a money field, while names compare
    case- and punctuation-insensitively but not fuzzily.
    """
    if value is None or value == "":
        return None
    kind = fld.kind

    if kind == "yesNo":
        if isinstance(value, bool):
            return value
        text = normalise_label(value)
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
        return None

    if kind in {"number", "money"}:
        try:
            num = float(str(value).replace(",", "").replace("£", "").strip())
        except (TypeError, ValueError):
            return None
        return round(num, 2)

    if kind == "date":
        text = str(value).strip()
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        # Accept a label written in UK order and canonicalise it.
        m = re.match(r"^(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})$", text)
        if m:
            return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
        return text

    if kind == "choice":
        # Snap a label written as a display label back to its stored value.
        target = normalise_label(value)
        for choice in fld.resolved_choices():
            if target == normalise_label(choice.value) or target == normalise_label(choice.label):
                return choice.value
            if target in choice.all_surface_forms():
                return choice.value
        return str(value).strip()

    if kind == "niNumber":
        return re.sub(r"[\s\-]", "", str(value)).upper()

    if kind == "email":
        return str(value).strip().lower()

    if kind == "phone":
        digits = re.sub(r"\D", "", str(value))
        if digits.startswith("44") and len(digits) == 12:
            digits = "0" + digits[2:]
        return digits

    if kind in {"name", "fullName"}:
        return normalise_label(titlecase_name(str(value)))

    # text
    return normalise_label(value)


def values_match(predicted: Any, truth: Any, fld: Field) -> bool:
    a, b = canonical(predicted, fld), canonical(truth, fld)
    if a is None or b is None:
        return a is None and b is None
    return a == b


def write_label_file(path: Path | str, flat: dict[str, Any],
                     meta: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = dict(flat)
    if meta:
        payload[META_KEY] = meta
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                     default=str), encoding="utf-8")


__all__ = ["flatten_labels", "load_label_file", "write_label_file",
           "qualified_key", "split_key", "canonical", "values_match",
           "validate_against_knowledge", "META_KEY"]
