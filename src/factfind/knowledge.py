"""Loader for fact-find-knowledge.json.

The knowledge file is the single source of truth for which fields exist, what
they are called on a document, how their answer is read, and what counts as a
valid value. This module turns it into typed objects and a flat field registry
keyed by "<sectionId>.<fieldKey>".
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

from .paths import KNOWLEDGE_FILE

# Kinds the pipeline knows how to parse. A kind in the knowledge file that is
# not in this set is a hard error -- we never silently fall back to raw text,
# because that is exactly how a wrong value gets filled.
SUPPORTED_KINDS = {
    "text", "name", "fullName", "date", "choice", "yesNo",
    "number", "money", "email", "phone", "niNumber",
}

# Lists the CRM supplies at runtime rather than the knowledge file.
# Used for kind 'choice' with choicesFrom. Kept here so the Python side can
# resolve them; the Angular side injects its own equivalents.
CHOICES_FROM: dict[str, list[str]] = {
    "countries": [
        "United Kingdom", "Ireland", "France", "Germany", "Spain", "Italy",
        "Portugal", "Netherlands", "Belgium", "Poland", "Romania", "Bulgaria",
        "Lithuania", "Latvia", "Hungary", "Czechia", "Slovakia", "Greece",
        "Sweden", "Norway", "Denmark", "Finland", "Switzerland", "Austria",
        "United States", "Canada", "Australia", "New Zealand", "South Africa",
        "Nigeria", "Ghana", "Kenya", "Zimbabwe", "India", "Pakistan",
        "Bangladesh", "Sri Lanka", "Nepal", "China", "Hong Kong", "Japan",
        "Philippines", "Malaysia", "Singapore", "Thailand", "Vietnam",
        "United Arab Emirates", "Saudi Arabia", "Turkey", "Iran", "Iraq",
        "Jamaica", "Trinidad and Tobago", "Barbados", "Brazil", "Mexico",
        "Argentina", "Colombia", "Ukraine", "Russia", "Somalia", "Egypt",
        "Morocco", "Algeria", "Afghanistan", "Albania", "Mauritius", "Malta",
        "Cyprus", "Israel", "Other",
    ],
    "nationalities": [
        "British", "Irish", "French", "German", "Spanish", "Italian",
        "Portuguese", "Dutch", "Belgian", "Polish", "Romanian", "Bulgarian",
        "Lithuanian", "Latvian", "Hungarian", "Czech", "Slovak", "Greek",
        "Swedish", "Norwegian", "Danish", "Finnish", "Swiss", "Austrian",
        "American", "Canadian", "Australian", "New Zealander", "South African",
        "Nigerian", "Ghanaian", "Kenyan", "Zimbabwean", "Indian", "Pakistani",
        "Bangladeshi", "Sri Lankan", "Nepalese", "Chinese", "Japanese",
        "Filipino", "Malaysian", "Singaporean", "Thai", "Vietnamese",
        "Emirati", "Saudi", "Turkish", "Iranian", "Iraqi", "Jamaican",
        "Trinidadian", "Barbadian", "Brazilian", "Mexican", "Argentine",
        "Colombian", "Ukrainian", "Russian", "Somali", "Egyptian", "Moroccan",
        "Algerian", "Afghan", "Albanian", "Mauritian", "Maltese", "Cypriot",
        "Israeli", "Other",
    ],
}


def normalise_label(text: Any) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace.

    Used for every label/alias comparison so "D.O.B." and "dob" match.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Choice:
    value: str
    label: str
    aliases: tuple[str, ...] = ()

    def all_surface_forms(self) -> list[str]:
        """Every string a writer might put down for this choice."""
        forms = {self.value, self.label, *self.aliases}
        out, seen = [], set()
        for f in forms:
            n = normalise_label(f)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out


@dataclass
class Field:
    section_id: str
    key: str
    label: str
    kind: str
    aliases: tuple[str, ...] = ()
    choices: tuple[Choice, ...] = ()
    choices_from: str | None = None
    choice_aliases: dict[str, list[str]] = dc_field(default_factory=dict)
    allow_other: bool = False
    vocabulary: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    min_age: int | None = None
    when: str | None = None
    required: bool = False
    only_if: dict[str, Any] | None = None
    years_until_age_from: str | None = None

    @property
    def qualified_key(self) -> str:
        """Globally unique id, e.g. 'personal-details.fullName'."""
        return f"{self.section_id}.{self.key}"

    @property
    def is_sensitive(self) -> bool:
        """Fields the brief requires zero silent errors on."""
        return self.kind in {"niNumber", "email", "phone"}

    def label_forms(self) -> list[str]:
        """Normalised label + aliases, longest first (match greedily)."""
        forms = {normalise_label(self.label)}
        forms.update(normalise_label(a) for a in self.aliases)
        return sorted((f for f in forms if f), key=len, reverse=True)

    def resolved_choices(self) -> list[Choice]:
        """Choices from the file, or from the CRM list named by choicesFrom."""
        if self.choices:
            return list(self.choices)
        if self.choices_from:
            values = CHOICES_FROM.get(self.choices_from, [])
            return [
                Choice(value=v, label=v, aliases=tuple(self.choice_aliases.get(v, [])))
                for v in values
            ]
        return []

    def allowed_values(self) -> list[str]:
        """Canonical output values for a choice field."""
        return [c.value for c in self.resolved_choices()]


@dataclass
class Section:
    id: str
    title: str
    headings: tuple[str, ...]
    fields: tuple[Field, ...]
    is_default: bool = False
    when: str | None = None
    save: dict[str, Any] = dc_field(default_factory=dict)

    @property
    def is_repeating(self) -> bool:
        """Sections saved one row per entry (dependants) repeat on a page."""
        return self.save.get("api") == "dependant"


@dataclass
class Knowledge:
    version: int
    sections: tuple[Section, ...]
    other_headings: tuple[str, ...]
    source_path: Path
    problems: list[str] = dc_field(default_factory=list)

    @property
    def fields(self) -> list[Field]:
        return [f for s in self.sections for f in s.fields]

    @property
    def field_keys(self) -> list[str]:
        return [f.qualified_key for f in self.fields]

    def _index(self) -> dict[str, Field]:
        cached = getattr(self, "_field_index", None)
        if cached is None:
            cached = {f.qualified_key: f for f in self.fields}
            object.__setattr__(self, "_field_index", cached)
        return cached

    def field(self, qualified_key: str) -> Field:
        try:
            return self._index()[qualified_key]
        except KeyError as exc:
            raise KeyError(f"unknown field {qualified_key!r}") from exc

    def section(self, section_id: str) -> Section:
        for s in self.sections:
            if s.id == section_id:
                return s
        raise KeyError(f"unknown section {section_id!r}")

    @property
    def default_section(self) -> Section:
        for s in self.sections:
            if s.is_default:
                return s
        return self.sections[0]

    def heading_index(self) -> list[tuple[str, str]]:
        """(normalised heading stem, section id), longest stem first."""
        pairs = [(normalise_label(h), s.id) for s in self.sections for h in s.headings]
        pairs += [(normalise_label(h), "__other__") for h in self.other_headings]
        return sorted((p for p in pairs if p[0]), key=lambda p: len(p[0]), reverse=True)


class KnowledgeError(ValueError):
    """The knowledge file is missing or internally inconsistent."""


def _validate(kn: Knowledge) -> list[str]:
    """Return a list of problems. Empty list means the file is coherent."""
    problems: list[str] = []
    seen_qualified: set[str] = set()

    if not kn.sections:
        problems.append("no sections defined")

    defaults = [s.id for s in kn.sections if s.is_default]
    if len(defaults) != 1:
        problems.append(
            f"expected exactly one section with default=true, found {len(defaults)}: {defaults}"
        )

    for section in kn.sections:
        if not section.fields:
            problems.append(f"section {section.id!r} has no fields")
        keys_in_section = {f.key for f in section.fields}
        for fld in section.fields:
            if fld.qualified_key in seen_qualified:
                problems.append(f"duplicate field {fld.qualified_key!r}")
            seen_qualified.add(fld.qualified_key)

            if fld.kind not in SUPPORTED_KINDS:
                problems.append(
                    f"{fld.qualified_key}: kind {fld.kind!r} is not supported "
                    f"(known: {sorted(SUPPORTED_KINDS)})"
                )
            if fld.kind == "choice" and not fld.choices and not fld.choices_from:
                problems.append(f"{fld.qualified_key}: kind 'choice' with no choices/choicesFrom")
            if fld.choices_from and fld.choices_from not in CHOICES_FROM:
                problems.append(
                    f"{fld.qualified_key}: choicesFrom {fld.choices_from!r} has no list "
                    f"(add it to knowledge.CHOICES_FROM)"
                )
            if fld.kind != "choice" and (fld.choices or fld.choices_from):
                problems.append(f"{fld.qualified_key}: choices given but kind is {fld.kind!r}")
            if fld.minimum is not None and fld.maximum is not None and fld.minimum > fld.maximum:
                problems.append(f"{fld.qualified_key}: min {fld.minimum} > max {fld.maximum}")

            if fld.only_if:
                dep = fld.only_if.get("key")
                if dep not in keys_in_section:
                    problems.append(
                        f"{fld.qualified_key}: onlyIf points at {dep!r}, "
                        f"not a field of section {section.id!r}"
                    )
                if "equals" not in fld.only_if and "filled" not in fld.only_if:
                    problems.append(
                        f"{fld.qualified_key}: onlyIf has neither 'equals' nor 'filled'"
                    )
            if fld.years_until_age_from and fld.years_until_age_from not in keys_in_section:
                problems.append(
                    f"{fld.qualified_key}: yearsUntilAgeFrom points at "
                    f"{fld.years_until_age_from!r}, not a field of this section"
                )

            # Aliases that collide across fields of the same section make the
            # linker ambiguous; worth knowing about, not fatal.
            for other in section.fields:
                if other.key <= fld.key:
                    continue
                shared = set(fld.label_forms()) & set(other.label_forms())
                if shared:
                    problems.append(
                        f"WARN {section.id}: {fld.key} and {other.key} share label(s) "
                        f"{sorted(shared)} -- the linker cannot tell them apart by label alone"
                    )
    return problems


def _parse_choice(raw: dict[str, Any]) -> Choice:
    return Choice(
        value=raw["value"],
        label=raw.get("label", raw["value"]),
        aliases=tuple(raw.get("aliases", []) or []),
    )


def _parse_field(section_id: str, raw: dict[str, Any]) -> Field:
    return Field(
        section_id=section_id,
        key=raw["key"],
        label=raw.get("label", raw["key"]),
        kind=raw.get("kind", "text"),
        aliases=tuple(raw.get("aliases", []) or []),
        choices=tuple(_parse_choice(c) for c in raw.get("choices", []) or []),
        choices_from=raw.get("choicesFrom"),
        choice_aliases=dict(raw.get("choiceAliases", {}) or {}),
        allow_other=bool(raw.get("allowOther", False)),
        vocabulary=tuple(raw.get("vocabulary", []) or []),
        minimum=raw.get("min"),
        maximum=raw.get("max"),
        min_age=raw.get("minAge"),
        when=raw.get("when"),
        required=bool(raw.get("required", False)),
        only_if=raw.get("onlyIf"),
        years_until_age_from=raw.get("yearsUntilAgeFrom"),
    )


def load_knowledge(path: Path | str | None = None, strict: bool = True) -> Knowledge:
    """Load and validate the knowledge file.

    strict=True raises KnowledgeError on any non-WARN problem, as the brief
    requires: stop and say exactly what is wrong.
    """
    path = Path(path or KNOWLEDGE_FILE)
    if not path.exists():
        raise KnowledgeError(f"knowledge file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnowledgeError(f"knowledge file is not valid JSON: {path}: {exc}") from exc

    sections = []
    for s in raw.get("sections", []):
        sections.append(Section(
            id=s["id"],
            title=s.get("title", s["id"]),
            headings=tuple(s.get("headings", []) or []),
            fields=tuple(_parse_field(s["id"], f) for f in s.get("fields", []) or []),
            is_default=bool(s.get("default", False)),
            when=s.get("when"),
            save=dict(s.get("save", {}) or {}),
        ))

    kn = Knowledge(
        version=int(raw.get("version", 0)),
        sections=tuple(sections),
        other_headings=tuple(raw.get("otherHeadings", []) or []),
        source_path=path,
    )

    problems = _validate(kn)
    fatal = [p for p in problems if not p.startswith("WARN")]
    if fatal and strict:
        raise KnowledgeError("knowledge file is inconsistent:\n  - " + "\n  - ".join(fatal))
    kn.problems = problems
    return kn


__all__ = [
    "Knowledge", "Section", "Field", "Choice", "KnowledgeError",
    "load_knowledge", "normalise_label", "SUPPORTED_KINDS", "CHOICES_FROM",
]
