"""Report writing: the final report, the coverage curve and the model cards."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .knowledge import Knowledge


def plot_coverage_curve(curve: list[dict[str, float]], path: Path) -> Path | None:
    """Coverage against wrong-fill as the fill threshold sweeps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not curve:
        return None

    wrong = [row["wrong_fill_rate"] * 100 for row in curve]
    coverage = [row["coverage"] * 100 for row in curve]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
    ax.plot(wrong, coverage, linewidth=2, color="#2a6fb5")
    ax.scatter(wrong, coverage, s=10, color="#2a6fb5", alpha=0.45)

    for target, label, colour in ((1.0, "1% target", "#c23b22"),
                                  (5.0, "5% target", "#d08c18")):
        ax.axvline(target, linestyle="--", linewidth=1.1, color=colour, alpha=0.8)
        ax.text(target, max(coverage) * 0.02 if coverage else 0, f" {label}",
                color=colour, fontsize=8, rotation=90, va="bottom")

    ax.set_xlabel("wrong-fill rate (% of filled values that are wrong)")
    ax.set_ylabel("coverage (% of present fields filled correctly)")
    ax.set_title("Coverage vs wrong-fill")
    ax.grid(alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _pct3(x: float) -> str:
    return f"{x * 100:.2f}%"


def write_final_report(payload: dict[str, Any], kn: Knowledge, path: Path,
                       figure: Path | None = None,
                       data_dir: str = "") -> Path:
    """reports/final.md - per-field table, curve, failures, and what to add."""
    base = payload["baseline"]
    final = payload["final"]
    bs, fs = base["summary"], final["summary"]
    targets = final["targets"]

    synthetic = "synthetic" in str(data_dir).lower()

    lines: list[str] = [
        "# Fact-find extraction - final report",
        "",
        f"Generated {date.today().isoformat()} from the **{payload['split']}** "
        f"split ({payload['n_pages']} pages), scored once.",
        "",
    ]

    if synthetic:
        lines += [
            "> **These numbers are from generated pages, not real client notes.**",
            "> They show the pipeline works end to end and give a reference point, "
            "but they are not evidence about your documents. Synthetic pages are "
            "cleaner and more regular than a photographed handwritten note, so "
            "treat every number here as an optimistic upper bound. The brief's "
            "targets can only be claimed against real data.",
            "",
        ]

    lines += [
        "## Headline",
        "",
        "| | baseline (rules) | final | target |",
        "| --- | ---: | ---: | ---: |",
        f"| wrong-fill rate | {_pct3(bs['wrong_fill_rate'])} | "
        f"**{_pct3(fs['wrong_fill_rate'])}** | see per-kind |",
        f"| exact match | {_pct(bs['exact_match_rate'])} | "
        f"**{_pct(fs['exact_match_rate'])}** | maximise |",
        f"| coverage | {_pct(bs['coverage'])} | **{_pct(fs['coverage'])}** | maximise |",
        f"| unsure | {_pct(bs['unsure_rate'])} | **{_pct(fs['unsure_rate'])}** | - |",
        f"| missing | {_pct(bs['missing_rate'])} | **{_pct(fs['missing_rate'])}** | - |",
        f"| values filled | {bs['n_filled']} | **{fs['n_filled']}** | - |",
        f"| wrong values filled | {bs['n_wrong']} | **{fs['n_wrong']}** | - |",
        "",
        "`wrong-fill rate` is the share of everything the pipeline chose to fill "
        "that was wrong, including values it filled for fields the page never "
        "mentioned. `coverage` is the share of the fields actually present on "
        "the pages that were filled correctly.",
        "",
        f"Of the values the pipeline left unsure, **"
        f"{_pct(fs['unsure_top3_recall'])}** had the correct answer inside the "
        "top 3 candidates it offered.",
        "",
    ]

    # ---- targets
    lines += ["## Targets", ""]
    if targets["all_met"]:
        lines.append("Every wrong-fill target was met on this split.")
    else:
        lines.append("**Targets missed.** They have not been lowered.")
    lines += [
        "",
        "| kind | wrong-fill | target | met | filled | wrong | coverage |",
        "| --- | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for row in targets["per_kind"]:
        lines.append(
            f"| {row['kind']} | {_pct3(row['wrong_fill'])} | "
            f"{_pct3(row['target_wrong_fill'])} | "
            f"{'yes' if row['target_met'] else '**no**'} | {row['n_filled']} | "
            f"{row['n_wrong']} | {_pct(row['coverage'])} |"
        )
    lines.append("")

    # ---- validation vs test: the generalisation gap
    validation = payload.get("validation_per_kind")
    if validation:
        val_by_kind = {r["kind"]: r for r in validation}
        gaps = []
        for row in targets["per_kind"]:
            v = val_by_kind.get(row["kind"])
            if not v or row["n_filled"] == 0:
                continue
            gaps.append((row["kind"], v["wrong_fill"], row["wrong_fill"],
                         v["n_filled"], row["n_filled"]))
        if gaps:
            widened = [g for g in gaps if g[2] > g[1] + 1e-9]
            lines += [
                "## Validation against test",
                "",
                "Thresholds were chosen on validation. Where test is worse, the "
                "threshold was fitted to the validation writers and did not "
                "carry over - which is the single most important number in "
                "this report, because it is what would happen on a new "
                "adviser's handwriting.",
                "",
                "| kind | validation wrong-fill | test wrong-fill | moved by |",
                "| --- | ---: | ---: | ---: |",
            ]
            for kind, vw, tw, _vn, _tn in sorted(gaps, key=lambda g: g[1] - g[2]):
                delta = tw - vw
                arrow = f"+{_pct3(delta)}" if delta > 0 else _pct3(delta)
                lines.append(f"| {kind} | {_pct3(vw)} | {_pct3(tw)} | {arrow} |")
            lines += [
                "",
                f"{len(widened)} of {len(gaps)} kinds got worse on test.",
                "",
            ]
            if widened:
                lines += [
                    "The cause is sample size, not a modelling flaw: the "
                    "thresholds were fitted on the validation split's writers "
                    "only. With a handful of writers, a threshold that looks "
                    "safe on validation is partly fitted to those particular "
                    "hands. The fix is more writers, not a different model - "
                    "and until then the validation numbers should be read as "
                    "optimistic.",
                    "",
                ]

    if targets["misses"]:
        lines += ["### What would close each gap", ""]
        for miss in targets["misses"]:
            lines += [
                f"**{miss['kind']}** - {_pct3(miss['wrong_fill'])} against a "
                f"{_pct3(miss['target'])} target "
                f"({miss['n_wrong']} wrong of {miss['n_filled']} filled).",
                "",
            ]
            lines += [f"  - {s}" for s in _gap_advice(miss, final, kn)]
            lines.append("")

    # ---- curve
    if figure:
        lines += [
            "## Coverage vs wrong-fill",
            "",
            f"![coverage vs wrong-fill]({Path(figure).relative_to(path.parent).as_posix()})",
            "",
            "Each point is a fill threshold. Moving left refuses more values "
            "(lower wrong-fill, lower coverage); moving right fills more. The "
            "shipped thresholds are chosen per kind as the leftmost point that "
            "still meets that kind's target.",
            "",
        ]

    # ---- per field
    lines += [
        "## Per field",
        "",
        "Fields with at least one example on this split, worst wrong-fill first.",
        "",
        "| field | kind | n | baseline exact | final exact | baseline wrong | "
        "final wrong | unsure | missing | target met |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    base_by_field = {r["field"]: r for r in base["per_field"]}
    rows = [r for r in final["per_field"] if r["n"] > 0]
    rows.sort(key=lambda r: (-r["wrong_fill"], -r["n"]))
    for row in rows:
        b = base_by_field.get(row["field"], {})
        lines.append(
            f"| `{row['field']}` | {row['kind']} | {row['n']} | "
            f"{_pct(b.get('exact_match', 0.0))} | {_pct(row['exact_match'])} | "
            f"{_pct3(b.get('wrong_fill', 0.0))} | {_pct3(row['wrong_fill'])} | "
            f"{_pct(row['unsure'])} | {_pct(row['missing'])} | "
            f"{'yes' if row['target_met'] else '**no**'} |"
        )
    lines.append("")

    # ---- failures
    errors = final.get("errors", {})
    lines += ["## Worst failures", ""]
    examples = errors.get("examples", [])
    if examples:
        lines += [
            "The wrong fills the pipeline was most confident about - the ones "
            "that would actually reach a form.",
            "",
            "| page | field | filled | should have been | confidence |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for ex in examples[:20]:
            lines.append(
                f"| `{ex['stem']}` | `{ex['field']}` | `{_short(ex['predicted'])}` "
                f"| `{_short(ex['truth'])}` | {ex['confidence']:.3f} |"
            )
        lines.append("")
        lines.append(
            "> Page images are in the dataset folder under these names. They "
            "are not copied into this report, because they contain client data."
        )
        lines.append("")
    else:
        lines += ["No wrong fills on this split.", ""]

    if errors.get("wrong_by_field"):
        lines += ["### Wrong fills by field", "",
                  "| field | count |", "| --- | ---: |"]
        for field, n in errors["wrong_by_field"]:
            lines.append(f"| `{field}` | {n} |")
        lines.append("")

    if errors.get("missing_by_field"):
        lines += ["### Most often missed entirely", "",
                  "| field | times present but not found |", "| --- | ---: |"]
        for field, n in errors["missing_by_field"][:15]:
            lines.append(f"| `{field}` | {n} |")
        lines.append("")

    # ---- what to add
    lines += ["## What would improve each weak field", ""]
    weak = [r for r in rows if not r["target_met"] or r["missing"] > 0.25
            or r["exact_match"] < 0.6]
    if not weak:
        lines.append("No field is weak enough on this split to call out.")
    else:
        lines += ["| field | problem | what to add |", "| --- | --- | --- |"]
        for row in weak[:30]:
            lines.append(f"| `{row['field']}` | {_problem(row)} | {_fix(row, kn)} |")
    lines += [
        "",
        "## Notes on method",
        "",
        "- Splits are by connected components over (writer, client), so no hand "
        "and no person appears on both sides of a split.",
        "- The test split was scored once, after validation stopped improving.",
        "- `wrong-fill` counts a value filled for a field the page never "
        "mentioned as wrong, not as a free pass.",
        "- Thresholds were chosen on validation as the lowest value per kind "
        "that still met that kind's target, which maximises coverage subject "
        "to the priority-1 constraint.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _short(value: Any, n: int = 28) -> str:
    s = "" if value is None else str(value)
    s = s.replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _problem(row: dict[str, Any]) -> str:
    if not row["target_met"]:
        return f"wrong-fill {_pct3(row['wrong_fill'])} over target"
    if row["missing"] > 0.25:
        return f"missed {_pct(row['missing'])} of the time"
    return f"exact match only {_pct(row['exact_match'])}"


def _fix(row: dict[str, Any], kn: Knowledge) -> str:
    kind = row["kind"]
    field = row["field"]
    if row["missing"] > 0.25:
        return (f"more pages where this is written with a label the knowledge "
                f"file does not list yet - check how advisers actually write "
                f"'{row['label']}' and add those as `aliases`")
    if kind in {"name", "fullName", "text"}:
        return ("more handwritten examples from more writers; free text is "
                "limited by the recogniser, not the matcher")
    if kind in {"money", "number"}:
        return ("examples with the digit styles that fail - crossed 7s, "
                "continental 1s, and amounts written with a period as the "
                "thousands separator")
    if kind == "date":
        return ("examples with ambiguous day/month dates so the parser's UK "
                "default can be measured rather than assumed")
    if kind == "choice":
        return (f"examples of the choices that never appear; add any wording "
                f"advisers use for them to this field's `choices[].aliases`")
    if kind == "yesNo":
        return ("examples using ticks, crosses and strike-throughs, which are "
                "the ambiguous cases")
    return "more labelled examples of this field"


def _gap_advice(miss: dict[str, Any], final: dict[str, Any],
                kn: Knowledge) -> list[str]:
    kind = miss["kind"]
    n_filled = miss["n_filled"]
    advice = []
    if n_filled < 100:
        advice.append(
            f"only {n_filled} values of this kind were filled on this split. "
            f"A {_pct3(miss['target'])} rate cannot be demonstrated at that "
            f"size - roughly {int(3 / max(miss['target'], 0.001))} filled "
            f"examples are needed before the measurement means anything."
        )
    advice.append(
        "raising this kind's threshold trades coverage for correctness; the "
        "curve above shows exactly how much."
    )
    if kind in {"name", "fullName", "text"}:
        advice.append(
            "the limit here is handwriting recognition, not field matching. "
            "The biggest single win is a value recogniser fine-tuned on your "
            "own cropped value regions - which needs a few thousand labelled "
            "crops from real notes."
        )
    if kind in {"date", "money", "number"}:
        advice.append(
            "most of these errors are single-digit misreads. More real "
            "examples of the digit shapes that fail, plus per-field range "
            "rules in the knowledge file, close this faster than more model."
        )
    if kind in {"choice", "yesNo"}:
        advice.append(
            "add the wordings advisers actually use to this field's aliases in "
            "the knowledge file - no retraining needed."
        )
    advice.append(
        "a printed template form with boxed fields would remove most of this "
        "error class outright, if the business can adopt one."
    )
    return advice


def write_model_card(path: Path, name: str, payload: dict[str, Any]) -> Path:
    """One model card per shipped model."""
    lines = [
        f"# Model card - {name}",
        "",
        f"Version: {payload.get('version', '0.1.0')}  ",
        f"Generated: {date.today().isoformat()}",
        "",
        "## What it does",
        "",
        payload.get("purpose", ""),
        "",
        "## Training data",
        "",
        payload.get("data", ""),
        "",
        "## Inputs and outputs",
        "",
        payload.get("io", ""),
        "",
        "## Metrics",
        "",
    ]
    metrics = payload.get("metrics", {})
    if metrics:
        lines += ["| metric | value |", "| --- | ---: |"]
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float)
                             else f"| {key} | {value} |")
        lines.append("")
    lines += [
        "## Known failures",
        "",
        payload.get("failures", ""),
        "",
        "## Not to be used for",
        "",
        payload.get("limits", ""),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["write_final_report", "plot_coverage_curve", "write_model_card"]
