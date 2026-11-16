"""Step 1 - dataset audit. Run this before training anything.

    python scripts/01_audit.py --data data/dataset
    python scripts/01_audit.py --data data/synthetic --no-label-check

Writes reports/audit.json and reports/audit.md. No client values are written
into either file -- only counts, field names and file stems.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from factfind.audit import audit
from factfind.knowledge import KnowledgeError, load_knowledge
from factfind.ocr import engine_report
from factfind.paths import DATASET_DIR, REPORTS_DIR


def write_markdown(report: dict, path: Path) -> None:
    lines: list[str] = ["# Dataset audit", ""]
    lines.append(f"Directory: `{report['directory']}`")
    lines.append("")

    if report.get("fatal"):
        lines += ["## Stopped", "", report["fatal"], ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines += [
        "## Size",
        "",
        f"- labelled pages: **{report['n_images_with_labels']}**",
        f"- images with no label file: **{report['n_images_without_labels']}**",
        f"- writers: **{report['n_writers']}**",
        f"- clients: **{report['n_clients']}**",
        f"- layouts: {', '.join(f'{k} ({v})' for k, v in report['layouts'].items())}",
        f"- fields per page: min {report['fields_per_page']['min']}, "
        f"mean {report['fields_per_page']['mean']}, max {report['fields_per_page']['max']}",
        "",
    ]

    dup = report.get("duplicates", {})
    lines += [
        "## Duplicates and unreadable pages",
        "",
        f"- exact duplicate groups: **{len(dup.get('exact_duplicate_groups', []))}** "
        f"({dup.get('n_exact_duplicates', 0)} redundant pages)",
        f"- near-duplicate pairs: **{dup.get('n_near_duplicate_pairs', 0)}**",
        f"- unreadable images: **{len(dup.get('unreadable', []))}**",
        "",
    ]
    if dup.get("unreadable"):
        lines.append("Unreadable: " + ", ".join(f"`{s}`" for s in dup["unreadable"][:20]))
        lines.append("")

    pii = report.get("personal_data", {})
    lines += [
        "## Personal data",
        "",
        f"- label files containing personal data: **{pii.get('n_files_with_personal_data', 0)}** "
        f"of {pii.get('total_files', 0)}",
    ]
    for kind, n in sorted(pii.get("by_type", {}).items()):
        lines.append(f"  - {kind}: {n} occurrence(s)")
    lines += [
        "",
        "> These values stay out of git (`data/` is ignored) and out of every report. "
        "If any of this is real client data, anonymise before sharing the folder.",
        "",
    ]

    if report.get("n_label_key_problems"):
        lines += ["## Label keys not in the knowledge file", ""]
        for problem in report["label_key_problems"][:30]:
            lines.append(f"- {problem}")
        lines.append("")

    dis = report.get("label_disagreements")
    if dis is not None:
        lines += [
            "## Labels OCR disagrees with",
            "",
            f"{len(dis)} flagged. These are the first candidates for relabelling: "
            "the page was read confidently, but nothing on it resembles the "
            "labelled value.",
            "",
        ]
        for row in dis[:25]:
            lines.append(f"- `{row['stem']}` **{row['field']}** - {row.get('detail', row['issue'])}")
        lines.append("")

    gaps = report.get("gaps", {})
    lines += [
        "## Gaps - what to add",
        "",
        f"{gaps.get('n_fields_needing_more_data', 0)} of "
        f"{len(report['coverage']['fields'])} fields have fewer examples than "
        f"needed to demonstrate their wrong-fill target.",
        "",
        f"Implied extra pages (rough): **{gaps.get('total_extra_pages_implied', 0)}**",
        "",
    ]
    for note in gaps.get("advice", []):
        lines.append(f"- {note}")
    lines.append("")

    short = gaps.get("fields_needing_more_data", [])
    if short:
        lines += [
            "| field | kind | have | need | add |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for row in short[:45]:
            lines.append(
                f"| `{row['field']}` | {row['kind']} | {row['n_examples']} "
                f"| {row['n_needed']} | **{row['shortfall']}** |"
            )
        lines.append("")

    zero = [r for r in report["coverage"]["fields"] if r["n_examples"] == 0]
    if zero:
        lines += [
            f"### {len(zero)} fields with no examples at all",
            "",
            "Nothing can be trained or measured for these:",
            "",
            ", ".join(f"`{r['field']}`" for r in zero[:80]),
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATASET_DIR))
    ap.add_argument("--no-label-check", action="store_true",
                    help="skip the OCR-vs-label check (much faster)")
    ap.add_argument("--label-check-limit", type=int, default=60)
    ap.add_argument("--out-prefix", default="audit")
    args = ap.parse_args()

    try:
        kn = load_knowledge()
    except KnowledgeError as exc:
        print("STOPPING. The knowledge file is not usable:\n")
        print(exc)
        return 2

    print(f"knowledge: v{kn.version}, {len(kn.sections)} sections, {len(kn.fields)} fields")
    for problem in kn.problems:
        print("  ", problem)
    print("ocr engines:", engine_report())
    print()

    report = audit(args.data, kn, check_labels=not args.no_label_check,
                   label_check_limit=args.label_check_limit)

    json_path = REPORTS_DIR / f"{args.out_prefix}.json"
    md_path = REPORTS_DIR / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown(report, md_path)

    if report.get("fatal"):
        print("STOPPING.", report["fatal"])
        print(f"\nwrote {md_path}")
        return 2

    print(f"pages {report['n_images_with_labels']} | writers {report['n_writers']} "
          f"| clients {report['n_clients']}")
    print(f"layouts: {report['layouts']}")
    print(f"fields with no examples: "
          f"{report['coverage']['n_fields_with_no_examples']}")
    print(f"fields short of target data: "
          f"{report['gaps']['n_fields_needing_more_data']}")
    print(f"near-duplicate pairs: {report['duplicates']['n_near_duplicate_pairs']}")
    if report.get("n_label_disagreements") is not None:
        print(f"labels OCR disagrees with: {report['n_label_disagreements']}")
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
