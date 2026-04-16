from __future__ import annotations

import argparse
import csv
from pathlib import Path


ANNOTATION_FIELDS = [
    "reviewer",
    "baseline_correctness",
    "citation_correctness",
    "baseline_grounding",
    "citation_grounding",
    "citation_quality",
    "preferred_system",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fillable human-evaluation CSV template from a sample file.")
    parser.add_argument("sample_csv", type=Path)
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.sample_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    template_fieldnames = fieldnames + [field for field in ANNOTATION_FIELDS if field not in fieldnames]

    for row in rows:
        row["reviewer"] = args.reviewer
        for field in ANNOTATION_FIELDS:
            row.setdefault(field, "")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=template_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(args.output)


if __name__ == "__main__":
    main()
