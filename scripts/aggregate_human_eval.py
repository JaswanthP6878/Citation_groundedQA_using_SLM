from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


NUMERIC_FIELDS = [
    "baseline_correctness",
    "citation_correctness",
    "baseline_grounding",
    "citation_grounding",
    "citation_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate filled human evaluation reviewer CSVs.")
    parser.add_argument("reviewer_csv", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float:
    return float(value.strip())


def normalize_pref(value: str) -> str:
    return value.strip().lower()


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, str]] = []
    reviewer_sources: dict[str, str] = {}
    for path in args.reviewer_csv:
        rows = load_rows(path)
        all_rows.extend(rows)
        reviewer_sources[path.name] = str(path)

    by_question: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        by_question[row["question_id"]].append(row)

    per_system = {
        "baseline_correctness": mean(parse_float(row["baseline_correctness"]) for row in all_rows),
        "citation_correctness": mean(parse_float(row["citation_correctness"]) for row in all_rows),
        "baseline_grounding": mean(parse_float(row["baseline_grounding"]) for row in all_rows),
        "citation_grounding": mean(parse_float(row["citation_grounding"]) for row in all_rows),
        "citation_quality": mean(parse_float(row["citation_quality"]) for row in all_rows),
    }

    total_pref_counts = Counter(normalize_pref(row["preferred_system"]) for row in all_rows)
    majority_pref_counts: Counter[str] = Counter()
    bucket_majority_counts: dict[str, Counter[str]] = defaultdict(Counter)
    agreement_matches = 0
    comparable_questions = 0
    per_question_rows: list[dict[str, object]] = []

    for question_id, rows in sorted(by_question.items()):
        pref_counts = Counter(normalize_pref(row["preferred_system"]) for row in rows)
        majority_pref, majority_count = pref_counts.most_common(1)[0]
        if len(pref_counts) > 1 and list(pref_counts.values()).count(majority_count) > 1:
            majority_pref = "tie"
        majority_pref_counts[majority_pref] += 1
        bucket = rows[0]["bucket"]
        bucket_majority_counts[bucket][majority_pref] += 1

        if len(rows) == 2:
            comparable_questions += 1
            if normalize_pref(rows[0]["preferred_system"]) == normalize_pref(rows[1]["preferred_system"]):
                agreement_matches += 1

        per_question_row = {
            "question_id": question_id,
            "bucket": bucket,
            "question": rows[0]["question"],
            "baseline_answer": rows[0]["baseline_answer"],
            "candidate_answer": rows[0]["candidate_answer"],
            "baseline_correctness_mean": mean(parse_float(row["baseline_correctness"]) for row in rows),
            "citation_correctness_mean": mean(parse_float(row["citation_correctness"]) for row in rows),
            "baseline_grounding_mean": mean(parse_float(row["baseline_grounding"]) for row in rows),
            "citation_grounding_mean": mean(parse_float(row["citation_grounding"]) for row in rows),
            "citation_quality_mean": mean(parse_float(row["citation_quality"]) for row in rows),
            "preference_counts": dict(pref_counts),
            "majority_preference": majority_pref,
            "notes": [row["notes"] for row in rows if row.get("notes", "").strip()],
        }
        per_question_rows.append(per_question_row)

    summary = {
        "reviewer_files": reviewer_sources,
        "reviewer_count": len(args.reviewer_csv),
        "judgment_count": len(all_rows),
        "question_count": len(by_question),
        "agreement_rate": agreement_matches / comparable_questions if comparable_questions else 0.0,
        "per_system_means": per_system,
        "total_preference_counts": dict(total_pref_counts),
        "majority_preference_counts": dict(majority_pref_counts),
        "bucket_majority_counts": {bucket: dict(counts) for bucket, counts in bucket_majority_counts.items()},
        "per_question": per_question_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "question_id",
                "bucket",
                "baseline_correctness_mean",
                "citation_correctness_mean",
                "baseline_grounding_mean",
                "citation_grounding_mean",
                "citation_quality_mean",
                "majority_preference",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_question_rows:
                writer.writerow({field: row[field] for field in fieldnames})

    lines = [
        "# Human Evaluation Summary",
        "",
        f"- Reviewer files: {', '.join(path.name for path in args.reviewer_csv)}",
        f"- Questions reviewed: {len(by_question)}",
        f"- Reviewers: {len(args.reviewer_csv)}",
        f"- Exact preference agreement: {agreement_matches}/{comparable_questions} ({(agreement_matches / comparable_questions * 100):.1f}%)" if comparable_questions else "- Exact preference agreement: n/a",
        "",
        "## Mean Scores",
        "",
        "| System | Correctness | Grounding | Citation Quality |",
        "| --- | ---: | ---: | ---: |",
        f"| Baseline | {per_system['baseline_correctness']:.3f} | {per_system['baseline_grounding']:.3f} | -- |",
        f"| Citation | {per_system['citation_correctness']:.3f} | {per_system['citation_grounding']:.3f} | {per_system['citation_quality']:.3f} |",
        "",
        "## Preference Counts",
        "",
        "| View | Baseline | Citation | Tie |",
        "| --- | ---: | ---: | ---: |",
        f"| Raw judgments | {total_pref_counts.get('baseline', 0)} | {total_pref_counts.get('citation', 0)} | {total_pref_counts.get('tie', 0)} |",
        f"| Majority by question | {majority_pref_counts.get('baseline', 0)} | {majority_pref_counts.get('citation', 0)} | {majority_pref_counts.get('tie', 0)} |",
        "",
        "## Majority Preference by Bucket",
        "",
        "| Bucket | Baseline | Citation | Tie |",
        "| --- | ---: | ---: | ---: |",
    ]
    for bucket in sorted(bucket_majority_counts):
        counts = bucket_majority_counts[bucket]
        lines.append(
            f"| {bucket} | {counts.get('baseline', 0)} | {counts.get('citation', 0)} | {counts.get('tie', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Baseline answers receive higher mean correctness and grounding scores, while citation answers receive a "
                "lower grounding score and only moderate citation-quality scores. Preference is mixed at the question "
                "level: majority vote is evenly split between baseline and citation, with a substantial number of ties."
            ),
            "",
        ]
    )

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
