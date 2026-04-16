from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare overlapping questions between two generation run JSON files.")
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument("--metric", default="token_f1", help="Primary per-question metric for improvement/regression examples.")
    parser.add_argument("--example-limit", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def load_run(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def question_index(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(question["question_id"]): question
        for question in payload.get("questions", [])
        if "question_id" in question
    }


def metric_mean(questions: list[dict[str, object]], metric_name: str) -> float:
    values = [float(question.get("metrics", {}).get(metric_name, 0.0)) for question in questions]
    return mean(values) if values else 0.0


def summarize_overlap(
    baseline_questions: list[dict[str, object]],
    candidate_questions: list[dict[str, object]],
) -> dict[str, float]:
    metrics = [
        "token_f1",
        "bertscore_f1",
        "citation_precision",
        "citation_rate",
        "hallucination",
        "answer_supported",
        "nli_citation_precision",
    ]
    summary: dict[str, float] = {"question_count": float(len(baseline_questions))}
    for metric_name in metrics:
        summary[f"baseline_{metric_name}"] = metric_mean(baseline_questions, metric_name)
        summary[f"candidate_{metric_name}"] = metric_mean(candidate_questions, metric_name)
        summary[f"delta_{metric_name}"] = (
            summary[f"candidate_{metric_name}"] - summary[f"baseline_{metric_name}"]
        )
    return summary


def example_record(
    question_id: str,
    baseline_question: dict[str, object],
    candidate_question: dict[str, object],
    metric_name: str,
) -> dict[str, object]:
    baseline_prediction = baseline_question.get("prediction", {})
    candidate_prediction = candidate_question.get("prediction", {})
    baseline_metrics = baseline_question.get("metrics", {})
    candidate_metrics = candidate_question.get("metrics", {})
    return {
        "question_id": question_id,
        "question": str(candidate_question.get("question", baseline_question.get("question", ""))),
        "baseline_answer": str(baseline_prediction.get("answer_text", "")),
        "candidate_answer": str(candidate_prediction.get("answer_text", "")),
        "baseline_raw_output": str(baseline_prediction.get("raw_output", "")),
        "candidate_raw_output": str(candidate_prediction.get("raw_output", "")),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": float(candidate_metrics.get(metric_name, 0.0)) - float(baseline_metrics.get(metric_name, 0.0)),
    }


def main() -> None:
    args = parse_args()

    baseline_run = load_run(args.baseline_run)
    candidate_run = load_run(args.candidate_run)
    baseline_index = question_index(baseline_run)
    candidate_index = question_index(candidate_run)

    overlap_ids = [question_id for question_id in candidate_index if question_id in baseline_index]
    baseline_questions = [baseline_index[question_id] for question_id in overlap_ids]
    candidate_questions = [candidate_index[question_id] for question_id in overlap_ids]

    summary = summarize_overlap(baseline_questions, candidate_questions)

    examples = [
        example_record(question_id, baseline_index[question_id], candidate_index[question_id], args.metric)
        for question_id in overlap_ids
    ]
    regressions = sorted(examples, key=lambda example: example["delta"])[: args.example_limit]
    improvements = sorted(examples, key=lambda example: example["delta"], reverse=True)[: args.example_limit]

    payload = {
        "baseline_run": str(args.baseline_run),
        "candidate_run": str(args.candidate_run),
        "comparison_metric": args.metric,
        "summary": summary,
        "top_regressions": regressions,
        "top_improvements": improvements,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# Comparison: {args.candidate_run.name} vs {args.baseline_run.name}",
        "",
        f"- Overlapping questions: {int(summary['question_count'])}",
        f"- Primary comparison metric: `{args.metric}`",
        "",
        "## Summary",
        "",
    ]

    for metric_name in [
        "token_f1",
        "bertscore_f1",
        "citation_precision",
        "citation_rate",
        "hallucination",
        "answer_supported",
        "nli_citation_precision",
    ]:
        lines.append(
            f"- `{metric_name}`: "
            f"{summary[f'baseline_{metric_name}']:.4f} -> {summary[f'candidate_{metric_name}']:.4f} "
            f"({summary[f'delta_{metric_name}']:+.4f})"
        )

    lines.extend(["", "## Top Regressions", ""])
    for example in regressions:
        lines.append(f"- `{example['question_id']}` {example['question']}")
        lines.append(f"  Baseline: `{example['baseline_answer']}`")
        lines.append(f"  Candidate: `{example['candidate_answer']}`")
        lines.append(f"  Delta {args.metric}: {example['delta']:+.4f}")
        lines.append("")

    lines.extend(["## Top Improvements", ""])
    for example in improvements:
        lines.append(f"- `{example['question_id']}` {example['question']}")
        lines.append(f"  Baseline: `{example['baseline_answer']}`")
        lines.append(f"  Candidate: `{example['candidate_answer']}`")
        lines.append(f"  Delta {args.metric}: {example['delta']:+.4f}")
        lines.append("")

    markdown = "\n".join(lines).rstrip() + "\n"
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
