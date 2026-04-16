from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from qasper_rag.artifacts import load_processed_chunks, load_processed_questions, resolve_processed_split_dir
from qasper_rag.generation_eval import collect_gold_answers


_WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small human-evaluation sample from two generation runs.")
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=505)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--strategy", default="section")
    parser.add_argument("--chunk-char-limit", type=int, default=500)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_yes_no_question(question_text: str) -> bool:
    lowered = question_text.strip().lower()
    return lowered.startswith(
        (
            "is ",
            "are ",
            "was ",
            "were ",
            "do ",
            "does ",
            "did ",
            "can ",
            "could ",
            "should ",
            "would ",
            "has ",
            "have ",
            "had ",
        )
    )


def sample_from_bucket(
    bucket: list[dict[str, object]],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    if len(bucket) <= sample_size:
        return list(bucket)
    return rng.sample(bucket, sample_size)


def squash_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def truncate_text(text: str, char_limit: int) -> str:
    normalized = squash_whitespace(text)
    if char_limit <= 0 or len(normalized) <= char_limit:
        return normalized
    return normalized[: max(0, char_limit - 3)].rstrip() + "..."


def format_chunk_context(
    chunk_ids: list[str],
    chunk_text_by_id: dict[str, str],
    char_limit: int,
) -> str:
    parts: list[str] = []
    for label, chunk_id in enumerate(chunk_ids, start=1):
        chunk_text = chunk_text_by_id.get(chunk_id)
        if not chunk_text:
            continue
        parts.append(f"[{label}] {truncate_text(chunk_text, char_limit)}")
    return "\n\n".join(parts)


def format_cited_context(
    citation_labels: list[int],
    chunk_ids: list[str],
    chunk_text_by_id: dict[str, str],
    char_limit: int,
) -> str:
    parts: list[str] = []
    for label in citation_labels:
        if label <= 0 or label > len(chunk_ids):
            continue
        chunk_id = chunk_ids[label - 1]
        chunk_text = chunk_text_by_id.get(chunk_id)
        if not chunk_text:
            continue
        parts.append(f"[{label}] {truncate_text(chunk_text, char_limit)}")
    return "\n\n".join(parts)


def format_nli_claim_summary(claim_results: list[dict[str, object]]) -> str:
    if not claim_results:
        return ""
    lines: list[str] = []
    for claim in claim_results:
        sentence_text = squash_whitespace(str(claim.get("sentence_text", "")))
        labels = ",".join(str(label) for label in claim.get("citation_labels", []))
        supported = "yes" if claim.get("supported", False) else "no"
        entailment = float(claim.get("max_entailment_score", 0.0))
        lines.append(f"[{labels}] supported={supported} entail={entailment:.3f} :: {sentence_text}")
    return "\n".join(lines)


def load_question_and_chunk_context(
    processed_root: Path | None,
    split: str,
    strategy: str,
) -> tuple[dict[str, object], dict[str, str]]:
    if not processed_root:
        return {}, {}
    split_dir = resolve_processed_split_dir(processed_root, split)
    question_lookup = {
        question.question_id: question
        for question in load_processed_questions(split_dir / "questions.jsonl")
    }
    chunk_text_by_id = {
        chunk.chunk_id: chunk.text
        for chunk in load_processed_chunks(split_dir / f"chunks.{strategy}.jsonl")
    }
    return question_lookup, chunk_text_by_id


def build_row(
    baseline_entry: dict[str, object],
    candidate_entry: dict[str, object],
    question_lookup: dict[str, object],
    chunk_text_by_id: dict[str, str],
    chunk_char_limit: int,
) -> dict[str, object]:
    baseline_metrics = dict(baseline_entry.get("metrics", {}))
    candidate_metrics = dict(candidate_entry.get("metrics", {}))
    baseline_prediction = dict(baseline_entry.get("prediction", {}))
    candidate_prediction = dict(candidate_entry.get("prediction", {}))

    question_id = str(candidate_entry["question_id"])
    question_text = str(candidate_entry.get("question", baseline_entry.get("question", "")))
    processed_question = question_lookup.get(question_id)

    gold_answers = list(collect_gold_answers(processed_question)) if processed_question else []
    gold_evidence: list[str] = []
    if processed_question:
        seen_evidence: set[str] = set()
        for answer in processed_question.answers:
            for evidence_text in answer.evidence:
                normalized = squash_whitespace(evidence_text)
                if normalized and normalized not in seen_evidence:
                    seen_evidence.add(normalized)
                    gold_evidence.append(normalized)

    baseline_retrieved_chunk_ids = [str(chunk_id) for chunk_id in baseline_entry.get("retrieved_chunk_ids", [])]
    candidate_retrieved_chunk_ids = [str(chunk_id) for chunk_id in candidate_entry.get("retrieved_chunk_ids", [])]
    candidate_citation_labels = [int(label) for label in candidate_prediction.get("citation_labels", [])]
    candidate_cited_chunk_ids = [str(chunk_id) for chunk_id in candidate_prediction.get("cited_chunk_ids", [])]

    return {
        "question_id": question_id,
        "paper_id": str(candidate_entry.get("paper_id", baseline_entry.get("paper_id", ""))),
        "question": question_text,
        "gold_answers": " || ".join(gold_answers),
        "gold_evidence": " || ".join(gold_evidence),
        "baseline_answer": str(baseline_prediction.get("answer_text", "")),
        "candidate_answer": str(candidate_prediction.get("answer_text", "")),
        "baseline_raw_output": str(baseline_prediction.get("raw_output", "")),
        "candidate_raw_output": str(candidate_prediction.get("raw_output", "")),
        "baseline_token_f1": float(baseline_metrics.get("token_f1", 0.0)),
        "candidate_token_f1": float(candidate_metrics.get("token_f1", 0.0)),
        "baseline_support": float(baseline_metrics.get("answer_supported", 0.0)),
        "candidate_support": float(candidate_metrics.get("answer_supported", 0.0)),
        "baseline_citation_precision": float(baseline_metrics.get("citation_precision", 0.0)),
        "candidate_citation_precision": float(candidate_metrics.get("citation_precision", 0.0)),
        "baseline_retrieved_chunk_ids": " || ".join(baseline_retrieved_chunk_ids),
        "candidate_retrieved_chunk_ids": " || ".join(candidate_retrieved_chunk_ids),
        "baseline_retrieved_context": format_chunk_context(
            baseline_retrieved_chunk_ids,
            chunk_text_by_id,
            chunk_char_limit,
        ),
        "candidate_retrieved_context": format_chunk_context(
            candidate_retrieved_chunk_ids,
            chunk_text_by_id,
            chunk_char_limit,
        ),
        "candidate_citation_labels": " || ".join(str(label) for label in candidate_citation_labels),
        "candidate_cited_chunk_ids": " || ".join(candidate_cited_chunk_ids),
        "candidate_cited_context": format_cited_context(
            candidate_citation_labels,
            candidate_retrieved_chunk_ids,
            chunk_text_by_id,
            chunk_char_limit,
        ),
        "candidate_nli_claim_summary": format_nli_claim_summary(
            list(candidate_entry.get("nli_claim_results", []))
        ),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    baseline = load_payload(args.baseline_run)
    candidate = load_payload(args.candidate_run)
    baseline_index = {entry["question_id"]: entry for entry in baseline.get("questions", [])}
    candidate_index = {entry["question_id"]: entry for entry in candidate.get("questions", [])}
    overlap_ids = [question_id for question_id in candidate_index if question_id in baseline_index]
    question_lookup, chunk_text_by_id = load_question_and_chunk_context(
        args.processed_root,
        args.split,
        args.strategy,
    )

    buckets = {
        "citation_better": [],
        "baseline_better": [],
        "large_disagreement": [],
        "yes_no": [],
    }
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for question_id in overlap_ids:
        row = build_row(
            baseline_index[question_id],
            candidate_index[question_id],
            question_lookup,
            chunk_text_by_id,
            args.chunk_char_limit,
        )
        question_text = str(row["question"])
        if row["candidate_token_f1"] > row["baseline_token_f1"]:
            buckets["citation_better"].append(row)
        if row["baseline_token_f1"] > row["candidate_token_f1"]:
            buckets["baseline_better"].append(row)
        if abs(row["candidate_token_f1"] - row["baseline_token_f1"]) >= 0.5:
            buckets["large_disagreement"].append(row)
        if is_yes_no_question(question_text):
            buckets["yes_no"].append(row)

    target_per_bucket = max(1, args.sample_size // len(buckets))
    for bucket_name in ("large_disagreement", "citation_better", "baseline_better", "yes_no"):
        for row in sample_from_bucket(buckets[bucket_name], target_per_bucket, rng):
            if row["question_id"] not in seen_ids:
                rows.append({"bucket": bucket_name, **row})
                seen_ids.add(row["question_id"])

    remaining = [question_id for question_id in overlap_ids if question_id not in seen_ids]
    if len(rows) < args.sample_size:
        for question_id in sample_from_bucket(remaining, args.sample_size - len(rows), rng):
            rows.append(
                {
                    "bucket": "random_fill",
                    **build_row(
                        baseline_index[question_id],
                        candidate_index[question_id],
                        question_lookup,
                        chunk_text_by_id,
                        args.chunk_char_limit,
                    ),
                }
            )

    rows = rows[: args.sample_size]

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["bucket"])
            writer.writeheader()
            writer.writerows(rows)

    markdown_lines = [
        f"# Human Evaluation Sample ({len(rows)} items)",
        "",
        f"- Baseline run: `{args.baseline_run.name}`",
        f"- Candidate run: `{args.candidate_run.name}`",
        "",
    ]
    for row in rows:
        markdown_lines.append(f"## {row['question_id']} [{row['bucket']}]")
        markdown_lines.append("")
        markdown_lines.append(f"- Question: {row['question']}")
        if row.get("gold_answers"):
            markdown_lines.append(f"- Gold answers: `{row['gold_answers']}`")
        if row.get("gold_evidence"):
            markdown_lines.append(f"- Gold evidence: `{row['gold_evidence']}`")
        markdown_lines.append(f"- Baseline answer: `{row['baseline_answer']}`")
        markdown_lines.append(f"- Candidate answer: `{row['candidate_answer']}`")
        markdown_lines.append(
            f"- Baseline metrics: F1={row['baseline_token_f1']:.4f}, support={row['baseline_support']:.4f}, cite={row['baseline_citation_precision']:.4f}"
        )
        markdown_lines.append(
            f"- Candidate metrics: F1={row['candidate_token_f1']:.4f}, support={row['candidate_support']:.4f}, cite={row['candidate_citation_precision']:.4f}"
        )
        if row.get("candidate_cited_context"):
            markdown_lines.append("- Candidate cited context:")
            markdown_lines.append("")
            markdown_lines.append("```text")
            markdown_lines.append(str(row["candidate_cited_context"]))
            markdown_lines.append("```")
        if row.get("candidate_nli_claim_summary"):
            markdown_lines.append("- Candidate NLI claim summary:")
            markdown_lines.append("")
            markdown_lines.append("```text")
            markdown_lines.append(str(row["candidate_nli_claim_summary"]))
            markdown_lines.append("```")
        markdown_lines.append("")
    markdown = "\n".join(markdown_lines).rstrip() + "\n"

    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
