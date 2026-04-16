from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qasper_rag.artifacts import load_processed_chunks, load_processed_questions, resolve_processed_split_dir
from qasper_rag.config import ensure_project_layout, get_default_project_paths
from qasper_rag.generation_eval import collect_gold_answers
from qasper_rag.retrieval_eval import collect_gold_evidence, chunk_matches_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize generation failures from a saved run JSON.")
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--bertscore-threshold", type=float, default=0.88)
    parser.add_argument("--token-f1-threshold", type=float, default=0.35)
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = get_default_project_paths(repo_root=REPO_ROOT)
    ensure_project_layout(paths)
    processed_root = args.processed_root or paths.qasper_processed_dir

    payload = json.loads(args.run_path.read_text(encoding="utf-8"))
    split = str(payload.get("split", "validation"))
    strategy = str(payload.get("strategy", "section"))
    split_dir = resolve_processed_split_dir(processed_root, split)
    question_lookup = {question.question_id: question for question in load_processed_questions(split_dir / "questions.jsonl")}
    chunk_lookup = {chunk.chunk_id: chunk for chunk in load_processed_chunks(split_dir / f"chunks.{strategy}.jsonl")}

    categories = {
        "retrieval_miss": [],
        "lexical_mismatch": [],
        "unsupported_citation": [],
        "overcautious_unanswerable": [],
        "grounding_failure_with_evidence": [],
    }

    for question_payload in payload.get("questions", []):
        question_id = str(question_payload["question_id"])
        processed_question = question_lookup.get(question_id)
        if processed_question is None:
            continue

        prediction = question_payload.get("prediction", {})
        metrics = question_payload.get("metrics", {})
        retrieved_chunk_ids = [str(chunk_id) for chunk_id in question_payload.get("retrieved_chunk_ids", [])]
        retrieved_chunks = [chunk_lookup[chunk_id] for chunk_id in retrieved_chunk_ids if chunk_id in chunk_lookup]
        gold_evidence = collect_gold_evidence(processed_question)
        evidence_hit = any(
            chunk_matches_evidence(chunk, evidence_text)
            for chunk in retrieved_chunks
            for evidence_text in gold_evidence
        )
        gold_answers = collect_gold_answers(processed_question)
        gold_has_unanswerable = any(answer.lower() == "unanswerable" for answer in gold_answers)
        answer_text = str(prediction.get("answer_text", ""))

        example = {
            "question_id": question_id,
            "question": processed_question.question,
            "answer_text": answer_text,
            "gold_answers": list(gold_answers),
            "metrics": {
                "token_f1": metrics.get("token_f1", 0.0),
                "bertscore_f1": metrics.get("bertscore_f1", 0.0),
                "answer_supported": metrics.get("answer_supported", 0.0),
                "citation_precision": metrics.get("citation_precision", 0.0),
                "nli_citation_precision": metrics.get("nli_citation_precision", 0.0),
            },
            "retrieved_chunk_ids": retrieved_chunk_ids[:5],
        }

        if not evidence_hit:
            categories["retrieval_miss"].append(example)

        if (
            metrics.get("answer_supported", 0.0) > 0.0
            and metrics.get("token_f1", 0.0) < args.token_f1_threshold
            and metrics.get("bertscore_f1", 0.0) >= args.bertscore_threshold
        ):
            categories["lexical_mismatch"].append(example)

        if metrics.get("citation_count", 0) > 0 and metrics.get("nli_citation_precision", metrics.get("citation_precision", 0.0)) < 0.5:
            categories["unsupported_citation"].append(example)

        if answer_text == "unanswerable" and not gold_has_unanswerable:
            categories["overcautious_unanswerable"].append(example)

        if evidence_hit and metrics.get("answer_supported", 0.0) == 0.0:
            categories["grounding_failure_with_evidence"].append(example)

    summary = {
        "run_path": str(args.run_path),
        "question_count": len(payload.get("questions", [])),
        "categories": {
            name: {
                "count": len(examples),
                "examples": examples[: args.example_limit],
            }
            for name, examples in categories.items()
        },
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown_lines = [
        f"# Error analysis for {args.run_path.name}",
        "",
        f"- Questions analyzed: {summary['question_count']}",
        "",
    ]
    for name, details in summary["categories"].items():
        markdown_lines.append(f"## {name.replace('_', ' ').title()}")
        markdown_lines.append("")
        markdown_lines.append(f"- Count: {details['count']}")
        for example in details["examples"]:
            markdown_lines.append(f"- `{example['question_id']}` {example['question']}")
            markdown_lines.append(f"  Answer: `{example['answer_text']}`")
            markdown_lines.append(f"  Gold: `{'; '.join(example['gold_answers'])}`")
            markdown_lines.append(
                "  Metrics: "
                f"F1={example['metrics']['token_f1']:.4f}, "
                f"BERTScore={example['metrics']['bertscore_f1']:.4f}, "
                f"support={example['metrics']['answer_supported']:.4f}, "
                f"cite={example['metrics']['citation_precision']:.4f}, "
                f"nli={example['metrics']['nli_citation_precision']:.4f}"
            )
        markdown_lines.append("")
    markdown = "\n".join(markdown_lines).rstrip() + "\n"

    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
