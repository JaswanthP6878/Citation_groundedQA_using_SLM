from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qasper_rag.artifacts import (
    load_processed_chunks,
    load_processed_questions,
    resolve_processed_split_dir,
)
from qasper_rag.config import ensure_project_layout, get_default_project_paths
from qasper_rag.nli_eval import HuggingFaceEntailmentScorer, evaluate_cited_claims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill NLI citation metrics into saved generation result JSON files.")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--pattern", type=str, default="generation/**/*.json")
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--suffix", type=str, default=None)
    return parser.parse_args()


def discover_result_paths(run_root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in run_root.glob(pattern) if path.is_file())


def select_output_path(path: Path, suffix: str | None) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def load_question_lookup(processed_root: Path, split: str) -> dict[str, object]:
    split_dir = resolve_processed_split_dir(processed_root, split)
    questions = load_processed_questions(split_dir / "questions.jsonl")
    return {question.question_id: question for question in questions}


def load_chunk_lookup(processed_root: Path, split: str, strategy: str) -> dict[str, object]:
    split_dir = resolve_processed_split_dir(processed_root, split)
    chunks = load_processed_chunks(split_dir / f"chunks.{strategy}.jsonl")
    return {chunk.chunk_id: chunk for chunk in chunks}


def main() -> None:
    args = parse_args()

    paths = get_default_project_paths(repo_root=REPO_ROOT)
    ensure_project_layout(paths)

    run_root = args.run_root or paths.user_runs_dir
    processed_root = args.processed_root or paths.qasper_processed_dir
    result_paths = discover_result_paths(run_root, args.pattern)
    if not result_paths:
        raise FileNotFoundError(f"No result files matched {args.pattern!r} under {run_root}.")

    scorer = HuggingFaceEntailmentScorer(args.model_name, device=args.device)
    question_lookups: dict[str, dict[str, object]] = {}
    chunk_lookups: dict[tuple[str, str], dict[str, object]] = {}
    processed_count = 0
    skipped_count = 0

    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "questions" not in payload or "metrics" not in payload:
            skipped_count += 1
            continue

        metrics = payload.get("metrics", {})
        questions_payload = payload.get("questions", [])
        if args.skip_existing and "nli_citation_precision" in metrics and all(
            "nli_citation_precision" in question_payload.get("metrics", {}) for question_payload in questions_payload
        ) and "nli_backfilled_at_utc" in payload:
            skipped_count += 1
            continue

        split = str(payload.get("split", "validation"))
        strategy = str(payload.get("strategy", "section"))

        if split not in question_lookups:
            question_lookups[split] = load_question_lookup(processed_root, split)
        question_lookup = question_lookups[split]

        chunk_cache_key = (split, strategy)
        if chunk_cache_key not in chunk_lookups:
            chunk_lookups[chunk_cache_key] = load_chunk_lookup(processed_root, split, strategy)
        chunk_lookup = chunk_lookups[chunk_cache_key]

        question_precisions: list[float] = []
        total_cited_sentences = 0
        total_supported_sentences = 0

        for question_payload in questions_payload:
            question_id = str(question_payload["question_id"])
            processed_question = question_lookup.get(question_id)
            if processed_question is None:
                raise KeyError(f"{path}: missing processed question for id {question_id}")

            prediction = question_payload.get("prediction", {})
            citation_labels = [int(label) for label in prediction.get("citation_labels", [])]
            cited_chunk_ids = [str(chunk_id) for chunk_id in prediction.get("cited_chunk_ids", [])]
            label_to_chunk_id = dict(zip(citation_labels, cited_chunk_ids))

            analysis = evaluate_cited_claims(
                processed_question.question,
                str(prediction.get("raw_output", "")),
                label_to_chunk_id,
                chunk_lookup,
                scorer,
                threshold=args.threshold,
                batch_size=args.batch_size,
            )

            question_metrics = question_payload.setdefault("metrics", {})
            question_metrics["nli_citation_precision"] = float(analysis["nli_citation_precision"])
            question_metrics["nli_cited_sentence_count"] = int(analysis["nli_cited_sentence_count"])
            question_metrics["nli_supported_sentence_count"] = int(analysis["nli_supported_sentence_count"])
            question_payload["nli_claim_results"] = analysis["claim_results"]

            question_precisions.append(float(analysis["nli_citation_precision"]))
            total_cited_sentences += int(analysis["nli_cited_sentence_count"])
            total_supported_sentences += int(analysis["nli_supported_sentence_count"])

        payload["metrics"]["nli_citation_precision"] = mean(question_precisions) if question_precisions else 0.0
        payload["metrics"]["nli_cited_sentence_count"] = total_cited_sentences
        payload["metrics"]["nli_supported_sentence_count"] = total_supported_sentences
        payload["metrics"]["nli_supported_sentence_rate"] = (
            total_supported_sentences / total_cited_sentences if total_cited_sentences else 0.0
        )
        payload["nli_model_name"] = args.model_name
        payload["nli_device"] = scorer.device
        payload["nli_threshold"] = args.threshold
        payload["nli_backfilled_at_utc"] = datetime.now(timezone.utc).isoformat()

        output_path = select_output_path(path, args.suffix)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        processed_count += 1
        print(f"updated {output_path}")

    print(
        json.dumps(
            {
                "processed_count": processed_count,
                "skipped_count": skipped_count,
                "model_name": args.model_name,
                "device": scorer.device,
                "threshold": args.threshold,
                "run_root": str(run_root),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
