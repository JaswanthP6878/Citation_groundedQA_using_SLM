from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qasper_rag.artifacts import load_processed_questions, resolve_processed_split_dir
from qasper_rag.config import ensure_project_layout, get_default_project_paths
from qasper_rag.generation_eval import collect_gold_answers, expand_multi_reference_pairs, max_group_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill BERTScore into saved generation result JSON files.")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--pattern", type=str, default="generation/**/*.json")
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--model-type", type=str, default="roberta-large")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--suffix", type=str, default=None)
    return parser.parse_args()


def discover_result_paths(run_root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in run_root.glob(pattern) if path.is_file())


def load_question_lookup(processed_root: Path, split: str) -> dict[str, object]:
    split_dir = resolve_processed_split_dir(processed_root, split)
    questions = load_processed_questions(split_dir / "questions.jsonl")
    return {question.question_id: question for question in questions}


def select_output_path(path: Path, suffix: str | None) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def main() -> None:
    args = parse_args()

    paths = get_default_project_paths(repo_root=REPO_ROOT)
    ensure_project_layout(paths)

    run_root = args.run_root or paths.user_runs_dir
    processed_root = args.processed_root or paths.qasper_processed_dir
    result_paths = discover_result_paths(run_root, args.pattern)
    if not result_paths:
        raise FileNotFoundError(f"No result files matched {args.pattern!r} under {run_root}.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for BERTScore backfilling.") from exc
    try:
        from bert_score import score as bert_score_score
    except ImportError as exc:
        raise RuntimeError("bert-score is required. Install it before running this script.") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    question_lookups: dict[str, dict[str, object]] = {}
    processed_count = 0
    skipped_count = 0

    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "questions" not in payload or "metrics" not in payload:
            skipped_count += 1
            continue

        metrics = payload.get("metrics", {})
        questions_payload = payload.get("questions", [])
        if args.skip_existing and "bertscore_f1" in metrics and all(
            "bertscore_f1" in question_payload.get("metrics", {}) for question_payload in questions_payload
        ) and "bertscore_backfilled_at_utc" in payload:
            skipped_count += 1
            continue

        split = str(payload.get("split", "validation"))
        if split not in question_lookups:
            question_lookups[split] = load_question_lookup(processed_root, split)
        question_lookup = question_lookups[split]

        predictions: list[str] = []
        reference_sets: list[tuple[str, ...]] = []
        target_payloads: list[dict[str, object]] = []
        missing_question_ids: list[str] = []

        for question_payload in questions_payload:
            question_id = str(question_payload["question_id"])
            processed_question = question_lookup.get(question_id)
            if processed_question is None:
                missing_question_ids.append(question_id)
                continue

            prediction = question_payload.get("prediction", {})
            answer_text = str(prediction.get("answer_text", ""))
            gold_answers = collect_gold_answers(processed_question)

            predictions.append(answer_text)
            reference_sets.append(gold_answers)
            target_payloads.append(question_payload)

        if missing_question_ids:
            raise KeyError(
                f"{path}: could not find processed questions for ids: {', '.join(missing_question_ids[:5])}"
            )

        expanded_predictions, expanded_references, group_sizes = expand_multi_reference_pairs(predictions, reference_sets)
        if expanded_predictions:
            _, _, f1_scores = bert_score_score(
                expanded_predictions,
                expanded_references,
                model_type=args.model_type,
                batch_size=args.batch_size,
                device=device,
                verbose=False,
                rescale_with_baseline=False,
            )
            bertscore_values = max_group_scores(f1_scores.tolist(), group_sizes)
        else:
            bertscore_values = []

        for question_payload, bertscore_value in zip(target_payloads, bertscore_values):
            question_payload.setdefault("metrics", {})["bertscore_f1"] = float(bertscore_value)

        payload["metrics"]["bertscore_f1"] = mean(bertscore_values) if bertscore_values else 0.0
        payload["bertscore_model_type"] = args.model_type
        payload["bertscore_device"] = device
        payload["bertscore_backfilled_at_utc"] = datetime.now(timezone.utc).isoformat()

        output_path = select_output_path(path, args.suffix)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        processed_count += 1
        print(f"updated {output_path}")

    print(
        json.dumps(
            {
                "processed_count": processed_count,
                "skipped_count": skipped_count,
                "device": device,
                "model_type": args.model_type,
                "run_root": str(run_root),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
