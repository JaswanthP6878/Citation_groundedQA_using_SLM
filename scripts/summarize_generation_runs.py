from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qasper_rag.config import ensure_project_layout, get_default_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown summary table from generation result JSONs.")
    parser.add_argument("run_paths", nargs="*", type=Path)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def discover_run_paths(run_root: Path, patterns: list[str]) -> list[Path]:
    discovered: list[Path] = []
    for pattern in patterns:
        discovered.extend(path for path in run_root.glob(pattern) if path.is_file())
    return sorted(set(discovered))


def payload_signature(payload: dict[str, object]) -> tuple[str, ...]:
    metrics = payload.get("metrics", {})
    return (
        str(payload.get("split", "")),
        str(payload.get("strategy", "")),
        str(payload.get("retrieval_method", "")),
        str(payload.get("prompt_style", "")),
        str(payload.get("generation_model", "")),
        str(metrics.get("question_count", "")),
    )


def select_latest_payloads(paths: list[Path]) -> list[tuple[Path, dict[str, object]]]:
    latest_by_signature: dict[tuple[str, ...], tuple[Path, dict[str, object]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        signature = payload_signature(payload)
        candidate_time = payload.get("generated_at_utc")
        current = latest_by_signature.get(signature)
        if current is None:
            latest_by_signature[signature] = (path, payload)
            continue
        current_time = current[1].get("generated_at_utc")
        if str(candidate_time) >= str(current_time):
            latest_by_signature[signature] = (path, payload)
    return sorted(latest_by_signature.values(), key=lambda item: item[0].name)


def format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown_table(rows: list[tuple[Path, dict[str, object]]]) -> str:
    header = (
        "| file | split | strategy | retrieval | prompt | model | questions | "
        "token_f1 | bertscore_f1 | hallucination | supported | cite_prec | cite_rate | nli_cite |\n"
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = [header.rstrip()]
    for path, payload in rows:
        metrics = payload.get("metrics", {})
        model_name = str(payload.get("generation_model", "")).split("/")[-1]
        lines.append(
            "| "
            + " | ".join(
                [
                    path.name,
                    str(payload.get("split", "")),
                    str(payload.get("strategy", "")),
                    str(payload.get("retrieval_method", "")),
                    str(payload.get("prompt_style", "")),
                    model_name,
                    format_metric(metrics.get("question_count", "")),
                    format_metric(metrics.get("token_f1", 0.0)),
                    format_metric(metrics.get("bertscore_f1", 0.0)),
                    format_metric(metrics.get("hallucination_rate", 0.0)),
                    format_metric(metrics.get("supported_answer_rate", 0.0)),
                    format_metric(metrics.get("citation_precision", 0.0)),
                    format_metric(metrics.get("citation_rate", 0.0)),
                    format_metric(metrics.get("nli_citation_precision", 0.0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    paths = get_default_project_paths(repo_root=REPO_ROOT)
    ensure_project_layout(paths)

    run_root = args.run_root or paths.user_runs_dir
    candidate_paths = list(args.run_paths) or discover_run_paths(run_root, args.pattern or ["generation/**/*.json"])
    if not candidate_paths:
        raise FileNotFoundError("No generation run JSON files matched the provided inputs.")

    rows = select_latest_payloads(candidate_paths) if args.latest_only else [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(candidate_paths)
    ]
    markdown = build_markdown_table(rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(
        json.dumps(
            {
                "row_count": len(rows),
                "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
