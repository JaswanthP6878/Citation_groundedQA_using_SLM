from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qasper_rag.config import ensure_project_layout, get_default_project_paths
from qasper_rag.loader import resolve_qasper_split_path
from qasper_rag.processing import process_qasper_split_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical processed QASPER artifacts.")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "dev", "test", "all"),
        default="all",
        help="Which split to process.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional explicit input JSON path. Requires exactly one split run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional explicit processed root. Defaults to the configured shared processed directory.",
    )
    parser.add_argument(
        "--create-shared-layout",
        action="store_true",
        help="Create the configured shared/user directory layout before processing.",
    )
    args = parser.parse_args()

    paths = get_default_project_paths(repo_root=REPO_ROOT)
    ensure_project_layout(paths, create_shared=args.create_shared_layout)

    output_root = args.output_root or paths.qasper_processed_dir
    split_names = ("train", "validation", "test") if args.split == "all" else (args.split,)

    if args.input and len(split_names) != 1:
        raise ValueError("--input can only be used when processing exactly one split.")

    for split_name in split_names:
        canonical_split = "validation" if split_name == "dev" else split_name
        input_path = args.input or resolve_qasper_split_path(paths.qasper_extracted_dir, canonical_split)
        split_output_dir = output_root / canonical_split
        processed = process_qasper_split_file(input_path, split_output_dir)
        manifest = processed["manifest"]
        print(
            f"{canonical_split}: papers={manifest['paper_count']} "
            f"questions={manifest['question_count']} "
            f"chunks={manifest['chunk_counts']}"
        )
        print(f"wrote {split_output_dir}")


if __name__ == "__main__":
    main()
