from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qasper_rag.chunking import FixedChunker, SectionAwareChunker, SlidingWindowChunker
from qasper_rag.loader import load_qasper_json


def build_chunker(args: argparse.Namespace):
    if args.strategy == "fixed":
        return FixedChunker(
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            include_abstract=not args.skip_abstract,
            include_figures_and_tables=args.include_figures,
        )
    if args.strategy == "sliding":
        return SlidingWindowChunker(
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            include_abstract=not args.skip_abstract,
            include_figures_and_tables=args.include_figures,
        )
    if args.strategy == "section":
        return SectionAwareChunker(
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            include_abstract=not args.skip_abstract,
            include_figures_and_tables=args.include_figures,
        )
    raise ValueError(f"Unsupported strategy: {args.strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview QASPER chunking output.")
    parser.add_argument("--input", type=Path, required=True, help="Path to one extracted QASPER split JSON file.")
    parser.add_argument(
        "--strategy",
        choices=("fixed", "sliding", "section"),
        default="section",
        help="Chunking strategy to preview.",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Chunk token budget.")
    parser.add_argument("--overlap-tokens", type=int, default=128, help="Chunk token overlap.")
    parser.add_argument("--paper-limit", type=int, default=2, help="How many papers to preview.")
    parser.add_argument("--show-chunks", type=int, default=2, help="How many chunks per paper to print.")
    parser.add_argument("--skip-abstract", action="store_true", help="Exclude abstracts from chunking.")
    parser.add_argument(
        "--include-figures",
        action="store_true",
        help="Include figure/table captions as chunkable text units.",
    )
    args = parser.parse_args()

    papers = load_qasper_json(args.input)
    chunker = build_chunker(args)

    for paper in papers[: args.paper_limit]:
        chunks = chunker.chunk_paper(paper)
        print(f"Paper: {paper.paper_id}")
        print(f"  Title: {paper.title}")
        print(f"  Questions: {paper.question_count}  Answers: {paper.answer_count}  Chunks: {len(chunks)}")
        for chunk in chunks[: args.show_chunks]:
            sections = [" > ".join(path) for path in chunk.section_paths]
            print(f"  - {chunk.chunk_id}  tokens={chunk.token_count}  sections={sections}")
            print(f"    {chunk.text[:200].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
