from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qasper_rag.artifacts import (
    group_chunks_by_paper,
    load_processed_chunks,
    load_processed_manifest,
    load_processed_questions,
    resolve_processed_split_dir,
)
from qasper_rag.config import ensure_project_layout, get_default_project_paths
from qasper_rag.retrieval import CrossEncoderReranker, build_retriever
from qasper_rag.retrieval_eval import aggregate_question_metrics, evaluate_ranked_chunks


class PaperRetrieverCache:
    def __init__(
        self,
        chunks_by_paper,
        *,
        method: str,
        dense_model_name: str,
        reranker_model_name: str,
        retrieval_depth: int,
        use_faiss: bool,
    ) -> None:
        self.chunks_by_paper = chunks_by_paper
        self.method = method
        self.dense_model_name = dense_model_name
        self.reranker_model_name = reranker_model_name
        self.retrieval_depth = retrieval_depth
        self.use_faiss = use_faiss
        self._retrievers = {}
        self._dense_encoder = None
        self._reranker = None

    def get(self, paper_id: str):
        if paper_id not in self._retrievers:
            paper_chunks = self.chunks_by_paper.get(paper_id, [])
            self._retrievers[paper_id] = build_retriever(
                self.method,
                paper_chunks,
                encoder=self._get_dense_encoder() if self._needs_dense() else None,
                dense_model_name=self.dense_model_name,
                reranker=self._get_reranker() if self._needs_reranker() else None,
                reranker_model_name=self.reranker_model_name,
                retrieval_depth=self.retrieval_depth,
                use_faiss=self.use_faiss,
            )
        return self._retrievers[paper_id]

    def _needs_dense(self) -> bool:
        return "dense" in self.method or "hybrid" in self.method

    def _needs_reranker(self) -> bool:
        return self.method.endswith("_rerank")

    def _get_dense_encoder(self):
        if self._dense_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                from qasper_rag.retrieval import HuggingFaceEncoder

                self._dense_encoder = HuggingFaceEncoder(self.dense_model_name)
            else:
                self._dense_encoder = SentenceTransformer(self.dense_model_name)
        return self._dense_encoder

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(model_name=self.reranker_model_name)
        return self._reranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QASPER retrieval on processed artifacts.")
    parser.add_argument("--split", choices=("train", "validation", "dev", "test"), default="validation")
    parser.add_argument("--strategy", choices=("fixed", "sliding", "section"), default="section")
    parser.add_argument(
        "--method",
        choices=("bm25", "dense", "hybrid", "bm25_rerank", "dense_rerank", "hybrid_rerank"),
        default="bm25",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mrr-k", type=int, default=10)
    parser.add_argument("--retrieval-depth", type=int, default=20)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--paper-id", type=str, default=None)
    parser.add_argument("--dense-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--reranker-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--disable-faiss", action="store_true")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Optional explicit processed artifact root. Defaults to the configured shared processed directory.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = get_default_project_paths(repo_root=REPO_ROOT)
    ensure_project_layout(paths)

    processed_root = args.processed_root or paths.qasper_processed_dir
    split_dir = resolve_processed_split_dir(processed_root, args.split)
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Processed split directory not found: {split_dir}. "
            "Build processed artifacts first or pass --processed-root."
        )
    manifest = load_processed_manifest(split_dir / "manifest.json")
    chunks = load_processed_chunks(split_dir / f"chunks.{args.strategy}.jsonl")
    questions = load_processed_questions(split_dir / "questions.jsonl")

    if args.paper_id:
        questions = [question for question in questions if question.paper_id == args.paper_id]
        chunks = [chunk for chunk in chunks if chunk.paper_id == args.paper_id]

    if args.question_limit is not None:
        questions = questions[: args.question_limit]

    chunks_by_paper = group_chunks_by_paper(chunks)
    retriever_cache = PaperRetrieverCache(
        chunks_by_paper,
        method=args.method,
        dense_model_name=args.dense_model,
        reranker_model_name=args.reranker_model,
        retrieval_depth=args.retrieval_depth,
        use_faiss=not args.disable_faiss,
    )

    per_question_metrics = []
    search_k = max(args.top_k, args.mrr_k, args.retrieval_depth if args.method.endswith("_rerank") else 0)
    for question in questions:
        retriever = retriever_cache.get(question.paper_id)
        ranked_results = retriever.search(question.question, search_k)
        per_question_metrics.append(
            evaluate_ranked_chunks(
                question,
                [result.chunk for result in ranked_results],
                recall_k=args.top_k,
                mrr_k=args.mrr_k,
            )
        )

    summary = aggregate_question_metrics(per_question_metrics)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation" if args.split == "dev" else args.split,
        "strategy": args.strategy,
        "method": args.method,
        "top_k": args.top_k,
        "mrr_k": args.mrr_k,
        "retrieval_depth": args.retrieval_depth,
        "dense_model": args.dense_model if "dense" in args.method or "hybrid" in args.method else None,
        "reranker_model": args.reranker_model if args.method.endswith("_rerank") else None,
        "processed_manifest": manifest,
        "metrics": summary,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))

    output_path = args.output or default_output_path(paths.user_runs_dir, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {output_path}")


def default_output_path(user_runs_dir: Path, args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        user_runs_dir
        / "retrieval"
        / ("validation" if args.split == "dev" else args.split)
        / args.strategy
        / f"{args.method}.{timestamp}.json"
    )


if __name__ == "__main__":
    main()
