from __future__ import annotations

import argparse
from dataclasses import replace
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
from qasper_rag.citation_verifier import verify_citation_prediction
from qasper_rag.generation import (
    HuggingFaceGenerator,
    build_generation_prediction,
    build_prompt,
    serialize_generation_prediction,
)
from qasper_rag.generation_eval import (
    aggregate_generation_metrics,
    evaluate_generation_prediction,
    serialize_generation_metrics,
)
from qasper_rag.retrieval import CrossEncoderReranker, build_retriever
from qasper_rag.nli_eval import HuggingFaceEntailmentScorer


class PaperRetrieverCache:
    def __init__(
        self,
        chunks_by_paper,
        *,
        method: str,
        dense_model_name: str,
        reranker_model_name: str,
        dense_device: str | None,
        reranker_device: str | None,
        retrieval_depth: int,
        use_faiss: bool,
    ) -> None:
        self.chunks_by_paper = chunks_by_paper
        self.method = method
        self.dense_model_name = dense_model_name
        self.reranker_model_name = reranker_model_name
        self.dense_device = dense_device
        self.reranker_device = reranker_device
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

                self._dense_encoder = HuggingFaceEncoder(
                    self.dense_model_name,
                    device=self.dense_device,
                )
            else:
                sentence_transformer_kwargs = {}
                if self.dense_device is not None:
                    sentence_transformer_kwargs["device"] = self.dense_device
                self._dense_encoder = SentenceTransformer(
                    self.dense_model_name,
                    **sentence_transformer_kwargs,
                )
        return self._dense_encoder

    def _get_reranker(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                self._reranker = CrossEncoderReranker(
                    model_name=self.reranker_model_name,
                    device=self.reranker_device,
                )
            else:
                cross_encoder_kwargs = {}
                if self.reranker_device is not None:
                    cross_encoder_kwargs["device"] = self.reranker_device
                self._reranker = CrossEncoderReranker(
                    model=CrossEncoder(self.reranker_model_name, **cross_encoder_kwargs),
                    model_name=self.reranker_model_name,
                )
        return self._reranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QASPER retrieval + generation + evaluation.")
    parser.add_argument("--split", choices=("train", "validation", "dev", "test"), default="validation")
    parser.add_argument("--strategy", choices=("fixed", "sliding", "section"), default="section")
    parser.add_argument(
        "--retrieval-method",
        choices=("none", "random", "bm25", "dense", "hybrid", "bm25_rerank", "dense_rerank", "hybrid_rerank"),
        default="hybrid",
    )
    parser.add_argument("--prompt-style", choices=("baseline", "citation_forcing"), default="baseline")
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--retrieval-depth", type=int, default=20)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--paper-id", type=str, default=None)
    parser.add_argument("--dense-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--dense-device", type=str, default=None)
    parser.add_argument("--reranker-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-device", type=str, default=None)
    parser.add_argument("--generation-model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--citation-verify", action="store_true")
    parser.add_argument("--citation-verifier-model", type=str, default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--citation-verifier-device", type=str, default=None)
    parser.add_argument("--citation-verifier-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-context-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--disable-faiss", action="store_true")
    parser.add_argument("--processed-root", type=Path, default=None)
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
        method=args.retrieval_method,
        dense_model_name=args.dense_model,
        reranker_model_name=args.reranker_model,
        dense_device=args.dense_device,
        reranker_device=args.reranker_device,
        retrieval_depth=args.retrieval_depth,
        use_faiss=not args.disable_faiss,
    )
    generator = HuggingFaceGenerator(
        args.generation_model,
        max_input_tokens=args.max_input_tokens,
    )
    citation_scorer = None
    if args.citation_verify and args.prompt_style == "citation_forcing":
        citation_scorer = HuggingFaceEntailmentScorer(
            args.citation_verifier_model,
            device=args.citation_verifier_device,
        )

    question_payloads = []
    question_metrics = []
    search_k = max(args.retrieval_k, args.retrieval_depth if args.retrieval_method.endswith("_rerank") else 0)
    for question in questions:
        retriever = retriever_cache.get(question.paper_id)
        ranked_results = retriever.search(question.question, search_k)
        retrieved_chunks = [result.chunk for result in ranked_results[: args.retrieval_k]]
        prompt = build_prompt(
            question,
            retrieved_chunks,
            prompt_style=args.prompt_style,
            max_context_tokens=args.max_context_tokens,
        )
        raw_output = generator.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        prediction = build_generation_prediction(
            question,
            prompt,
            raw_output,
            model_name=args.generation_model,
        )
        verification_payload = None
        if citation_scorer is not None:
            prediction, verification_payload = verify_citation_prediction(
                question.question,
                prompt,
                prediction,
                retrieved_chunks,
                citation_scorer,
                generation_output=raw_output,
                threshold=args.citation_verifier_threshold,
            )
        metrics = evaluate_generation_prediction(question, prediction, retrieved_chunks)
        if verification_payload is not None:
            metrics = replace(
                metrics,
                nli_citation_precision=float(verification_payload["nli_citation_precision"]),
                nli_cited_sentence_count=int(verification_payload["nli_cited_sentence_count"]),
                nli_supported_sentence_count=int(verification_payload["nli_supported_sentence_count"]),
            )
        question_metrics.append(metrics)
        question_payload = {
            "question_id": question.question_id,
            "paper_id": question.paper_id,
            "question": question.question,
            "prediction": serialize_generation_prediction(prediction),
            "metrics": serialize_generation_metrics(metrics),
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved_chunks],
        }
        if verification_payload is not None:
            question_payload["nli_claim_results"] = verification_payload["claim_results"]
        question_payloads.append(question_payload)

    summary = aggregate_generation_metrics(question_metrics)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation" if args.split == "dev" else args.split,
        "strategy": args.strategy,
        "retrieval_method": args.retrieval_method,
        "prompt_style": args.prompt_style,
        "retrieval_k": args.retrieval_k,
        "retrieval_depth": args.retrieval_depth,
        "dense_model": args.dense_model if "dense" in args.retrieval_method or "hybrid" in args.retrieval_method else None,
        "dense_device": args.dense_device,
        "reranker_model": args.reranker_model if args.retrieval_method.endswith("_rerank") else None,
        "reranker_device": args.reranker_device,
        "generation_model": args.generation_model,
        "citation_verify": args.citation_verify,
        "citation_verifier_model": args.citation_verifier_model if args.citation_verify else None,
        "citation_verifier_device": args.citation_verifier_device if args.citation_verify else None,
        "citation_verifier_threshold": args.citation_verifier_threshold if args.citation_verify else None,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_context_tokens": args.max_context_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "processed_manifest": manifest,
        "metrics": summary,
        "questions": question_payloads,
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
        / "generation"
        / ("validation" if args.split == "dev" else args.split)
        / args.strategy
        / f"{args.retrieval_method}.{args.prompt_style}.{timestamp}.json"
    )


if __name__ == "__main__":
    main()
