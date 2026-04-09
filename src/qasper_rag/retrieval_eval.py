from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import mean

from .artifacts import ProcessedChunk, ProcessedQuestion, iter_gold_evidence

_NORMALIZE_RE = re.compile(r"\s+")


def normalize_match_text(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.strip().lower())


def chunk_matches_evidence(chunk: ProcessedChunk, evidence_text: str) -> bool:
    normalized_chunk = normalize_match_text(chunk.text)
    normalized_evidence = normalize_match_text(evidence_text)
    if not normalized_chunk or not normalized_evidence:
        return False
    return normalized_evidence in normalized_chunk


def collect_gold_evidence(question: ProcessedQuestion) -> tuple[str, ...]:
    return tuple(iter_gold_evidence(question))


@dataclass(frozen=True)
class QuestionRetrievalMetrics:
    question_id: str
    paper_id: str
    evidence_count: int
    eligible: bool
    recall_at_k: float
    mrr: float
    evidence_hit: float


def evaluate_ranked_chunks(
    question: ProcessedQuestion,
    ranked_chunks: list[ProcessedChunk],
    *,
    recall_k: int = 5,
    mrr_k: int | None = 10,
) -> QuestionRetrievalMetrics:
    gold_evidence = collect_gold_evidence(question)
    if not gold_evidence:
        return QuestionRetrievalMetrics(
            question_id=question.question_id,
            paper_id=question.paper_id,
            evidence_count=0,
            eligible=False,
            recall_at_k=0.0,
            mrr=0.0,
            evidence_hit=0.0,
        )

    top_recall_chunks = ranked_chunks[:recall_k]
    matched_evidence = [
        evidence_text
        for evidence_text in gold_evidence
        if any(chunk_matches_evidence(chunk, evidence_text) for chunk in top_recall_chunks)
    ]
    recall_at_k = len(matched_evidence) / len(gold_evidence)
    evidence_hit = 1.0 if matched_evidence else 0.0

    rank_cutoff = mrr_k if mrr_k is not None else len(ranked_chunks)
    reciprocal_rank = 0.0
    for rank, chunk in enumerate(ranked_chunks[:rank_cutoff], start=1):
        if any(chunk_matches_evidence(chunk, evidence_text) for evidence_text in gold_evidence):
            reciprocal_rank = 1.0 / rank
            break

    return QuestionRetrievalMetrics(
        question_id=question.question_id,
        paper_id=question.paper_id,
        evidence_count=len(gold_evidence),
        eligible=True,
        recall_at_k=recall_at_k,
        mrr=reciprocal_rank,
        evidence_hit=evidence_hit,
    )


def aggregate_question_metrics(metrics: list[QuestionRetrievalMetrics]) -> dict[str, float | int]:
    eligible_metrics = [metric for metric in metrics if metric.eligible]
    if not eligible_metrics:
        return {
            "question_count": len(metrics),
            "eligible_question_count": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "evidence_hit_rate": 0.0,
        }

    return {
        "question_count": len(metrics),
        "eligible_question_count": len(eligible_metrics),
        "recall_at_k": mean(metric.recall_at_k for metric in eligible_metrics),
        "mrr": mean(metric.mrr for metric in eligible_metrics),
        "evidence_hit_rate": mean(metric.evidence_hit for metric in eligible_metrics),
    }
