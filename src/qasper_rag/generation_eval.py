from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import mean
import string
from typing import Iterable, Sequence

from .artifacts import ProcessedChunk, ProcessedQuestion
from .generation import GenerationPrediction
from .retrieval_eval import chunk_matches_evidence, collect_gold_evidence

_WHITESPACE_RE = re.compile(r"\s+")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer_text(text: str) -> str:
    lowered = text.lower()
    no_articles = _ARTICLES_RE.sub(" ", lowered)
    no_punctuation = no_articles.translate(str.maketrans("", "", string.punctuation))
    return _WHITESPACE_RE.sub(" ", no_punctuation).strip()


def answer_tokens(text: str) -> list[str]:
    normalized = normalize_answer_text(text)
    return normalized.split() if normalized else []


def token_f1_score(prediction: str, gold_answer: str) -> float:
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(gold_answer)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    common = 0
    for token, count in pred_counts.items():
        common += min(count, gold_counts.get(token, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def collect_gold_answers(question: ProcessedQuestion) -> tuple[str, ...]:
    seen: set[str] = set()
    answers: list[str] = []
    for answer in question.answers:
        canonical = answer.canonical_answer.strip()
        if canonical and canonical not in seen:
            seen.add(canonical)
            answers.append(canonical)
    return tuple(answers)


@dataclass(frozen=True)
class QuestionGenerationMetrics:
    question_id: str
    paper_id: str
    token_f1: float
    citation_precision: float
    hallucination: float
    citation_count: int
    cited_support_count: int
    answer_supported: float
    bertscore_f1: float = 0.0


def expand_multi_reference_pairs(
    predictions: Sequence[str],
    reference_sets: Sequence[Sequence[str]],
) -> tuple[list[str], list[str], list[int]]:
    expanded_predictions: list[str] = []
    expanded_references: list[str] = []
    group_sizes: list[int] = []

    for prediction, references in zip(predictions, reference_sets):
        normalized_references = [reference for reference in references if reference]
        if not normalized_references:
            normalized_references = [""]
        group_sizes.append(len(normalized_references))
        expanded_predictions.extend(prediction for _ in normalized_references)
        expanded_references.extend(normalized_references)

    return expanded_predictions, expanded_references, group_sizes


def max_group_scores(scores: Sequence[float], group_sizes: Sequence[int]) -> list[float]:
    grouped_scores: list[float] = []
    offset = 0
    for group_size in group_sizes:
        if group_size <= 0:
            grouped_scores.append(0.0)
            continue
        grouped_scores.append(max(float(score) for score in scores[offset : offset + group_size]))
        offset += group_size
    return grouped_scores


def evaluate_generation_prediction(
    question: ProcessedQuestion,
    prediction: GenerationPrediction,
    retrieved_chunks: list[ProcessedChunk],
) -> QuestionGenerationMetrics:
    gold_answers = collect_gold_answers(question)
    token_f1 = (
        max(token_f1_score(prediction.answer_text, gold_answer) for gold_answer in gold_answers)
        if gold_answers
        else 0.0
    )
    gold_evidence = tuple(collect_gold_evidence(question))

    chunk_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    cited_chunks = [chunk_by_id[chunk_id] for chunk_id in prediction.cited_chunk_ids if chunk_id in chunk_by_id]
    supporting_cited_chunks = [
        chunk
        for chunk in cited_chunks
        if chunk_supports_any_evidence(chunk, gold_evidence)
    ]

    if cited_chunks:
        citation_precision = len(supporting_cited_chunks) / len(cited_chunks)
        support_pool = cited_chunks
    else:
        citation_precision = 0.0
        support_pool = retrieved_chunks

    if gold_evidence:
        answer_supported = 1.0 if any(chunk_supports_any_evidence(chunk, gold_evidence) for chunk in support_pool) else 0.0
    else:
        answer_supported = 1.0 if token_f1 > 0.0 else 0.0

    hallucination = 0.0 if token_f1 > 0.0 and answer_supported > 0.0 else 1.0

    return QuestionGenerationMetrics(
        question_id=question.question_id,
        paper_id=question.paper_id,
        token_f1=token_f1,
        citation_precision=citation_precision,
        hallucination=hallucination,
        citation_count=len(cited_chunks),
        cited_support_count=len(supporting_cited_chunks),
        answer_supported=answer_supported,
        bertscore_f1=0.0,
    )


def chunk_supports_any_evidence(chunk: ProcessedChunk, evidence_texts: Iterable[str]) -> bool:
    return any(chunk_matches_evidence(chunk, evidence_text) for evidence_text in evidence_texts)


def aggregate_generation_metrics(metrics: list[QuestionGenerationMetrics]) -> dict[str, float | int]:
    if not metrics:
        return {
            "question_count": 0,
            "token_f1": 0.0,
            "bertscore_f1": 0.0,
            "citation_precision": 0.0,
            "hallucination_rate": 0.0,
            "citation_rate": 0.0,
            "supported_answer_rate": 0.0,
        }

    return {
        "question_count": len(metrics),
        "token_f1": mean(metric.token_f1 for metric in metrics),
        "bertscore_f1": mean(metric.bertscore_f1 for metric in metrics),
        "citation_precision": mean(metric.citation_precision for metric in metrics),
        "hallucination_rate": mean(metric.hallucination for metric in metrics),
        "citation_rate": mean(1.0 if metric.citation_count else 0.0 for metric in metrics),
        "supported_answer_rate": mean(metric.answer_supported for metric in metrics),
    }


def serialize_generation_metrics(metrics: QuestionGenerationMetrics) -> dict[str, float | int | str]:
    return {
        "question_id": metrics.question_id,
        "paper_id": metrics.paper_id,
        "token_f1": metrics.token_f1,
        "citation_precision": metrics.citation_precision,
        "hallucination": metrics.hallucination,
        "citation_count": metrics.citation_count,
        "cited_support_count": metrics.cited_support_count,
        "answer_supported": metrics.answer_supported,
        "bertscore_f1": metrics.bertscore_f1,
    }
