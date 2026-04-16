from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol

from .artifacts import ProcessedChunk
from .generation import extract_citation_labels, normalize_generation_answer, strip_citation_markers

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PREFIX_RE = re.compile(r"^(?:(?:final\s+)?answer|response)\s*:\s*", flags=re.IGNORECASE)


class SupportsEntailmentScore(Protocol):
    def score_entailment(
        self,
        premises: list[str],
        hypotheses: list[str],
        *,
        batch_size: int = 16,
    ) -> list[float]:
        ...


@dataclass(frozen=True)
class CitedClaim:
    sentence_index: int
    sentence_text: str
    hypothesis_text: str
    citation_labels: tuple[int, ...]
    cited_chunk_ids: tuple[str, ...]


class HuggingFaceEntailmentScorer:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for NLI citation scoring.") from exc

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.entailment_index = resolve_entailment_index(self.model.config)

    def score_entailment(
        self,
        premises: list[str],
        hypotheses: list[str],
        *,
        batch_size: int = 16,
    ) -> list[float]:
        scores: list[float] = []
        for start_index in range(0, len(premises), batch_size):
            batch_premises = premises[start_index : start_index + batch_size]
            batch_hypotheses = hypotheses[start_index : start_index + batch_size]
            batch = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self._torch.no_grad():
                logits = self.model(**batch).logits
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                if logits.shape[-1] == 1:
                    probabilities = self._torch.sigmoid(logits.squeeze(-1))
                    scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
                else:
                    probabilities = self._torch.softmax(logits, dim=-1)[:, self.entailment_index]
                    scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
        return scores


def resolve_entailment_index(config: Any) -> int:
    label_map = getattr(config, "id2label", None) or {}
    if isinstance(label_map, dict):
        for raw_index, label in label_map.items():
            if "entail" in str(label).lower():
                return int(raw_index)

    reverse_label_map = getattr(config, "label2id", None) or {}
    if isinstance(reverse_label_map, dict):
        for label, raw_index in reverse_label_map.items():
            if "entail" in str(label).lower():
                return int(raw_index)

    num_labels = getattr(config, "num_labels", None)
    if num_labels == 3:
        return 2
    if num_labels == 2:
        return 1
    raise ValueError("Could not infer entailment label index from model config.")


def split_answer_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(normalized) if sentence.strip()]
    return sentences or [normalized]


def build_claim_hypothesis(question_text: str, sentence_text: str) -> str:
    stripped = _PREFIX_RE.sub("", strip_citation_markers(sentence_text)).strip()
    if not stripped:
        return ""
    normalized = normalize_generation_answer(question_text, stripped)
    if normalized == "yes":
        return f'The answer to the question "{question_text}" is yes.'
    if normalized == "no":
        return f'The answer to the question "{question_text}" is no.'
    if normalized == "unanswerable":
        return f'The answer to the question "{question_text}" is unanswerable based on the cited evidence.'
    return stripped


def extract_cited_claims(
    question_text: str,
    raw_output: str,
    label_to_chunk_id: dict[int, str],
) -> list[CitedClaim]:
    claims: list[CitedClaim] = []
    for sentence_index, sentence_text in enumerate(split_answer_sentences(raw_output), start=1):
        citation_labels = extract_citation_labels(sentence_text)
        if not citation_labels:
            continue
        cited_chunk_ids = tuple(
            label_to_chunk_id[label]
            for label in citation_labels
            if label in label_to_chunk_id
        )
        if not cited_chunk_ids:
            continue
        hypothesis_text = build_claim_hypothesis(question_text, sentence_text)
        if not hypothesis_text:
            continue
        claims.append(
            CitedClaim(
                sentence_index=sentence_index,
                sentence_text=sentence_text,
                hypothesis_text=hypothesis_text,
                citation_labels=citation_labels,
                cited_chunk_ids=cited_chunk_ids,
            )
        )
    return claims


def evaluate_cited_claims(
    question_text: str,
    raw_output: str,
    label_to_chunk_id: dict[int, str],
    chunk_by_id: dict[str, ProcessedChunk],
    scorer: SupportsEntailmentScore,
    *,
    threshold: float = 0.5,
    batch_size: int = 16,
) -> dict[str, Any]:
    claims = extract_cited_claims(question_text, raw_output, label_to_chunk_id)
    if not claims:
        return {
            "nli_citation_precision": 0.0,
            "nli_cited_sentence_count": 0,
            "nli_supported_sentence_count": 0,
            "claim_results": [],
        }

    premises: list[str] = []
    hypotheses: list[str] = []
    group_sizes: list[int] = []
    for claim in claims:
        valid_chunk_ids = [chunk_id for chunk_id in claim.cited_chunk_ids if chunk_id in chunk_by_id]
        group_sizes.append(len(valid_chunk_ids))
        premises.extend(chunk_by_id[chunk_id].text for chunk_id in valid_chunk_ids)
        hypotheses.extend(claim.hypothesis_text for _ in valid_chunk_ids)

    entailment_scores = scorer.score_entailment(premises, hypotheses, batch_size=batch_size) if premises else []
    claim_results: list[dict[str, Any]] = []
    offset = 0
    supported_sentence_count = 0
    for claim, group_size in zip(claims, group_sizes):
        claim_scores = entailment_scores[offset : offset + group_size]
        offset += group_size
        max_entailment = max(claim_scores) if claim_scores else 0.0
        supported = bool(claim_scores) and max_entailment >= threshold
        if supported:
            supported_sentence_count += 1
        claim_results.append(
            {
                "sentence_index": claim.sentence_index,
                "sentence_text": claim.sentence_text,
                "hypothesis_text": claim.hypothesis_text,
                "citation_labels": list(claim.citation_labels),
                "cited_chunk_ids": list(claim.cited_chunk_ids),
                "max_entailment_score": float(max_entailment),
                "supported": supported,
            }
        )

    return {
        "nli_citation_precision": supported_sentence_count / len(claims),
        "nli_cited_sentence_count": len(claims),
        "nli_supported_sentence_count": supported_sentence_count,
        "claim_results": claim_results,
    }
