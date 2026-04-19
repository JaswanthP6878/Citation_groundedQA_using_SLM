from __future__ import annotations

from dataclasses import replace
from typing import Any

from .artifacts import ProcessedChunk
from .generation import (
    GenerationPrediction,
    PromptPackage,
    compact_generation_output,
    normalize_generation_answer,
    strip_citation_markers,
)
from .nli_eval import SupportsEntailmentScore, evaluate_cited_claims


def verify_citation_prediction(
    question_text: str,
    prompt: PromptPackage,
    prediction: GenerationPrediction,
    retrieved_chunks: list[ProcessedChunk],
    scorer: SupportsEntailmentScore,
    *,
    generation_output: str | None = None,
    threshold: float = 0.5,
    batch_size: int = 16,
) -> tuple[GenerationPrediction, dict[str, Any]]:
    if prompt.prompt_style != "citation_forcing":
        return prediction, _empty_verification_result()
    source_output = generation_output or prediction.raw_output
    if prediction.answer_text == "unanswerable" and source_output.strip().lower() == "unanswerable":
        return prediction, _empty_verification_result()

    chunk_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    verification = evaluate_cited_claims(
        question_text,
        source_output,
        prompt.label_to_chunk_id,
        chunk_by_id,
        scorer,
        threshold=threshold,
        batch_size=batch_size,
    )
    supported_claims = [
        claim
        for claim in verification["claim_results"]
        if claim.get("supported", False)
    ]
    if not supported_claims:
        abstained = replace(
            prediction,
            raw_output="unanswerable",
            answer_text="unanswerable",
            citation_labels=(),
            cited_chunk_ids=(),
        )
        return abstained, verification

    best_claim = max(
        supported_claims,
        key=lambda claim: (
            float(claim.get("max_entailment_score", 0.0)),
            -int(claim.get("sentence_index", 0)),
        ),
    )
    verified_raw_output = compact_generation_output(
        str(best_claim["sentence_text"]),
        prompt_style=prompt.prompt_style,
    )
    citation_labels = tuple(int(label) for label in best_claim.get("citation_labels", []))
    cited_chunk_ids = tuple(str(chunk_id) for chunk_id in best_claim.get("cited_chunk_ids", []))
    answer_text = normalize_generation_answer(
        question_text,
        strip_citation_markers(verified_raw_output),
        prompt_style=prompt.prompt_style,
        cited_chunk_count=len(cited_chunk_ids),
    )
    verified_prediction = replace(
        prediction,
        raw_output=verified_raw_output,
        answer_text=answer_text,
        citation_labels=citation_labels,
        cited_chunk_ids=cited_chunk_ids,
    )
    return verified_prediction, verification


def _empty_verification_result() -> dict[str, Any]:
    return {
        "nli_citation_precision": 0.0,
        "nli_cited_sentence_count": 0,
        "nli_supported_sentence_count": 0,
        "claim_results": [],
    }
