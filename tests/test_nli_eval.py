from qasper_rag.artifacts import ProcessedChunk
from qasper_rag.nli_eval import build_claim_hypothesis, evaluate_cited_claims, extract_cited_claims


class FakeEntailmentScorer:
    def score_entailment(self, premises, hypotheses, *, batch_size=16):
        scores = []
        for premise, hypothesis in zip(premises, hypotheses):
            if "results section" in premise.lower() and "results section" in hypothesis.lower():
                scores.append(0.95)
            elif "banana support" in premise.lower() and "banana support" in hypothesis.lower():
                scores.append(0.85)
            else:
                scores.append(0.1)
        return scores


def test_build_claim_hypothesis_handles_yes_no() -> None:
    assert build_claim_hypothesis("Is support in results?", "Answer: Yes [1].") == (
        'The answer to the question "Is support in results?" is yes.'
    )


def test_extract_cited_claims_preserves_sentence_level_citations() -> None:
    claims = extract_cited_claims(
        "Where is the banana support?",
        "Banana support appears in the results section [2]. A second unsupported claim [1].",
        {1: "c1", 2: "c2"},
    )

    assert len(claims) == 2
    assert claims[0].cited_chunk_ids == ("c2",)
    assert claims[0].hypothesis_text == "Banana support appears in the results section."


def test_evaluate_cited_claims_scores_supported_sentences() -> None:
    chunk_by_id = {
        "c1": ProcessedChunk(
            chunk_id="c1",
            paper_id="paper-1",
            strategy="section",
            text="The introduction gives background only.",
            token_count=6,
            unit_ids=("u1",),
            section_paths=(("Introduction",),),
            source_types=("section",),
            metadata={},
        ),
        "c2": ProcessedChunk(
            chunk_id="c2",
            paper_id="paper-1",
            strategy="section",
            text="The banana support appears in the results section of the paper.",
            token_count=11,
            unit_ids=("u2",),
            section_paths=(("Results",),),
            source_types=("section",),
            metadata={},
        ),
    }

    analysis = evaluate_cited_claims(
        "Where is the banana support?",
        "Banana support appears in the results section [2]. Unsupported claim [1].",
        {1: "c1", 2: "c2"},
        chunk_by_id,
        FakeEntailmentScorer(),
        threshold=0.5,
    )

    assert analysis["nli_cited_sentence_count"] == 2
    assert analysis["nli_supported_sentence_count"] == 1
    assert analysis["nli_citation_precision"] == 0.5
