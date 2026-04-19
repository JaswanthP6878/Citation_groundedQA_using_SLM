from qasper_rag.artifacts import ProcessedAnswer, ProcessedChunk, ProcessedQuestion
from qasper_rag.citation_verifier import verify_citation_prediction
from qasper_rag.generation import build_generation_prediction, build_prompt


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


def make_chunks():
    return [
        ProcessedChunk(
            chunk_id="c1",
            paper_id="paper-1",
            strategy="section",
            text="The introduction gives background only.",
            token_count=5,
            unit_ids=("u1",),
            section_paths=(("Introduction",),),
            source_types=("section",),
            metadata={},
        ),
        ProcessedChunk(
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
    ]


def make_question():
    return ProcessedQuestion(
        paper_id="paper-1",
        paper_title="Paper",
        question_id="q1",
        question="Where is the banana support?",
        question_writer="writer",
        nlp_background="",
        topic_background="",
        paper_read="yes",
        search_query="",
        answers=(
            ProcessedAnswer(
                annotation_id="a1",
                worker_id="w1",
                answer_type="extractive",
                canonical_answer="banana support",
                unanswerable=False,
                yes_no=None,
                extractive_spans=("banana support",),
                free_form_answer="",
                evidence=("The banana support appears in the results section of the paper.",),
                highlighted_evidence=("banana support",),
            ),
        ),
    )


def test_verify_citation_prediction_keeps_best_supported_claim() -> None:
    question = make_question()
    chunks = make_chunks()
    prompt = build_prompt(question, chunks, prompt_style="citation_forcing")
    prediction = build_generation_prediction(
        question,
        prompt,
        "Unsupported claim [1]. Banana support appears in the results section [2].",
        model_name="mock-model",
    )

    verified_prediction, verification = verify_citation_prediction(
        question.question,
        prompt,
        prediction,
        chunks,
        FakeEntailmentScorer(),
        generation_output="Unsupported claim [1]. Banana support appears in the results section [2].",
        threshold=0.5,
    )

    assert verification["nli_cited_sentence_count"] == 2
    assert verification["nli_supported_sentence_count"] == 1
    assert verified_prediction.answer_text == "Banana support appears in the results section."
    assert verified_prediction.citation_labels == (2,)
    assert verified_prediction.cited_chunk_ids == ("c2",)


def test_verify_citation_prediction_abstains_when_no_claim_verifies() -> None:
    question = make_question()
    chunks = make_chunks()
    prompt = build_prompt(question, chunks, prompt_style="citation_forcing")
    prediction = build_generation_prediction(
        question,
        prompt,
        "Orange evidence [1].",
        model_name="mock-model",
    )

    verified_prediction, verification = verify_citation_prediction(
        question.question,
        prompt,
        prediction,
        chunks,
        FakeEntailmentScorer(),
        generation_output="Orange evidence [1].",
        threshold=0.5,
    )

    assert verification["nli_supported_sentence_count"] == 0
    assert verified_prediction.answer_text == "unanswerable"
    assert verified_prediction.raw_output == "unanswerable"
    assert verified_prediction.citation_labels == ()
    assert verified_prediction.cited_chunk_ids == ()
