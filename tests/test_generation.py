from qasper_rag.artifacts import ProcessedAnswer, ProcessedChunk, ProcessedQuestion
from qasper_rag.generation import (
    build_generation_prediction,
    build_prompt,
    compact_generation_output,
    compress_chunk_for_question,
    extract_citation_labels,
    extract_question_terms,
    normalize_generation_answer,
    strip_citation_markers,
    truncate_chunks_to_token_budget,
)
from qasper_rag.generation_eval import (
    aggregate_generation_metrics,
    collect_gold_answers,
    evaluate_generation_prediction,
    expand_multi_reference_pairs,
    max_group_scores,
    token_f1_score,
)


def make_chunks():
    return [
        ProcessedChunk(
            chunk_id="c1",
            paper_id="paper-1",
            strategy="section",
            text="banana intro passage",
            token_count=3,
            unit_ids=("u1",),
            section_paths=(("Introduction",),),
            source_types=("section",),
            metadata={},
        ),
        ProcessedChunk(
            chunk_id="c2",
            paper_id="paper-1",
            strategy="section",
            text="banana support evidence passage",
            token_count=4,
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
                evidence=("banana support evidence passage",),
                highlighted_evidence=("banana support",),
            ),
        ),
    )


def make_yes_no_question():
    return ProcessedQuestion(
        paper_id="paper-1",
        paper_title="Paper",
        question_id="q2",
        question="Is the banana support in the results section?",
        question_writer="writer",
        nlp_background="",
        topic_background="",
        paper_read="yes",
        search_query="",
        answers=(
            ProcessedAnswer(
                annotation_id="a2",
                worker_id="w2",
                answer_type="yes",
                canonical_answer="yes",
                unanswerable=False,
                yes_no=True,
                extractive_spans=(),
                free_form_answer="",
                evidence=("banana support evidence passage",),
                highlighted_evidence=("banana support",),
            ),
        ),
    )


def test_build_prompt_formats_context_and_styles() -> None:
    prompt = build_prompt(make_question(), make_chunks(), prompt_style="citation_forcing")

    assert prompt.prompt_style == "citation_forcing"
    assert prompt.label_to_chunk_id == {1: "c1", 2: "c2"}
    assert "Task:" in prompt.user_prompt
    assert "[1] Introduction" in prompt.user_prompt
    assert "[2] Results" in prompt.user_prompt
    assert "Every factual sentence must end with citations" in prompt.user_prompt
    assert "Do not write labels like 'Context'" in prompt.system_prompt


def test_extract_and_strip_citations() -> None:
    text = "Banana support is in results [2, 1]. Another citation [2]."

    assert extract_citation_labels(text) == (2, 1)
    assert strip_citation_markers(text) == "Banana support is in results. Another citation."


def test_compact_output_and_context_budget() -> None:
    compacted = compact_generation_output("First sentence [1].\n\nSecond paragraph should disappear.")
    assert compacted == "First sentence [1]."

    trimmed = truncate_chunks_to_token_budget(make_chunks(), 3)
    assert len(trimmed) == 1
    assert trimmed[0].chunk_id == "c1"
    assert trimmed[0].token_count == 3


def test_normalize_generation_answer_handles_yes_no_and_unanswerable() -> None:
    assert normalize_generation_answer(
        "Is the banana support in the results section?",
        "Answer: Yes, the results section contains the evidence.",
    ) == "yes"
    assert normalize_generation_answer(
        "How many language pairs were evaluated?",
        "The context does not specify the number of language pairs. Answer: unanswerable.",
    ) == "unanswerable"
    assert normalize_generation_answer(
        "What are the other algorithms tested?",
        "Other algorithms tested are not specified in the provided context.",
    ) == "unanswerable"
    assert normalize_generation_answer(
        "What is the baseline?",
        "According to the provided context, the baseline is LASER.",
    ) == "the baseline is LASER."


def test_question_term_and_chunk_compression_prefers_relevant_sentences() -> None:
    chunk = ProcessedChunk(
        chunk_id="c3",
        paper_id="paper-1",
        strategy="section",
        text=(
            "This paper introduces a general training recipe for multilingual models. "
            "The datasets used are Europarl and MultiUN. "
            "Additional optimization details are reported in the appendix."
        ),
        token_count=24,
        unit_ids=("u3",),
        section_paths=(("Experiments",),),
        source_types=("section",),
        metadata={},
    )

    assert extract_question_terms("what datasets did they use?") == {"datasets"}

    compressed = compress_chunk_for_question("what datasets did they use?", chunk, max_sentences=1, max_tokens=20)
    assert compressed.text == "The datasets used are Europarl and MultiUN."
    assert compressed.metadata["compressed_for_prompt"] is True


def test_token_f1_and_gold_answers() -> None:
    question = make_question()

    assert collect_gold_answers(question) == ("banana support",)
    assert token_f1_score("banana support", "banana support") == 1.0
    assert token_f1_score("banana", "banana support") > 0.0


def test_multi_reference_pair_expansion_and_group_max_scoring() -> None:
    predictions = ["banana support", "yes"]
    reference_sets = [("banana", "banana support"), ("yes",)]

    expanded_predictions, expanded_references, group_sizes = expand_multi_reference_pairs(predictions, reference_sets)

    assert expanded_predictions == ["banana support", "banana support", "yes"]
    assert expanded_references == ["banana", "banana support", "yes"]
    assert group_sizes == [2, 1]
    assert max_group_scores([0.2, 0.9, 0.8], group_sizes) == [0.9, 0.8]


def test_generation_metrics_capture_citations_and_hallucination_proxy() -> None:
    question = make_question()
    chunks = make_chunks()
    prompt = build_prompt(question, chunks, prompt_style="citation_forcing")
    prediction = build_generation_prediction(
        question,
        prompt,
        "Banana support appears in the results section [2].",
        model_name="mock-model",
    )

    metrics = evaluate_generation_prediction(question, prediction, chunks)
    summary = aggregate_generation_metrics([metrics])

    assert prediction.answer_text == "Banana support appears in the results section."
    assert prediction.cited_chunk_ids == ("c2",)
    assert metrics.token_f1 > 0.0
    assert metrics.bertscore_f1 == 0.0
    assert metrics.citation_precision == 1.0
    assert metrics.hallucination == 0.0
    assert summary["question_count"] == 1
    assert summary["bertscore_f1"] == 0.0
    assert summary["supported_answer_rate"] == 1.0


def test_build_generation_prediction_normalizes_yes_no_with_citations() -> None:
    question = make_yes_no_question()
    chunks = make_chunks()
    prompt = build_prompt(question, chunks, prompt_style="citation_forcing")
    prediction = build_generation_prediction(
        question,
        prompt,
        "Answer: Yes, the banana support is reported in results [2].",
        model_name="mock-model",
    )

    assert prediction.answer_text == "yes"
    assert prediction.cited_chunk_ids == ("c2",)


def test_build_generation_prediction_marks_uncited_citation_answer_unanswerable() -> None:
    question = make_question()
    chunks = make_chunks()
    prompt = build_prompt(question, chunks, prompt_style="citation_forcing")
    prediction = build_generation_prediction(
        question,
        prompt,
        "Banana support appears in the results section.",
        model_name="mock-model",
    )

    assert prediction.answer_text == "unanswerable"
    assert prediction.cited_chunk_ids == ()


def test_generation_metrics_penalize_unsupported_citations() -> None:
    question = make_question()
    chunks = make_chunks()
    prompt = build_prompt(question, chunks, prompt_style="citation_forcing")
    prediction = build_generation_prediction(
        question,
        prompt,
        "Orange evidence [1].",
        model_name="mock-model",
    )

    metrics = evaluate_generation_prediction(question, prediction, chunks)

    assert metrics.token_f1 == 0.0
    assert metrics.citation_precision == 0.0
    assert metrics.hallucination == 1.0
