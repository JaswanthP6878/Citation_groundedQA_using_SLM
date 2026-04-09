import numpy as np

from qasper_rag.artifacts import ProcessedAnswer, ProcessedChunk, ProcessedQuestion
from qasper_rag.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    retrieve,
)
from qasper_rag.retrieval_eval import aggregate_question_metrics, evaluate_ranked_chunks


class FakeBM25Backend:
    def __init__(self, tokenized_chunks):
        self.tokenized_chunks = tokenized_chunks

    def get_scores(self, query_tokens):
        scores = []
        for tokens in self.tokenized_chunks:
            scores.append(sum(1.0 for token in query_tokens if token in tokens))
        return scores


class FakeEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "banana" in lowered else 0.0,
                    1.0 if "citrus" in lowered or "orange" in lowered else 0.0,
                    1.0 if "support" in lowered else 0.0,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for query, passage in pairs:
            query_lower = query.lower()
            passage_lower = passage.lower()
            score = 0.0
            if "support" in query_lower and "support" in passage_lower:
                score += 10.0
            if "banana" in query_lower and "banana" in passage_lower:
                score += 3.0
            scores.append(score)
        return np.asarray(scores, dtype=np.float32)


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
            text="orange citrus background",
            token_count=3,
            unit_ids=("u2",),
            section_paths=(("Background",),),
            source_types=("section",),
            metadata={},
        ),
        ProcessedChunk(
            chunk_id="c3",
            paper_id="paper-1",
            strategy="section",
            text="banana support evidence passage",
            token_count=4,
            unit_ids=("u3",),
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


def test_bm25_retriever_ranks_by_token_overlap() -> None:
    chunks = make_chunks()
    retriever = BM25Retriever(chunks, bm25_backend=FakeBM25Backend([["banana", "intro"], ["orange"], ["banana", "support"]]))

    results = retriever.search("banana support", 2)

    assert [result.chunk.chunk_id for result in results] == ["c3", "c1"]


def test_dense_retriever_works_with_fake_encoder_without_faiss() -> None:
    chunks = make_chunks()
    retriever = DenseRetriever(chunks, encoder=FakeEncoder(), use_faiss=False)

    results = retriever.search("banana support", 2)

    assert [result.chunk.chunk_id for result in results] == ["c3", "c1"]


def test_hybrid_and_reranker_promote_best_support_chunk() -> None:
    chunks = make_chunks()
    bm25 = BM25Retriever(chunks, bm25_backend=FakeBM25Backend([["banana", "intro"], ["orange"], ["banana", "support"]]))
    dense = DenseRetriever(chunks, encoder=FakeEncoder(), use_faiss=False)
    hybrid = HybridRetriever({"bm25": bm25, "dense": dense}, retrieval_depth=3)
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    rerank_retriever = RerankRetriever(hybrid, reranker, candidate_k=3)

    results = rerank_retriever.search("banana support", 2)

    assert [result.chunk.chunk_id for result in results] == ["c3", "c1"]
    assert retrieve("banana support", chunks, "dense", 1, encoder=FakeEncoder(), use_faiss=False)[0].chunk_id == "c3"


def test_retrieval_metrics_capture_recall_mrr_and_hit_rate() -> None:
    chunks = make_chunks()
    question = make_question()
    ranked_chunks = [chunks[0], chunks[2], chunks[1]]

    metrics = evaluate_ranked_chunks(question, ranked_chunks, recall_k=2, mrr_k=3)
    summary = aggregate_question_metrics([metrics])

    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 0.5
    assert metrics.evidence_hit == 1.0
    assert summary["eligible_question_count"] == 1
    assert summary["evidence_hit_rate"] == 1.0
