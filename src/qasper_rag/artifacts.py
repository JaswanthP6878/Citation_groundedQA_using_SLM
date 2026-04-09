from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ProcessedAnswer:
    annotation_id: str
    worker_id: str
    answer_type: str
    canonical_answer: str
    unanswerable: bool
    yes_no: bool | None
    extractive_spans: tuple[str, ...]
    free_form_answer: str
    evidence: tuple[str, ...]
    highlighted_evidence: tuple[str, ...]


@dataclass(frozen=True)
class ProcessedQuestion:
    paper_id: str
    paper_title: str
    question_id: str
    question: str
    question_writer: str
    nlp_background: str
    topic_background: str
    paper_read: str
    search_query: str
    answers: tuple[ProcessedAnswer, ...]


@dataclass(frozen=True)
class ProcessedChunk:
    chunk_id: str
    paper_id: str
    strategy: str
    text: str
    token_count: int
    unit_ids: tuple[str, ...]
    section_paths: tuple[tuple[str, ...], ...]
    source_types: tuple[str, ...]
    metadata: dict[str, Any]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_processed_questions(path: str | Path) -> list[ProcessedQuestion]:
    return [parse_processed_question(record) for record in load_jsonl(path)]


def load_processed_chunks(path: str | Path) -> list[ProcessedChunk]:
    return [parse_processed_chunk(record) for record in load_jsonl(path)]


def load_processed_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_processed_question(record: dict[str, Any]) -> ProcessedQuestion:
    return ProcessedQuestion(
        paper_id=str(record["paper_id"]),
        paper_title=str(record.get("paper_title", "")),
        question_id=str(record["question_id"]),
        question=str(record["question"]),
        question_writer=str(record.get("question_writer", "")),
        nlp_background=str(record.get("nlp_background", "")),
        topic_background=str(record.get("topic_background", "")),
        paper_read=str(record.get("paper_read", "")),
        search_query=str(record.get("search_query", "")),
        answers=tuple(parse_processed_answer(answer) for answer in record.get("answers", [])),
    )


def parse_processed_answer(record: dict[str, Any]) -> ProcessedAnswer:
    return ProcessedAnswer(
        annotation_id=str(record.get("annotation_id", "")),
        worker_id=str(record.get("worker_id", "")),
        answer_type=str(record.get("answer_type", "")),
        canonical_answer=str(record.get("canonical_answer", "")),
        unanswerable=bool(record.get("unanswerable", False)),
        yes_no=record.get("yes_no"),
        extractive_spans=tuple(str(span) for span in record.get("extractive_spans", [])),
        free_form_answer=str(record.get("free_form_answer", "")),
        evidence=tuple(str(text) for text in record.get("evidence", [])),
        highlighted_evidence=tuple(str(text) for text in record.get("highlighted_evidence", [])),
    )


def parse_processed_chunk(record: dict[str, Any]) -> ProcessedChunk:
    return ProcessedChunk(
        chunk_id=str(record["chunk_id"]),
        paper_id=str(record["paper_id"]),
        strategy=str(record["strategy"]),
        text=str(record["text"]),
        token_count=int(record.get("token_count", 0)),
        unit_ids=tuple(str(unit_id) for unit_id in record.get("unit_ids", [])),
        section_paths=tuple(tuple(str(part) for part in path) for path in record.get("section_paths", [])),
        source_types=tuple(str(source_type) for source_type in record.get("source_types", [])),
        metadata=dict(record.get("metadata", {})),
    )


def resolve_processed_split_dir(processed_root: str | Path, split: str) -> Path:
    canonical_split = "validation" if split == "dev" else split
    return Path(processed_root) / canonical_split


def iter_gold_evidence(question: ProcessedQuestion) -> Iterable[str]:
    seen: set[str] = set()
    for answer in question.answers:
        for evidence_text in answer.evidence:
            normalized = evidence_text.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                yield normalized


def group_chunks_by_paper(chunks: Iterable[ProcessedChunk]) -> dict[str, list[ProcessedChunk]]:
    grouped: dict[str, list[ProcessedChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.paper_id, []).append(chunk)
    return grouped
