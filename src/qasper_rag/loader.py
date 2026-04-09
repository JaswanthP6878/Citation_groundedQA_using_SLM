from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    FigureOrTable,
    QasperAnswer,
    QasperPaper,
    QasperQuestion,
    Section,
    normalize_whitespace,
    parse_section_path,
)

DATASET_FILENAMES = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


def resolve_qasper_split_path(root: str | Path, split: str) -> Path:
    canonical_split = split.lower()
    if canonical_split not in DATASET_FILENAMES:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {sorted(DATASET_FILENAMES)}.")
    return Path(root) / DATASET_FILENAMES[canonical_split]


def load_qasper_directory(root: str | Path) -> dict[str, list[QasperPaper]]:
    root_path = Path(root)
    loaded: dict[str, list[QasperPaper]] = {}
    for split in ("train", "validation", "test"):
        split_path = resolve_qasper_split_path(root_path, split)
        if split_path.exists():
            loaded[split] = load_qasper_json(split_path)
    return loaded


def load_qasper_json(path: str | Path) -> list[QasperPaper]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [parse_qasper_paper(record) for record in _iter_raw_papers(payload)]


def load_qasper_huggingface(
    split: str | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, list[QasperPaper]] | list[QasperPaper]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets is not installed. Use the raw JSON loader or install `datasets`."
        ) from exc

    if split is not None:
        canonical_split = "validation" if split.lower() == "dev" else split.lower()
        dataset = load_dataset(
            "allenai/qasper",
            split=canonical_split,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return [parse_qasper_paper(example) for example in dataset]

    loaded: dict[str, list[QasperPaper]] = {}
    for split_name in ("train", "validation", "test"):
        dataset = load_dataset(
            "allenai/qasper",
            split=split_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        loaded[split_name] = [parse_qasper_paper(example) for example in dataset]
    return loaded


def parse_qasper_paper(record: dict[str, Any]) -> QasperPaper:
    paper_id = normalize_whitespace(record.get("id"))
    title = normalize_whitespace(record.get("title"))
    abstract = normalize_whitespace(record.get("abstract"))

    sections = tuple(_parse_section(index, section_record) for index, section_record in enumerate(_as_records(record.get("full_text"))))
    questions = tuple(_parse_question(question_record) for question_record in _as_records(record.get("qas")))
    figures_and_tables = tuple(
        FigureOrTable(
            caption=normalize_whitespace(figure_record.get("caption")),
            file=normalize_whitespace(figure_record.get("file")),
        )
        for figure_record in _as_records(record.get("figures_and_tables"))
    )

    return QasperPaper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        sections=sections,
        questions=questions,
        figures_and_tables=figures_and_tables,
    )


def _iter_raw_papers(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for record in payload:
            yield record
        return

    if isinstance(payload, dict):
        if {"id", "title", "qas"}.issubset(payload):
            yield payload
            return
        for paper_id, paper_record in payload.items():
            if not isinstance(paper_record, dict):
                continue
            merged_record = dict(paper_record)
            merged_record.setdefault("id", paper_id)
            yield merged_record
        return

    raise TypeError(f"Unsupported top-level QASPER payload: {type(payload)!r}")


def _as_records(raw_sequence: Any) -> list[dict[str, Any]]:
    if raw_sequence is None:
        return []

    if isinstance(raw_sequence, list):
        return [record for record in raw_sequence if isinstance(record, dict)]

    if isinstance(raw_sequence, dict):
        list_lengths = {len(value) for value in raw_sequence.values() if isinstance(value, list)}
        if not list_lengths:
            return [raw_sequence]
        if len(list_lengths) != 1:
            raise ValueError("Expected nested QASPER fields to have aligned list lengths.")
        length = list_lengths.pop()
        records: list[dict[str, Any]] = []
        for index in range(length):
            record: dict[str, Any] = {}
            for key, value in raw_sequence.items():
                record[key] = value[index] if isinstance(value, list) else value
            records.append(record)
        return records

    raise TypeError(f"Unsupported nested QASPER payload: {type(raw_sequence)!r}")


def _parse_section(index: int, section_record: dict[str, Any]) -> Section:
    raw_name = normalize_whitespace(section_record.get("section_name"))
    paragraphs = tuple(
        paragraph
        for paragraph in (
            normalize_whitespace(paragraph) for paragraph in section_record.get("paragraphs", [])
        )
        if paragraph
    )
    return Section(
        index=index,
        raw_name=raw_name,
        section_path=parse_section_path(raw_name),
        paragraphs=paragraphs,
    )


def _parse_question(question_record: dict[str, Any]) -> QasperQuestion:
    answers = tuple(_parse_answer(answer_record) for answer_record in _as_records(question_record.get("answers")))
    return QasperQuestion(
        question_id=normalize_whitespace(question_record.get("question_id")),
        question=normalize_whitespace(question_record.get("question")),
        question_writer=normalize_whitespace(question_record.get("question_writer")),
        nlp_background=normalize_whitespace(question_record.get("nlp_background")),
        topic_background=normalize_whitespace(question_record.get("topic_background")),
        paper_read=normalize_whitespace(question_record.get("paper_read")),
        search_query=normalize_whitespace(question_record.get("search_query")),
        answers=answers,
    )


def _parse_answer(answer_record: dict[str, Any]) -> QasperAnswer:
    answer_payload = answer_record.get("answer", {})
    if isinstance(answer_payload, list):
        if len(answer_payload) != 1:
            raise ValueError("Expected each QASPER answer record to contain exactly one nested answer object.")
        answer_payload = answer_payload[0]

    extractive_spans = tuple(
        span for span in (normalize_whitespace(span) for span in answer_payload.get("extractive_spans", [])) if span
    )
    free_form_answer = normalize_whitespace(answer_payload.get("free_form_answer"))
    unanswerable = bool(answer_payload.get("unanswerable"))

    yes_no: bool | None
    if unanswerable or extractive_spans or free_form_answer:
        yes_no = None
    else:
        yes_no = bool(answer_payload.get("yes_no"))

    evidence = tuple(
        text for text in (normalize_whitespace(text) for text in answer_payload.get("evidence", [])) if text
    )
    highlighted_evidence = tuple(
        text
        for text in (normalize_whitespace(text) for text in answer_payload.get("highlighted_evidence", []))
        if text
    )

    return QasperAnswer(
        annotation_id=normalize_whitespace(answer_record.get("annotation_id")),
        worker_id=normalize_whitespace(answer_record.get("worker_id")),
        unanswerable=unanswerable,
        extractive_spans=extractive_spans,
        yes_no=yes_no,
        free_form_answer=free_form_answer,
        evidence=evidence,
        highlighted_evidence=highlighted_evidence,
    )
