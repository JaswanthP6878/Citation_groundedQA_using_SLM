from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Sequence

from .config import DEFAULT_CHUNKING_SPECS, ChunkingSpec
from .loader import load_qasper_json
from .schema import QasperAnswer, QasperPaper, QasperQuestion


def build_processed_split(
    papers: Sequence[QasperPaper],
    *,
    chunk_specs: dict[str, ChunkingSpec] | None = None,
) -> dict[str, object]:
    active_specs = chunk_specs or DEFAULT_CHUNKING_SPECS
    paper_records = [paper_to_record(paper) for paper in papers]
    question_records = [question_to_record(paper, question) for paper in papers for question in paper.questions]
    chunk_records = {
        name: [chunk_to_record(chunk) for paper in papers for chunk in spec.build_chunker().chunk_paper(paper)]
        for name, spec in active_specs.items()
    }

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_count": len(papers),
        "question_count": len(question_records),
        "answer_count": sum(len(question.answers) for paper in papers for question in paper.questions),
        "chunk_counts": {name: len(records) for name, records in chunk_records.items()},
        "chunking_specs": {name: asdict(spec) for name, spec in active_specs.items()},
    }

    return {
        "manifest": manifest,
        "papers": paper_records,
        "questions": question_records,
        "chunks": chunk_records,
    }


def process_qasper_split_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_specs: dict[str, ChunkingSpec] | None = None,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    papers = load_qasper_json(input_path)
    processed = build_processed_split(papers, chunk_specs=chunk_specs)
    processed["manifest"]["source_input"] = str(input_path)  # type: ignore[index]

    write_json(output_dir / "manifest.json", processed["manifest"])
    write_jsonl(output_dir / "papers.jsonl", processed["papers"])
    write_jsonl(output_dir / "questions.jsonl", processed["questions"])
    for strategy_name, records in processed["chunks"].items():  # type: ignore[union-attr]
        write_jsonl(output_dir / f"chunks.{strategy_name}.jsonl", records)

    return processed


def paper_to_record(paper: QasperPaper) -> dict[str, object]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "question_count": paper.question_count,
        "answer_count": paper.answer_count,
        "sections": [
            {
                "index": section.index,
                "raw_name": section.raw_name,
                "section_path": list(section.section_path),
                "display_name": section.display_name,
                "paragraphs": list(section.paragraphs),
            }
            for section in paper.sections
        ],
        "figures_and_tables": [asdict(figure) for figure in paper.figures_and_tables],
    }


def question_to_record(paper: QasperPaper, question: QasperQuestion) -> dict[str, object]:
    return {
        "paper_id": paper.paper_id,
        "paper_title": paper.title,
        "question_id": question.question_id,
        "question": question.question,
        "question_writer": question.question_writer,
        "nlp_background": question.nlp_background,
        "topic_background": question.topic_background,
        "paper_read": question.paper_read,
        "search_query": question.search_query,
        "answers": [answer_to_record(answer) for answer in question.answers],
    }


def answer_to_record(answer: QasperAnswer) -> dict[str, object]:
    return {
        "annotation_id": answer.annotation_id,
        "worker_id": answer.worker_id,
        "answer_type": answer.answer_type,
        "canonical_answer": answer.canonical_answer,
        "unanswerable": answer.unanswerable,
        "yes_no": answer.yes_no,
        "extractive_spans": list(answer.extractive_spans),
        "free_form_answer": answer.free_form_answer,
        "evidence": list(answer.evidence),
        "highlighted_evidence": list(answer.highlighted_evidence),
    }


def chunk_to_record(chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "strategy": chunk.strategy,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "unit_ids": list(chunk.unit_ids),
        "section_paths": [list(path) for path in chunk.section_paths],
        "source_types": list(chunk.source_types),
        "metadata": chunk.metadata,
    }


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
