from pathlib import Path

from qasper_rag.artifacts import (
    group_chunks_by_paper,
    load_processed_chunks,
    load_processed_manifest,
    load_processed_questions,
)
from qasper_rag.processing import process_qasper_split_file


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_qasper.json"


def test_load_processed_artifacts_round_trip(tmp_path: Path) -> None:
    process_qasper_split_file(FIXTURE_PATH, tmp_path)

    manifest = load_processed_manifest(tmp_path / "manifest.json")
    chunks = load_processed_chunks(tmp_path / "chunks.section.jsonl")
    questions = load_processed_questions(tmp_path / "questions.jsonl")

    assert manifest["paper_count"] == 1
    assert len(chunks) == 4
    assert len(questions) == 3
    grouped = group_chunks_by_paper(chunks)
    assert list(grouped) == ["paper-1"]
