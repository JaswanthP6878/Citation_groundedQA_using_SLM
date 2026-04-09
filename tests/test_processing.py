from pathlib import Path

from qasper_rag.processing import build_processed_split, process_qasper_split_file
from qasper_rag.loader import load_qasper_json


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_qasper.json"


def test_build_processed_split_creates_expected_records() -> None:
    papers = load_qasper_json(FIXTURE_PATH)
    processed = build_processed_split(papers)

    assert processed["manifest"]["paper_count"] == 1
    assert processed["manifest"]["question_count"] == 3
    assert processed["manifest"]["chunking_specs"]["fixed"]["max_tokens"] == 256
    assert len(processed["papers"]) == 1
    assert len(processed["questions"]) == 3
    assert processed["chunks"]["section"][0]["chunk_id"].startswith("paper-1:section:")


def test_process_qasper_split_file_writes_jsonl_outputs(tmp_path: Path) -> None:
    process_qasper_split_file(FIXTURE_PATH, tmp_path)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "papers.jsonl").exists()
    assert (tmp_path / "questions.jsonl").exists()
    assert (tmp_path / "chunks.fixed.jsonl").exists()
    assert (tmp_path / "chunks.sliding.jsonl").exists()
    assert (tmp_path / "chunks.section.jsonl").exists()
