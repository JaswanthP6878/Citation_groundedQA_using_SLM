from pathlib import Path

from qasper_rag.loader import load_qasper_json


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_qasper.json"


def test_load_qasper_json_parses_internal_schema() -> None:
    papers = load_qasper_json(FIXTURE_PATH)

    assert len(papers) == 1
    paper = papers[0]

    assert paper.paper_id == "paper-1"
    assert paper.sections[1].section_path == ("Method", "Encoder")
    assert paper.question_count == 3
    assert paper.answer_count == 3
    assert paper.questions[0].answers[0].answer_type == "extractive"
    assert paper.questions[0].answers[0].canonical_answer == "recurrent encoder"
    assert paper.questions[1].answers[0].answer_type == "yes"
    assert paper.questions[2].answers[0].answer_type == "unanswerable"


def test_iter_text_units_can_include_figures() -> None:
    paper = load_qasper_json(FIXTURE_PATH)[0]

    units = paper.iter_text_units(include_abstract=True, include_figures_and_tables=True)

    assert units[0].section_path == ("Abstract",)
    assert any(unit.source_type == "figure_or_table" for unit in units)
