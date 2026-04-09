from pathlib import Path

from qasper_rag.chunking import FixedChunker, SectionAwareChunker, SlidingWindowChunker
from qasper_rag.loader import load_qasper_json


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_qasper.json"


def load_sample_paper():
    return load_qasper_json(FIXTURE_PATH)[0]


def test_fixed_chunker_preserves_paragraph_units_with_overlap() -> None:
    paper = load_sample_paper()
    chunker = FixedChunker(max_tokens=14, overlap_tokens=4, include_abstract=False)

    chunks = chunker.chunk_paper(paper)

    assert len(chunks) >= 3
    assert chunks[0].unit_ids == ("paper-1:s0:p0", "paper-1:s0:p1")
    assert "paper-1:s0:p1" in chunks[1].unit_ids


def test_section_chunker_does_not_cross_section_boundaries() -> None:
    paper = load_sample_paper()
    chunker = SectionAwareChunker(max_tokens=14, overlap_tokens=4, include_abstract=False)

    chunks = chunker.chunk_paper(paper)

    assert len(chunks) >= 3
    assert all(len(chunk.section_paths) == 1 for chunk in chunks)
    assert chunks[0].section_paths[0] == ("Introduction",)
    assert chunks[1].section_paths[0] == ("Method", "Encoder")


def test_sliding_window_chunker_uses_exact_token_overlap() -> None:
    paper = load_sample_paper()
    chunker = SlidingWindowChunker(max_tokens=10, overlap_tokens=3, include_abstract=False)

    chunks = chunker.chunk_paper(paper)

    assert len(chunks) >= 2
    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()
    assert first_tokens[-3:] == second_tokens[:3]
