from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

SourceType = Literal["abstract", "section", "figure_or_table"]
SectionPath = tuple[str, ...]

_SECTION_DELIMITER = re.compile(r"\s*:::\s*")


def normalize_whitespace(text: str | None) -> str:
    return " ".join((text or "").split()).strip()


def parse_section_path(raw_name: str | None) -> SectionPath:
    cleaned_name = normalize_whitespace(raw_name)
    if not cleaned_name:
        return ("Untitled Section",)
    parts = [part for part in _SECTION_DELIMITER.split(cleaned_name) if part]
    return tuple(parts) if parts else ("Untitled Section",)


@dataclass(frozen=True)
class FigureOrTable:
    caption: str
    file: str


@dataclass(frozen=True)
class QasperAnswer:
    annotation_id: str
    worker_id: str
    unanswerable: bool
    extractive_spans: tuple[str, ...]
    yes_no: bool | None
    free_form_answer: str
    evidence: tuple[str, ...]
    highlighted_evidence: tuple[str, ...]

    @property
    def answer_type(self) -> str:
        if self.unanswerable:
            return "unanswerable"
        if self.extractive_spans:
            return "extractive"
        if self.free_form_answer:
            return "free_form"
        if self.yes_no is True:
            return "yes"
        if self.yes_no is False:
            return "no"
        return "unknown"

    @property
    def canonical_answer(self) -> str:
        if self.answer_type == "extractive":
            return " ".join(self.extractive_spans)
        if self.answer_type == "free_form":
            return self.free_form_answer
        if self.answer_type in {"yes", "no", "unanswerable"}:
            return self.answer_type
        return ""


@dataclass(frozen=True)
class QasperQuestion:
    question_id: str
    question: str
    question_writer: str
    nlp_background: str
    topic_background: str
    paper_read: str
    search_query: str
    answers: tuple[QasperAnswer, ...]


@dataclass(frozen=True)
class Section:
    index: int
    raw_name: str
    section_path: SectionPath
    paragraphs: tuple[str, ...]

    @property
    def display_name(self) -> str:
        return " > ".join(self.section_path)


@dataclass(frozen=True)
class TextUnit:
    unit_id: str
    paper_id: str
    source_type: SourceType
    section_path: SectionPath
    text: str
    section_index: int | None = None
    paragraph_index: int | None = None


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    strategy: str
    text: str
    token_count: int
    unit_ids: tuple[str, ...]
    section_paths: tuple[SectionPath, ...]
    source_types: tuple[SourceType, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def primary_section_path(self) -> SectionPath:
        return self.section_paths[0] if self.section_paths else tuple()


@dataclass(frozen=True)
class QasperPaper:
    paper_id: str
    title: str
    abstract: str
    sections: tuple[Section, ...]
    questions: tuple[QasperQuestion, ...]
    figures_and_tables: tuple[FigureOrTable, ...]

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def answer_count(self) -> int:
        return sum(len(question.answers) for question in self.questions)

    def iter_text_units(
        self,
        *,
        include_abstract: bool = True,
        include_figures_and_tables: bool = False,
    ) -> list[TextUnit]:
        units: list[TextUnit] = []

        if include_abstract and self.abstract:
            units.append(
                TextUnit(
                    unit_id=f"{self.paper_id}:abstract",
                    paper_id=self.paper_id,
                    source_type="abstract",
                    section_path=("Abstract",),
                    text=self.abstract,
                )
            )

        for section in self.sections:
            for paragraph_index, paragraph in enumerate(section.paragraphs):
                if not paragraph:
                    continue
                units.append(
                    TextUnit(
                        unit_id=f"{self.paper_id}:s{section.index}:p{paragraph_index}",
                        paper_id=self.paper_id,
                        source_type="section",
                        section_path=section.section_path,
                        text=paragraph,
                        section_index=section.index,
                        paragraph_index=paragraph_index,
                    )
                )

        if include_figures_and_tables:
            for figure_index, figure in enumerate(self.figures_and_tables):
                if not figure.caption:
                    continue
                units.append(
                    TextUnit(
                        unit_id=f"{self.paper_id}:f{figure_index}",
                        paper_id=self.paper_id,
                        source_type="figure_or_table",
                        section_path=("Figures and Tables",),
                        text=figure.caption,
                    )
                )

        return units
