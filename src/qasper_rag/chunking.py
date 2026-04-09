from __future__ import annotations

from collections.abc import Sequence
import re

from .schema import Chunk, QasperPaper, SectionPath, SourceType, TextUnit

_TOKEN_RE = re.compile(r"\S+")


def whitespace_token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _TOKEN_RE.finditer(text)]


def whitespace_token_count(text: str) -> int:
    return len(whitespace_token_spans(text))


def split_text_windows(text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
    _validate_chunk_params(max_tokens, overlap_tokens)
    spans = whitespace_token_spans(text)
    if not spans:
        return []

    step = max_tokens - overlap_tokens
    windows: list[str] = []
    start_index = 0
    while start_index < len(spans):
        end_index = min(start_index + max_tokens, len(spans))
        start_char = spans[start_index][0]
        end_char = spans[end_index - 1][1]
        windows.append(text[start_char:end_char].strip())
        if end_index == len(spans):
            break
        start_index += step
    return windows


def _validate_chunk_params(max_tokens: int, overlap_tokens: int) -> None:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative.")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")


def _unique_preserving_order(items: Sequence[SectionPath | SourceType]) -> tuple[SectionPath | SourceType, ...]:
    ordered: list[SectionPath | SourceType] = []
    for item in items:
        if item not in ordered:
            ordered.append(item)
    return tuple(ordered)


class BaseChunker:
    strategy = "base"

    def __init__(
        self,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        include_abstract: bool = True,
        include_figures_and_tables: bool = False,
    ) -> None:
        _validate_chunk_params(max_tokens, overlap_tokens)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.include_abstract = include_abstract
        self.include_figures_and_tables = include_figures_and_tables

    def chunk_paper(self, paper: QasperPaper) -> list[Chunk]:
        raise NotImplementedError

    def _prepare_units(self, paper: QasperPaper) -> list[TextUnit]:
        return paper.iter_text_units(
            include_abstract=self.include_abstract,
            include_figures_and_tables=self.include_figures_and_tables,
        )

    def _build_chunk(
        self,
        *,
        paper: QasperPaper,
        chunk_index: int,
        units: Sequence[TextUnit],
        text: str,
        metadata: dict[str, object] | None = None,
    ) -> Chunk:
        token_count = whitespace_token_count(text)
        section_paths = _unique_preserving_order([unit.section_path for unit in units])
        source_types = _unique_preserving_order([unit.source_type for unit in units])
        return Chunk(
            chunk_id=f"{paper.paper_id}:{self.strategy}:{chunk_index}",
            paper_id=paper.paper_id,
            strategy=self.strategy,
            text=text,
            token_count=token_count,
            unit_ids=tuple(unit.unit_id for unit in units),
            section_paths=tuple(section_paths),  # type: ignore[arg-type]
            source_types=tuple(source_types),  # type: ignore[arg-type]
            metadata=metadata or {},
        )

    def _split_oversized_unit(self, paper: QasperPaper, chunk_index: int, unit: TextUnit) -> list[Chunk]:
        fragments = split_text_windows(
            unit.text,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        chunks: list[Chunk] = []
        for fragment_index, fragment in enumerate(fragments):
            chunks.append(
                self._build_chunk(
                    paper=paper,
                    chunk_index=chunk_index + fragment_index,
                    units=[unit],
                    text=fragment,
                    metadata={"fragment_index": fragment_index, "fragmented_unit_id": unit.unit_id},
                )
            )
        return chunks

    def _chunk_unit_sequence(
        self,
        paper: QasperPaper,
        units: Sequence[TextUnit],
        *,
        starting_index: int = 0,
    ) -> list[Chunk]:
        if not units:
            return []

        unit_token_counts = {unit.unit_id: whitespace_token_count(unit.text) for unit in units}
        chunks: list[Chunk] = []
        current_units: list[TextUnit] = []
        current_token_total = 0
        next_chunk_index = starting_index

        def flush_current() -> None:
            nonlocal current_units, current_token_total, next_chunk_index
            if not current_units:
                return
            chunk_text = "\n\n".join(unit.text for unit in current_units)
            chunks.append(
                self._build_chunk(
                    paper=paper,
                    chunk_index=next_chunk_index,
                    units=current_units,
                    text=chunk_text,
                )
            )
            next_chunk_index += 1
            overlap_units = self._tail_units_for_overlap(current_units, unit_token_counts)
            current_units = list(overlap_units)
            current_token_total = sum(unit_token_counts[unit.unit_id] for unit in current_units)

        for unit in units:
            unit_tokens = unit_token_counts[unit.unit_id]
            if unit_tokens == 0:
                continue
            if unit_tokens > self.max_tokens:
                flush_current()
                oversized_chunks = self._split_oversized_unit(paper, next_chunk_index, unit)
                chunks.extend(oversized_chunks)
                next_chunk_index += len(oversized_chunks)
                current_units = []
                current_token_total = 0
                continue

            if current_units and current_token_total + unit_tokens > self.max_tokens:
                flush_current()

            if current_units and current_token_total + unit_tokens > self.max_tokens:
                while current_units and current_token_total + unit_tokens > self.max_tokens:
                    dropped_unit = current_units.pop(0)
                    current_token_total -= unit_token_counts[dropped_unit.unit_id]

            current_units.append(unit)
            current_token_total += unit_tokens

            if current_token_total == self.max_tokens:
                flush_current()

        if current_units:
            chunk_text = "\n\n".join(unit.text for unit in current_units)
            chunks.append(
                self._build_chunk(
                    paper=paper,
                    chunk_index=next_chunk_index,
                    units=current_units,
                    text=chunk_text,
                )
            )

        return chunks

    def _tail_units_for_overlap(
        self,
        units: Sequence[TextUnit],
        unit_token_counts: dict[str, int],
    ) -> tuple[TextUnit, ...]:
        if self.overlap_tokens == 0:
            return tuple()

        overlap_units: list[TextUnit] = []
        token_total = 0
        for unit in reversed(units):
            overlap_units.insert(0, unit)
            token_total += unit_token_counts[unit.unit_id]
            if token_total >= self.overlap_tokens:
                break
        return tuple(overlap_units)


class FixedChunker(BaseChunker):
    strategy = "fixed"

    def chunk_paper(self, paper: QasperPaper) -> list[Chunk]:
        return self._chunk_unit_sequence(paper, self._prepare_units(paper))


class SectionAwareChunker(BaseChunker):
    strategy = "section"

    def chunk_paper(self, paper: QasperPaper) -> list[Chunk]:
        units = self._prepare_units(paper)
        if not units:
            return []

        grouped_units: list[list[TextUnit]] = []
        for unit in units:
            if not grouped_units:
                grouped_units.append([unit])
                continue
            last_group = grouped_units[-1]
            if last_group[-1].section_path == unit.section_path and last_group[-1].source_type == unit.source_type:
                last_group.append(unit)
            else:
                grouped_units.append([unit])

        chunks: list[Chunk] = []
        next_chunk_index = 0
        for group in grouped_units:
            group_chunks = self._chunk_unit_sequence(paper, group, starting_index=next_chunk_index)
            chunks.extend(group_chunks)
            next_chunk_index += len(group_chunks)
        return chunks


class SlidingWindowChunker(BaseChunker):
    strategy = "sliding"

    def chunk_paper(self, paper: QasperPaper) -> list[Chunk]:
        units = self._prepare_units(paper)
        if not units:
            return []

        full_text, unit_ranges = self._compose_text(units)
        token_spans = whitespace_token_spans(full_text)
        if not token_spans:
            return []

        step = self.max_tokens - self.overlap_tokens
        chunks: list[Chunk] = []
        window_start = 0
        chunk_index = 0

        while window_start < len(token_spans):
            window_end = min(window_start + self.max_tokens, len(token_spans))
            start_char = token_spans[window_start][0]
            end_char = token_spans[window_end - 1][1]
            chunk_text = full_text[start_char:end_char].strip()
            overlapping_units = [
                unit
                for unit, unit_start, unit_end in unit_ranges
                if unit_end > start_char and unit_start < end_char
            ]
            chunks.append(
                self._build_chunk(
                    paper=paper,
                    chunk_index=chunk_index,
                    units=overlapping_units,
                    text=chunk_text,
                    metadata={
                        "window_start_token": window_start,
                        "window_end_token": window_end,
                    },
                )
            )

            if window_end == len(token_spans):
                break
            window_start += step
            chunk_index += 1

        return chunks

    def _compose_text(self, units: Sequence[TextUnit]) -> tuple[str, list[tuple[TextUnit, int, int]]]:
        pieces: list[str] = []
        ranges: list[tuple[TextUnit, int, int]] = []
        cursor = 0
        separator = "\n\n"

        for index, unit in enumerate(units):
            if index > 0:
                pieces.append(separator)
                cursor += len(separator)
            start_char = cursor
            pieces.append(unit.text)
            cursor += len(unit.text)
            ranges.append((unit, start_char, cursor))

        return "".join(pieces), ranges
