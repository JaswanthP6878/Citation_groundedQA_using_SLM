from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Protocol

from .artifacts import ProcessedChunk, ProcessedQuestion

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_ANSWER_PREFIX_RE = re.compile(
    r"^(?:(?:final\s+)?answer|response)\s*:\s*",
    flags=re.IGNORECASE,
)
_ANSWER_IS_PREFIX_RE = re.compile(
    r"^(?:the\s+)?answer\s+is\s+",
    flags=re.IGNORECASE,
)
_YES_NO_PREFIX_RE = re.compile(r"^(yes|no)\b", flags=re.IGNORECASE)
_LEADING_FILLER_RE = re.compile(
    r"^(?:according to|based on|from)\s+(?:the\s+)?(?:provided\s+)?context[:,]?\s*",
    flags=re.IGNORECASE,
)
_QUESTION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "use",
        "used",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
)
_UNANSWERABLE_PATTERNS = (
    re.compile(r"\bunanswerable\b", flags=re.IGNORECASE),
    re.compile(r"\binsufficient\s+(?:context|information)\b", flags=re.IGNORECASE),
    re.compile(r"\bnot enough information\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:cannot|can't)\s+(?:be\s+)?(?:determined|answered)\b.*\bcontext\b", flags=re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+(?:provide|specify|mention|state)\b", flags=re.IGNORECASE),
    re.compile(
        r"\bnot\s+(?:provided|specified|mentioned|stated|clear|available)\b.*\b(?:provided\s+)?context\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:specific\s+)?(?:information|details?|figures?|evidence)\b.*\b(?:provided\s+)?context\b",
        flags=re.IGNORECASE,
    ),
)


def format_section_path(section_path: tuple[str, ...]) -> str:
    return " > ".join(section_path) if section_path else "Unknown Section"


def extract_question_terms(question_text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(question_text.lower())
        if token not in _QUESTION_STOPWORDS and len(token) > 1
    }


def is_yes_no_question(question_text: str) -> bool:
    lowered = question_text.strip().lower()
    return lowered.startswith(
        (
            "is ",
            "are ",
            "was ",
            "were ",
            "do ",
            "does ",
            "did ",
            "can ",
            "could ",
            "should ",
            "would ",
            "has ",
            "have ",
            "had ",
        )
    )


@dataclass(frozen=True)
class PromptPackage:
    prompt_style: str
    system_prompt: str
    user_prompt: str
    label_to_chunk_id: dict[int, str]
    retrieved_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class GenerationPrediction:
    question_id: str
    paper_id: str
    prompt_style: str
    model_name: str
    raw_output: str
    answer_text: str
    citation_labels: tuple[int, ...]
    cited_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]


class SupportsGeneration(Protocol):
    def generate(
        self,
        prompt: PromptPackage,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        ...


class HuggingFaceGenerator:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        max_input_tokens: int | None = None,
        attn_implementation: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for local generation.") from exc

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_tokens = max_input_tokens
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device.startswith("cuda") else "auto"

        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        elif self.device.startswith("cuda"):
            model_kwargs["attn_implementation"] = "eager"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        prompt: PromptPackage,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ]
        model_inputs = self._prepare_inputs(messages)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with self._torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        generated_tokens = outputs[:, input_ids.shape[1] :]
        decoded = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        del outputs
        del generated_tokens
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()
        return decoded

    def _prepare_inputs(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = _render_fallback_chat(messages)

        tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if self.max_input_tokens is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = self.max_input_tokens
        encoded = self.tokenizer(rendered, **tokenizer_kwargs)
        return {key: value.to(self.device) for key, value in encoded.items()}


def _render_fallback_chat(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message["role"].capitalize()
        lines.append(f"{role}:\n{message['content']}")
    lines.append("Assistant:\n")
    return "\n\n".join(lines)


def build_prompt(
    question: ProcessedQuestion,
    chunks: list[ProcessedChunk],
    *,
    prompt_style: str = "baseline",
    max_context_tokens: int | None = None,
) -> PromptPackage:
    prompt_chunks_source = prepare_prompt_chunks(
        question.question,
        chunks,
        max_context_tokens=max_context_tokens,
    )
    yes_no_question = is_yes_no_question(question.question)
    answer_instruction = (
        "If the question is yes/no, answer with exactly 'yes' or 'no'."
        if yes_no_question
        else "Return only the answer."
    )
    prompt_chunks = format_prompt_chunks(prompt_chunks_source)
    context_block = "\n\n".join(prompt_chunks) if prompt_chunks else "No retrieved context was provided."
    label_to_chunk_id = {index + 1: chunk.chunk_id for index, chunk in enumerate(prompt_chunks_source)}
    retrieved_chunk_ids = tuple(chunk.chunk_id for chunk in prompt_chunks_source)

    if prompt_style == "baseline":
        system_prompt = (
            "You answer questions about scientific papers using only the provided context. "
            "Answer in one short sentence when possible. "
            "If the context is insufficient, answer with exactly 'unanswerable' and nothing else."
        )
        user_prompt = (
            f"Paper title: {question.paper_title}\n"
            f"Question: {question.question}\n\n"
            "Task:\n"
            f"- {answer_instruction}\n"
            "- Use only the provided context.\n"
            "- Keep the answer concise.\n"
            "- If the context is insufficient, answer with exactly 'unanswerable'.\n\n"
            "Context:\n"
            f"{context_block}\n\n"
            "Answer:"
        )
    elif prompt_style == "citation_forcing":
        system_prompt = (
            "You answer questions about scientific papers using only the provided context. "
            "Answer in one short sentence when possible. "
            "Every factual sentence must end with one or more context ids like [1] or [2, 3]. "
            "Use only citation ids that appear in the provided context. "
            "Do not write labels like 'Context', 'Source', or explanatory notes around citations. "
            "If the context is insufficient, answer with exactly 'unanswerable' and nothing else."
        )
        citation_answer_instruction = answer_instruction
        if yes_no_question:
            citation_answer_instruction = (
                "If the question is yes/no, answer with exactly 'yes' or 'no' and cite that same short sentence."
            )
        user_prompt = (
            f"Paper title: {question.paper_title}\n"
            f"Question: {question.question}\n\n"
            "Task:\n"
            f"- {citation_answer_instruction}\n"
            "- Use only the provided context.\n"
            "- Keep the answer concise.\n"
            "- Every factual sentence must end with citations like [1] or [1, 2].\n"
            "- Use only citation ids that appear in the context block.\n"
            "- If the context is insufficient, answer with exactly 'unanswerable'.\n\n"
            "Context:\n"
            f"{context_block}\n\n"
            "Answer:"
        )
    else:
        raise ValueError(f"Unsupported prompt style: {prompt_style}")

    return PromptPackage(
        prompt_style=prompt_style,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        label_to_chunk_id=label_to_chunk_id,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )


def format_prompt_chunks(chunks: list[ProcessedChunk]) -> list[str]:
    formatted = []
    for index, chunk in enumerate(chunks, start=1):
        section = format_section_path(chunk.section_paths[0]) if chunk.section_paths else "Unknown Section"
        formatted.append(f"[{index}] {section}\n{chunk.text}")
    return formatted


def prepare_prompt_chunks(
    question_text: str,
    chunks: list[ProcessedChunk],
    *,
    max_context_tokens: int | None,
    max_sentences_per_chunk: int = 2,
    max_chunk_tokens: int = 96,
) -> list[ProcessedChunk]:
    compressed_chunks = [
        compress_chunk_for_question(
            question_text,
            chunk,
            max_sentences=max_sentences_per_chunk,
            max_tokens=max_chunk_tokens,
        )
        for chunk in chunks
    ]
    return truncate_chunks_to_token_budget(compressed_chunks, max_context_tokens)


def compress_chunk_for_question(
    question_text: str,
    chunk: ProcessedChunk,
    *,
    max_sentences: int = 2,
    max_tokens: int = 96,
) -> ProcessedChunk:
    normalized_text = " ".join(chunk.text.split()).strip()
    if not normalized_text:
        return chunk

    sentences = split_chunk_sentences(normalized_text)
    if len(sentences) <= max_sentences and len(normalized_text.split()) <= max_tokens:
        return chunk

    question_terms = extract_question_terms(question_text)
    selected_sentences = select_relevant_sentences(
        sentences,
        question_terms,
        max_sentences=max_sentences,
    )
    compressed_text = " ".join(selected_sentences).strip()
    metadata = dict(chunk.metadata)
    metadata["compressed_for_prompt"] = True
    compressed_chunk = replace(
        chunk,
        text=compressed_text,
        token_count=len(compressed_text.split()),
        metadata=metadata,
    )
    return truncate_chunk_text(compressed_chunk, max_tokens)


def split_chunk_sentences(text: str) -> list[str]:
    sentences = [
        " ".join(sentence.split()).strip()
        for sentence in _SENTENCE_SPLIT_RE.split(text)
        if sentence.strip()
    ]
    return sentences or [text]


def select_relevant_sentences(
    sentences: list[str],
    question_terms: set[str],
    *,
    max_sentences: int,
) -> list[str]:
    scored: list[tuple[int, int, int, int]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = set(_WORD_RE.findall(sentence.lower()))
        overlap = len(question_terms & sentence_terms)
        token_count = len(sentence.split())
        scored.append((overlap, -token_count, -index, index))

    top_sentences = sorted(scored, reverse=True)[:max_sentences]
    selected_indices = sorted(item[3] for item in top_sentences)
    selected = [sentences[index] for index in selected_indices]

    if question_terms and all(len(question_terms & set(_WORD_RE.findall(sentence.lower()))) == 0 for sentence in selected):
        return sentences[:1]
    return selected


def truncate_chunks_to_token_budget(
    chunks: list[ProcessedChunk],
    max_context_tokens: int | None,
) -> list[ProcessedChunk]:
    if max_context_tokens is None:
        return chunks

    selected: list[ProcessedChunk] = []
    used_tokens = 0
    for chunk in chunks:
        chunk_tokens = max(chunk.token_count, len(chunk.text.split()))
        remaining = max_context_tokens - used_tokens
        if remaining <= 0:
            break

        if chunk_tokens <= remaining:
            selected.append(chunk)
            used_tokens += chunk_tokens
            continue

        if not selected:
            selected.append(truncate_chunk_text(chunk, remaining))
        break
    return selected


def truncate_chunk_text(chunk: ProcessedChunk, max_tokens: int) -> ProcessedChunk:
    if max_tokens <= 0:
        return replace(chunk, text="", token_count=0)

    words = chunk.text.split()
    if len(words) <= max_tokens:
        return chunk

    truncated_words = words[:max_tokens]
    truncated_text = " ".join(truncated_words)
    metadata = dict(chunk.metadata)
    metadata["truncated_for_prompt"] = True
    return replace(
        chunk,
        text=truncated_text,
        token_count=len(truncated_words),
        metadata=metadata,
    )


def extract_citation_labels(text: str) -> tuple[int, ...]:
    labels: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        for raw_value in match.group(1).split(","):
            label = int(raw_value.strip())
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return tuple(labels)


def strip_citation_markers(text: str) -> str:
    stripped = _CITATION_RE.sub("", text)
    collapsed = " ".join(stripped.split()).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", collapsed)


def compact_generation_output(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    first_block = _PARAGRAPH_SPLIT_RE.split(stripped, maxsplit=1)[0].strip()
    single_line = " ".join(first_block.splitlines()).strip()
    return single_line


def normalize_generation_answer(
    question_text: str,
    answer_text: str,
    *,
    prompt_style: str = "baseline",
    cited_chunk_count: int = 0,
) -> str:
    normalized = " ".join(answer_text.split()).strip()
    if not normalized:
        return ""

    normalized = _ANSWER_PREFIX_RE.sub("", normalized)
    normalized = _ANSWER_IS_PREFIX_RE.sub("", normalized)
    normalized = _LEADING_FILLER_RE.sub("", normalized)
    normalized = normalized.strip(" \t\r\n-:")
    lowered = normalized.lower()

    if any(pattern.search(lowered) for pattern in _UNANSWERABLE_PATTERNS):
        return "unanswerable"

    if is_yes_no_question(question_text):
        match = _YES_NO_PREFIX_RE.match(lowered)
        if match:
            return match.group(1)

    if prompt_style == "citation_forcing" and cited_chunk_count == 0:
        return "unanswerable"

    return normalized


def build_generation_prediction(
    question: ProcessedQuestion,
    prompt: PromptPackage,
    raw_output: str,
    *,
    model_name: str,
) -> GenerationPrediction:
    cleaned_output = compact_generation_output(raw_output)
    citation_labels = extract_citation_labels(cleaned_output)
    cited_chunk_ids = tuple(
        prompt.label_to_chunk_id[label]
        for label in citation_labels
        if label in prompt.label_to_chunk_id
    )
    answer_text = normalize_generation_answer(
        question.question,
        strip_citation_markers(cleaned_output),
        prompt_style=prompt.prompt_style,
        cited_chunk_count=len(cited_chunk_ids),
    )
    return GenerationPrediction(
        question_id=question.question_id,
        paper_id=question.paper_id,
        prompt_style=prompt.prompt_style,
        model_name=model_name,
        raw_output=cleaned_output,
        answer_text=answer_text,
        citation_labels=citation_labels,
        cited_chunk_ids=cited_chunk_ids,
        retrieved_chunk_ids=prompt.retrieved_chunk_ids,
    )


def serialize_generation_prediction(prediction: GenerationPrediction) -> dict[str, Any]:
    return {
        "question_id": prediction.question_id,
        "paper_id": prediction.paper_id,
        "prompt_style": prediction.prompt_style,
        "model_name": prediction.model_name,
        "raw_output": prediction.raw_output,
        "answer_text": prediction.answer_text,
        "citation_labels": list(prediction.citation_labels),
        "cited_chunk_ids": list(prediction.cited_chunk_ids),
        "retrieved_chunk_ids": list(prediction.retrieved_chunk_ids),
    }
