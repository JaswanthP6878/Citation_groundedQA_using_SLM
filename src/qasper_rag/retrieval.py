from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

import numpy as np

from .artifacts import ProcessedChunk

_TOKEN_RE = re.compile(r"\b\w+\b")


def default_tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


class SupportsEncode(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        ...


class SupportsPredict(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        ...


class HuggingFaceEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for the HuggingFace dense encoder.") from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ):
        embeddings = []
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index : start_index + batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self._torch.no_grad():
                outputs = self.model(**batch)
                token_embeddings = outputs.last_hidden_state
                attention_mask = batch["attention_mask"].unsqueeze(-1)
                masked_embeddings = token_embeddings * attention_mask
                pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                if normalize_embeddings:
                    pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu())

        if not embeddings:
            stacked = self._torch.empty((0, 0), dtype=self._torch.float32)
        else:
            stacked = self._torch.cat(embeddings, dim=0)

        if convert_to_numpy:
            return stacked.numpy()
        return stacked


class HuggingFaceCrossEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for the HuggingFace cross encoder.") from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, pairs: list[tuple[str, str]]):
        scores = []
        batch_size = 32
        for start_index in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start_index : start_index + batch_size]
            queries = [query for query, _ in batch_pairs]
            passages = [passage for _, passage in batch_pairs]
            batch = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self._torch.no_grad():
                logits = self.model(**batch).logits.squeeze(-1)
                scores.extend(logits.detach().cpu().tolist())
        return scores


@dataclass(frozen=True)
class RetrievalResult:
    chunk: ProcessedChunk
    score: float
    rank: int
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Retriever:
    backend_name = "bm25"

    def __init__(
        self,
        chunks: list[ProcessedChunk],
        *,
        tokenizer=default_tokenize,
        bm25_backend: Any | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.tokenizer = tokenizer
        self.tokenized_chunks = [self.tokenizer(chunk.text) for chunk in self.chunks]
        if not self.chunks:
            self.backend = bm25_backend
        elif bm25_backend is None:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:
                raise RuntimeError("rank_bm25 is required for BM25 retrieval.") from exc
            self.backend = BM25Okapi(self.tokenized_chunks)
        else:
            self.backend = bm25_backend

    def search(self, query: str, k: int) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        query_tokens = self.tokenizer(query)
        raw_scores = self.backend.get_scores(query_tokens)
        scores = np.asarray(raw_scores, dtype=np.float32)
        ranked_indices = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank + 1,
                backend=self.backend_name,
            )
            for rank, index in enumerate(ranked_indices)
        ]


class DenseRetriever:
    backend_name = "dense"

    def __init__(
        self,
        chunks: list[ProcessedChunk],
        *,
        encoder: SupportsEncode | None = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        normalize_embeddings: bool = True,
        use_faiss: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.chunks = list(chunks)
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.encoder = encoder or self._load_default_encoder(model_name)
        if self.chunks:
            self.chunk_embeddings = self._encode_texts([chunk.text for chunk in self.chunks])
            self.faiss_index = self._build_faiss_index(self.chunk_embeddings) if use_faiss else None
        else:
            self.chunk_embeddings = np.zeros((0, 0), dtype=np.float32)
            self.faiss_index = None

    def search(self, query: str, k: int) -> list[RetrievalResult]:
        if not self.chunks:
            return []

        query_embedding = self._encode_texts([query])
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), min(k, len(self.chunks)))
            result_indices = indices[0].tolist()
            result_scores = scores[0].tolist()
        else:
            similarity_scores = self.chunk_embeddings @ query_embedding[0]
            result_indices = np.argsort(-similarity_scores, kind="stable")[:k].tolist()
            result_scores = [float(similarity_scores[index]) for index in result_indices]

        results: list[RetrievalResult] = []
        for rank, (index, score) in enumerate(zip(result_indices, result_scores), start=1):
            if index < 0:
                continue
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=float(score),
                    rank=rank,
                    backend=self.backend_name,
                )
            )
        return results

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        try:
            encoded = self.encoder.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
        except TypeError:
            encoded = self.encoder.encode(texts)

        array = np.asarray(encoded, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if self.normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            array = array / norms
        return array

    def _build_faiss_index(self, embeddings: np.ndarray):
        try:
            import faiss
        except ImportError:
            return None
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))
        return index

    def _load_default_encoder(self, model_name: str) -> SupportsEncode:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return HuggingFaceEncoder(model_name)
        return SentenceTransformer(model_name)


class HybridRetriever:
    backend_name = "hybrid"

    def __init__(
        self,
        retrievers: dict[str, Any],
        *,
        rrf_k: int = 60,
        retrieval_depth: int | None = None,
    ) -> None:
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.retrieval_depth = retrieval_depth

    def search(self, query: str, k: int) -> list[RetrievalResult]:
        depth = max(k, self.retrieval_depth or k)
        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, ProcessedChunk] = {}
        sources: dict[str, list[str]] = {}

        for backend_name, retriever in self.retrievers.items():
            results = retriever.search(query, depth)
            for result in results:
                chunk_id = result.chunk.chunk_id
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + result.rank)
                chunk_lookup[chunk_id] = result.chunk
                sources.setdefault(chunk_id, []).append(backend_name)

        ranked_chunk_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)[:k]
        return [
            RetrievalResult(
                chunk=chunk_lookup[chunk_id],
                score=float(fused_scores[chunk_id]),
                rank=rank,
                backend=self.backend_name,
                metadata={"sources": sources.get(chunk_id, [])},
            )
            for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)
        ]


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model: SupportsPredict | None = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = model or self._load_default_model(model_name)

    def rerank(self, query: str, results: list[RetrievalResult], *, top_k: int | None = None) -> list[RetrievalResult]:
        if not results:
            return []
        pairs = [(query, result.chunk.text) for result in results]
        scores = np.asarray(self.model.predict(pairs), dtype=np.float32)
        ranked_indices = np.argsort(-scores, kind="stable")
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]
        return [
            RetrievalResult(
                chunk=results[index].chunk,
                score=float(scores[index]),
                rank=rank,
                backend="rerank",
                metadata={"base_backend": results[index].backend, **results[index].metadata},
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]

    def _load_default_model(self, model_name: str) -> SupportsPredict:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return HuggingFaceCrossEncoder(model_name)
        return CrossEncoder(model_name)


class RerankRetriever:
    backend_name = "rerank"

    def __init__(self, base_retriever: Any, reranker: CrossEncoderReranker, *, candidate_k: int = 20) -> None:
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self, query: str, k: int) -> list[RetrievalResult]:
        candidates = self.base_retriever.search(query, max(k, self.candidate_k))
        return self.reranker.rerank(query, candidates, top_k=k)


def build_retriever(
    method: str,
    chunks: list[ProcessedChunk],
    *,
    encoder: SupportsEncode | None = None,
    dense_model_name: str = "BAAI/bge-small-en-v1.5",
    reranker: CrossEncoderReranker | None = None,
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    tokenizer=default_tokenize,
    bm25_backend: Any | None = None,
    use_faiss: bool = True,
    retrieval_depth: int = 20,
    rrf_k: int = 60,
) -> Any:
    normalized_method = method.lower()

    if normalized_method == "bm25":
        return BM25Retriever(chunks, tokenizer=tokenizer, bm25_backend=bm25_backend)

    if normalized_method == "dense":
        return DenseRetriever(
            chunks,
            encoder=encoder,
            model_name=dense_model_name,
            use_faiss=use_faiss,
        )

    if normalized_method == "hybrid":
        return HybridRetriever(
            {
                "bm25": BM25Retriever(chunks, tokenizer=tokenizer, bm25_backend=bm25_backend),
                "dense": DenseRetriever(
                    chunks,
                    encoder=encoder,
                    model_name=dense_model_name,
                    use_faiss=use_faiss,
                ),
            },
            retrieval_depth=retrieval_depth,
            rrf_k=rrf_k,
        )

    if normalized_method.endswith("_rerank"):
        base_method = normalized_method.removesuffix("_rerank")
        base_retriever = build_retriever(
            base_method,
            chunks,
            encoder=encoder,
            dense_model_name=dense_model_name,
            tokenizer=tokenizer,
            bm25_backend=bm25_backend,
            use_faiss=use_faiss,
            retrieval_depth=retrieval_depth,
            rrf_k=rrf_k,
        )
        active_reranker = reranker or CrossEncoderReranker(model_name=reranker_model_name)
        return RerankRetriever(base_retriever, active_reranker, candidate_k=retrieval_depth)

    raise ValueError(f"Unsupported retrieval method: {method}")


def search(
    query: str,
    chunks: list[ProcessedChunk],
    method: str,
    k: int,
    **kwargs: Any,
) -> list[RetrievalResult]:
    retriever = build_retriever(method, chunks, **kwargs)
    return retriever.search(query, k)


def retrieve(
    query: str,
    chunks: list[ProcessedChunk],
    method: str,
    k: int,
    **kwargs: Any,
) -> list[ProcessedChunk]:
    return [result.chunk for result in search(query, chunks, method, k, **kwargs)]
