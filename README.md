# Citation-Grounded Scientific QA with Local Retrieval-Augmented Generation

A fully **local, privacy-preserving** retrieval-augmented generation (RAG) pipeline that answers questions over scientific papers using a small language model (≤4B parameters). No document ever leaves the machine, and every answer is checked for whether it's actually grounded in the retrieved evidence.

## Why this exists

Hallucination, generating claims the source text doesn't support, is the main failure mode that keeps LLM-based QA out of high-stakes settings. The usual fix, cloud RAG tools (e.g. NotebookLM, hosted OpenAI APIs), requires uploading the corpus to a third party, which is a non-starter under HIPAA/GDPR or for any sensitive internal documents.

This project asks: **how much of a sub-7B model's hallucination problem is actually a retrieval problem, not a generation problem, and can the whole pipeline run privately, entirely on one consumer/lab GPU?**

## What it does

- Indexes and retrieves over **QASPER**: 1,585 NLP papers, 5,049 question–answer pairs with gold evidence spans.
- Runs generation with **Phi-3.5-mini-instruct** (3.8B params) locally, with side-by-side comparisons against Qwen2.5-7B, Qwen3-4B, Llama-3-8B, and Mistral-7B.
- Supports a **citation-forcing** prompt mode where the model must bracket every claim with a reference to a retrieved chunk, and verifies those citations with an NLI entailment check.
- Evaluates with both automatic metrics (token F1, BERTScore, recall/MRR, citation precision, NLI entailment) and a blinded **human evaluation** (2 reviewers, 83.3% agreement).

### Pipeline

```
Paper → Chunking → Retrieval → Generation → Evaluation
        ├─ fixed              ├─ BM25                ├─ baseline prompt       ├─ token F1 / BERTScore
        ├─ sliding-window     ├─ dense (BGE-small)    ├─ citation-forcing      ├─ recall@5 / MRR@10
        └─ section-aware      ├─ hybrid (RRF)         prompt                  ├─ citation precision
                               └─ + cross-encoder rerank                       ├─ NLI citation check
                                                                                └─ human eval
```

## Key results

**Retrieval quality is the bigger lever than prompt engineering.** The best configuration, section-aware chunking + hybrid retrieval + reranking, beats fixed chunking by ~6 points of token F1 and cuts hallucination by nearly 7 points relative to a weaker retrieval setup, and dramatically outperforms generating with no retrieval at all:

| Setting | Token F1 | Hallucination rate | Supported answer rate |
|---|---:|---:|---:|
| No retrieval (parametric-only) | 0.135 | 86.6% | 13.4% |
| Fixed chunks + dense retrieval | 0.306 | 51.3% | 68.9% |
| **Section-aware + hybrid + rerank (best)** | **0.324** | **44.8%** | **76.5%** |

Retrieval ablation (section-aware chunks, validation split):

| Method | Recall@5 | MRR@10 |
|---|---:|---:|
| BM25 | 0.577 | 0.411 |
| Dense (BGE-small) | 0.589 | 0.482 |
| Hybrid (BM25 + dense) | 0.631 | 0.510 |
| Hybrid + cross-encoder rerank | 0.641 | 0.503 |

**Citation forcing doesn't automatically mean grounded answers.** Forcing the model to cite sources raised citation *coverage* but also introduced 790 unsupported citations and 162 new grounding failures on the same 1,005-question set. Visible attribution is not the same as correctness, and the error analysis points to retrieval misses (not generation) as the dominant remaining failure mode.

Full methodology, all ablations, and the human evaluation protocol are in [`reports/final_report.pdf`](reports/final_report.pdf) (proposal: [`reports/proposal.pdf`](reports/proposal.pdf)).

## Tech stack

Python 3.10 · PyTorch · HuggingFace Transformers/Accelerate · Sentence-Transformers · FAISS · rank-bm25 · `microsoft/Phi-3.5-mini-instruct` · `BAAI/bge-small-en-v1.5` · `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Repo structure

```
src/qasper_rag/   # core library: loading, chunking, retrieval, generation, evaluation
scripts/          # eval drivers + SCC batch-job scripts
reports/          # proposal, midway, final report (LaTeX + PDF) and result artifacts
tests/            # unit tests for chunking, retrieval, generation, citation verification
```

## Getting started

```bash
chmod +x setup_env.sh
./setup_env.sh          # creates the `rag_env` conda environment
```

See [`setup.md`](setup.md) for path configuration (local vs. shared cluster). Example evaluation run:

```bash
python scripts/evaluate_qasper_generation.py \
  --split validation \
  --strategy section \
  --retrieval-method hybrid \
  --prompt-style baseline \
  --generation-model microsoft/Phi-3.5-mini-instruct
```

## Authors

Jaswanth Pinnepu, Zukhriddin Fakhriddinov, Atharv Gulati, Aayush Kumar, Boston University

---

Course project for CS505 (Boston University); no formal license is attached.
