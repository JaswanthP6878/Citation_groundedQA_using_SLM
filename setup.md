# NLP_research_project

## Reports

LaTeX source and build instructions for the CS505 milestone reports are in [`reports/`](reports/README.md).

## Reproducible Paths

The repo uses `src/qasper_rag/config.py` to resolve three path classes:

- shared team data: `/projectnb/cs505am/projects/nlp_research_project`
- per-user SCC work: `/projectnb/cs505am/students/$USER/...`
- local fallback paths under `data/` when SCC is not mounted

Shared QASPER artifacts live in:

`/projectnb/cs505am/projects/nlp_research_project/qasper`

The user project root on SCC is detected in this order:

1. `NLP_PROJECT_USER_ROOT` if set
2. `/projectnb/cs505am/students/$USER/zukhriddin_nlp_research_project` if it exists
3. `/projectnb/cs505am/students/$USER/nlp_research_project` if it exists

The shared project root on SCC is detected in this order:

1. `NLP_PROJECT_SHARED_ROOT` if set
2. `/projectnb/cs505am/projects/zukhriddin_nlp_research_project` if it exists
3. `/projectnb/cs505am/projects/nlp_research_project` if it exists

That keeps old and renamed SCC project folders working without hardcoding one path.

To inspect the paths the repo will use:

```bash
python3 scripts/show_project_paths.py
```

## SCC Setup

Recommended `.condarc` on SCC:

```yaml
envs_dirs:
  - /projectnb/cs505am/students/$USER/.conda/envs
  - ~/.conda/envs

pkgs_dirs:
  - /projectnb/cs505am/students/$USER/.conda/pkgs
  - ~/.conda/pkgs

env_prompt: ({name})
```

Create the environment:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

Defaults:

- environment name: `rag_env` unless `ENV_NAME` is set
- Hugging Face cache: `/projectnb/cs505am/students/$USER/.cache/huggingface`
- user project root: detected automatically as described above

The shared SCC dataset path is intentionally under `projects/`, not `materials/`, because `materials/` is not student-writable.

## Data + Chunking

Implemented under `src/qasper_rag/`:

- `loader.py`: raw-file and Hugging Face QASPER loaders
- `schema.py`: normalized paper/question/answer dataclasses
- `chunking.py`: fixed, sliding-window, and section-aware chunkers
- `processing.py`: deterministic processed split builder

Shared deterministic outputs should live in:

`/projectnb/cs505am/projects/nlp_research_project/qasper/processed`

Those artifacts are the team-shared interface for retrieval and generation. Personal model caches, indexes, and run outputs should stay in each student directory.

Useful commands:

```bash
./scripts/prepare_shared_qasper.sh
```

```bash
python3 scripts/build_qasper_processed.py --split all
```

```bash
PYTHONPATH=src python3 scripts/preview_qasper_chunking.py \
  --input /projectnb/cs505am/projects/nlp_research_project/qasper/extracted/qasper-train-v0.3.json \
  --strategy section \
  --paper-limit 2 \
  --show-chunks 2
```

## Task 3 Retrieval

Retrieval reads from:

`/projectnb/cs505am/projects/nlp_research_project/qasper/processed`

and writes user-specific outputs under:

`/projectnb/cs505am/students/$USER/.../runs/retrieval`

Implemented stack:

- BM25
- dense retrieval with `BAAI/bge-small-en-v1.5`
- hybrid RRF
- cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- metrics: Recall@5, MRR@10, evidence hit rate

Current best retrieval setting:

- default retriever for generation: `hybrid`

Example commands:

```bash
python scripts/evaluate_qasper_retrieval.py \
  --split validation \
  --strategy section \
  --method hybrid
```

```bash
python scripts/evaluate_qasper_retrieval.py \
  --split validation \
  --strategy section \
  --method hybrid_rerank \
  --top-k 5 \
  --mrr-k 10 \
  --retrieval-depth 20
```

Batch usage on SCC:

```bash
mkdir -p logs
qsub -N qasper_dense_val -o logs/qasper_dense_val.log scripts/qsub_qasper_retrieval.sh dense validation section
```

## Task 4 Generation

Implemented files:

- `src/qasper_rag/generation.py`
- `src/qasper_rag/generation_eval.py`
- `scripts/evaluate_qasper_generation.py`
- `scripts/qsub_qasper_generation.sh`

Prompt styles:

- `baseline`
- `citation_forcing`

Current generation behavior includes:

- prompt context compression to question-relevant sentences
- exact normalization for `yes`, `no`, and `unanswerable` when the model clearly intends them
- citation parsing and citation-to-chunk mapping
- stricter citation-mode fallback for uncited answers

Interactive example:

```bash
python scripts/evaluate_qasper_generation.py \
  --split validation \
  --strategy section \
  --retrieval-method hybrid \
  --prompt-style baseline \
  --generation-model microsoft/Phi-3.5-mini-instruct
```

SCC batch example:

```bash
qsub -N qasper_gen_val -o logs/qasper_gen_val.log \
  scripts/qsub_qasper_generation.sh hybrid citation_forcing validation section
```

Reliable sampled SCC run:

```bash
qsub -N qasper_gen_sample -o logs/qasper_gen_sample.log \
  scripts/qsub_qasper_generation.sh hybrid citation_forcing validation section 5 20 100 2048 1200 128
```

The positional arguments after retrieval depth are:

1. `question_limit`
2. `max_input_tokens`
3. `max_context_tokens`
4. `max_new_tokens`

That avoids the SCC issue where `QUESTION_LIMIT=... qsub ...` in the submit shell does not reliably propagate into the batch job unless `-v` is used.

## SCC GPU Notes

Empirically verified on `academic-gpu` with `microsoft/Phi-3.5-mini-instruct`:

- queue: `academic-gpu`
- resources that worked: `1 GPU`, `4 omp slots`, `2h`
- current safe generation settings:
  - `--max-input-tokens 2048`
  - `--max-context-tokens 1200`
  - `--max-new-tokens 128`
- `attn_implementation='eager'` is used on CUDA because `flash-attn` is not installed on the current SCC setup

If you increase the context budget substantially, expect a higher OOM risk on SCC GPUs.

## Current Validation Snapshot

Best full-validation generation runs so far on `section` chunks with `hybrid` retrieval:

- baseline:
  - token F1 `0.3187`
  - hallucination rate `0.4657`
  - supported answer rate `0.7612`
- citation forcing:
  - token F1 `0.3167`
  - citation precision `0.3423`
  - citation rate `0.9254`
  - hallucination rate `0.5572`
  - supported answer rate `0.5741`

Interpretation:

- baseline is currently the better answer-quality setting
- citation forcing is currently the better citation-behavior setting
- the next high-value experiments are citation post-processing, `hybrid_rerank` generation, and then Qwen model comparisons
