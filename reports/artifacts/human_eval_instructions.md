# Human Evaluation Instructions

Use this rubric with:

- [human_eval_sample.csv](/Users/zukhriddin/Downloads/CS505_NLP/NLP_research_project/reports/artifacts/human_eval_sample.csv)
- [human_eval_sample.md](/Users/zukhriddin/Downloads/CS505_NLP/NLP_research_project/reports/artifacts/human_eval_sample.md)
- [human_eval_template_reviewer_1.csv](/Users/zukhriddin/Downloads/CS505_NLP/NLP_research_project/reports/artifacts/human_eval_template_reviewer_1.csv)
- [human_eval_template_reviewer_2.csv](/Users/zukhriddin/Downloads/CS505_NLP/NLP_research_project/reports/artifacts/human_eval_template_reviewer_2.csv)

Goal:

- Compare the baseline Phi system against the citation-forcing Phi system on a small, fixed sample.
- Focus on answer usefulness and grounding, not just lexical overlap.

Recommended sample size:

- `20–30` questions total
- at least `2` reviewers if possible

For each example, score both answers independently on the following dimensions:

## What the Enriched Columns Mean

- `gold_answers`: canonical answer strings from QASPER. Use these first for correctness.
- `gold_evidence`: gold supporting evidence from QASPER. Use this as the strongest reference for what the paper actually says.
- `baseline_retrieved_context`: the baseline system's retrieved chunks, labeled in prompt order.
- `candidate_retrieved_context`: the citation system's retrieved chunks, labeled in prompt order.
- `candidate_citation_labels`: the prompt labels the model cited, such as `2` in `...[2]`.
- `candidate_cited_context`: the exact retrieved chunk text for the cited labels. Use this to judge whether `[2]` or `[3]` really supports the answer.
- `candidate_nli_claim_summary`: automatic NLI-style support checks for the cited claim sentences. Use this as a hint, not as the final judgment.

Recommended review order:

1. Read the question.
2. Check `gold_answers` and `gold_evidence`.
3. Judge `baseline_answer` against the gold information and `baseline_retrieved_context`.
4. Judge `candidate_answer` against the gold information and `candidate_cited_context`.
5. Use `candidate_nli_claim_summary` only to double-check borderline cases.

## 1. Answer Correctness

Scale:

- `2`: clearly correct
- `1`: partially correct / incomplete
- `0`: incorrect or misleading

Question to ask:

- Does the answer correctly address the question asked?

## 2. Grounding / Support

Scale:

- `2`: directly supported by the paper context / cited evidence
- `1`: plausibly supported but not cleanly grounded
- `0`: unsupported, speculative, or contradicted

Question to ask:

- Is the answer actually justified by the available evidence?

## 3. Citation Quality

Use only for the citation-forcing answer.

Scale:

- `2`: citations are relevant and support the claim
- `1`: citations are partly relevant but weak / incomplete
- `0`: citations are wrong, irrelevant, or do not support the claim

Question to ask:

- Do the cited passages justify the statement being made?

## 4. Preference

Choose one:

- `baseline`
- `citation`
- `tie`

Question to ask:

- If you had to keep one answer for a user-facing system, which one would you keep?

## Suggested Annotation Columns

If you create a spreadsheet, use these columns:

- `question_id`
- `bucket`
- `question`
- `gold_answers`
- `gold_evidence`
- `baseline_answer`
- `baseline_retrieved_context`
- `candidate_answer`
- `candidate_cited_context`
- `candidate_nli_claim_summary`
- `baseline_correctness`
- `citation_correctness`
- `baseline_grounding`
- `citation_grounding`
- `citation_quality`
- `preferred_system`
- `notes`

## Notes for Reviewers

- If both answers are poor, still choose the less harmful one in `preferred_system`, or mark `tie` if they fail equally.
- Penalize confident unsupported claims heavily.
- Short answers like `yes`, `no`, or `unanswerable` are acceptable when they are actually correct.
- Do not reward verbosity by itself.

## Expected Outcome

This human evaluation is intended to answer:

- whether citation forcing improves trustworthiness enough to justify lower automatic scores
- whether the baseline system is still preferable overall
- which error types matter most for the final report
