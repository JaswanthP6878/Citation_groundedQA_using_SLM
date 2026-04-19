# Human Evaluation Summary

- Reviewer files: human_eval_template_reviewer_1_filled.csv, human_eval_template_reviewer_2_filled.csv
- Questions reviewed: 24
- Reviewers: 2
- Exact preference agreement: 20/24 (83.3%)

## Mean Scores

| System | Correctness | Grounding | Citation Quality |
| --- | ---: | ---: | ---: |
| Baseline | 1.021 | 1.062 | -- |
| Citation | 0.875 | 0.708 | 0.583 |

## Preference Counts

| View | Baseline | Citation | Tie |
| --- | ---: | ---: | ---: |
| Raw judgments | 22 | 19 | 7 |
| Majority by question | 9 | 9 | 6 |

## Majority Preference by Bucket

| Bucket | Baseline | Citation | Tie |
| --- | ---: | ---: | ---: |
| baseline_better | 4 | 1 | 0 |
| citation_better | 0 | 3 | 3 |
| large_disagreement | 4 | 2 | 0 |
| random_fill | 0 | 0 | 1 |
| yes_no | 1 | 3 | 2 |

## Interpretation

Baseline answers receive higher mean correctness and grounding scores, while citation answers receive a lower grounding score and only moderate citation-quality scores. Preference is mixed at the question level: majority vote is evenly split between baseline and citation, with a substantial number of ties.
