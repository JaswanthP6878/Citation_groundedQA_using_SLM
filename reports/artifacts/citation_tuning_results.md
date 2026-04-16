| run | questions | token_f1 | citation_precision | citation_rate | hallucination | supported_answer_rate | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hybrid_rerank.citation_forcing.20260413T014339Z.json` (first 100 overlap) | 100 | 0.3569 | 0.3528 | 0.9274 | 0.5100 | 0.6900 | strongest earlier citation sample used as overlap baseline |
| `hybrid_rerank.citation_forcing.20260414T025317Z.json` | 100 | 0.2976 | 0.3755 | 0.9500 | 0.5200 | 0.6700 | strict one-sentence citation contract |
| `hybrid_rerank.citation_forcing.20260414T215250Z.json` | 100 | 0.3234 | 0.3032 | 0.9300 | 0.5900 | 0.5900 | decimal-safe compactor plus relaxed prompt |
| `hybrid_rerank.citation_forcing.20260414T224113Z.json` | 100 | 0.3288 | 0.3032 | 0.9300 | 0.5800 | 0.6100 | added vacuous-restatement fallback to `unanswerable` |

Conclusion:

- The deterministic post-processing fixes recovered some answer quality after the strict prompt regression.
- The latest run (`20260414T224113Z`) is better than the immediately previous rerun (`20260414T215250Z`) on token F1, hallucination, and support.
- None of the post-full-validation citation tuning variants beat the earlier stronger citation sample on the same 100-question overlap.
- The current project conclusion remains unchanged: `hybrid_rerank + baseline` is the best overall Phi configuration, while citation forcing remains a tradeoff that improves explicit attribution but not overall grounded answer quality.
