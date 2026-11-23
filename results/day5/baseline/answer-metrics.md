# Development answer evaluation

Status: complete
Saved judgments: 320/320
Comparable questions: 80/80

| Mode | Correctness | Relevancy | Faithfulness | Faithfulness N | Context precision | Precision N | Abstention rate | Empty-context rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.256 | 0.269 | 1.000 | 22 | 0.715 | 23 | 0.725 | 0.713 |
| vector | 1.000 | 1.000 | 1.000 | 80 | 0.355 | 80 | 0.000 | 0.000 |
| hybrid | 1.000 | 1.000 | 1.000 | 80 | 0.350 | 80 | 0.000 | 0.000 |
| hybrid_rerank | 1.000 | 1.000 | 1.000 | 80 | 0.333 | 80 | 0.000 | 0.000 |

Scores are macro averages across the comparable questions.
Null faithfulness and context precision are excluded.
Their denominators are shown in the table.
Context precision = relevant chunks / returned chunks.
Relevance labels are assistant-reviewed pooled labels.
Scores use a custom same-model LLM judge, not RAGAS.
Calibration: 6/6 sanity checks passed; accuracy remains unproven.
This is development evaluation, not the held-out test.
Retry/parser policy changes are documented in DECISIONS.md.
