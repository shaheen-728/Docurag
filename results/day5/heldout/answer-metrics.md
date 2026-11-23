# Held-out answer evaluation

Status: complete
Saved judgments: 20/20
Comparable questions: 20/20

| Mode | Correctness | Relevancy | Faithfulness | Faithfulness N | Context precision | Precision N | Abstention rate | Empty-context rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 0.900 | 0.900 | 1.000 | 18 | 0.450 | 20 | 0.100 | 0.000 |

Scores are macro averages across the comparable questions.
Null faithfulness and context precision are excluded.
Their denominators are shown in the table.
Context precision = relevant chunks / returned chunks.
Relevance labels are assistant-reviewed pooled labels.
Scores use a custom same-model LLM judge, not RAGAS.
Calibration: 6/6 sanity checks passed; accuracy remains unproven.
This is the frozen held-out test of the selected configuration.
Retry/parser policy changes are documented in DECISIONS.md.
