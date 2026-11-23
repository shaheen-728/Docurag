# Held-out retrieval evaluation

Questions: 20. Baseline chunking. Vector retrieval. Top k: 5.

| Mode | Hit@5 | MRR@5 | Pooled Recall@5 | Precision@5 |
| --- | ---: | ---: | ---: | ---: |
| vector | 0.950 | 0.950 | 0.938 | 0.450 |

Retrieval hits: 19/20.
Mean API-reported search latency: 755.57 ms.
Mean client-observed search latency: 773.38 ms.

Labels are assistant-reviewed, not human-verified; partial support counts.
Recall uses the top-5 plus generation-source pool, not exhaustive labels.
This pool differs from the chunk-size comparison pool.
Search latencies are not the database-only chunk-size timings.
These are retrieval metrics, not generated-answer correctness scores.
Do not tune the selected configuration using these held-out results.
No model API calls are made by this reporting script.

Run SHA256: faa455b116b1c8b9e458f5e02b12b666acc8e12bcf6decc3674ad2beebab052f
Review SHA256: 50f3f2dc2281f01a90606d1aa9e5a9c5d878f801ed4d8b5c30182bed3332c369
