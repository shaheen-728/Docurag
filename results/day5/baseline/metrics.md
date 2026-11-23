# Development retrieval pilot

Questions: 80. Top k: 5.

| Mode | Hit@5 | MRR@5 | Pooled Recall@5 | Mean client ms |
| --- | ---: | ---: | ---: | ---: |
| keyword | 0.250 | 0.250 | 0.196 | 13.1 |
| vector | 1.000 | 0.973 | 0.984 | 870.4 |
| hybrid | 1.000 | 0.969 | 0.972 | 821.0 |
| hybrid_rerank | 1.000 | 0.988 | 0.946 | 5701.7 |

Labels are assistant-reviewed, not human-verified.
Recall is measured against pooled labels, not all relevant corpus chunks.
Latency includes cold-start effects; no p95 claim is made.
This development pilot is not the held-out test.
