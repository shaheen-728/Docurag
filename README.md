# DocuRAG

[![CI](https://github.com/shaheen-728/Docurag/actions/workflows/ci.yml/badge.svg)](https://github.com/shaheen-728/Docurag/actions/workflows/ci.yml)

Document question answering with streaming responses, source citations,
PostgreSQL/pgvector retrieval, and reproducible evaluation.

## Live API demo

- [Interactive API docs](https://docurag-demo.onrender.com/docs)
- [Health endpoint](https://docurag-demo.onrender.com/health)

The public demo serves 38 documents and 991 chunks.

### Try streaming chat

```bash
curl -N -fS --max-time 120 \
  https://docurag-demo.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can I change the data in an immutable ConfigMap?",
    "source": "configuration--configmap.html"
  }'
```

Responses use Server-Sent Events: `token`, `citations`, `done`, and `error`.
A successful answer ends with a `done` event containing
`"status": "completed"`.

### Deployment and demo limitations

- FastAPI runs on Render using a non-root Docker image.
- PostgreSQL/pgvector runs on Neon with verified TLS.
- Render Key Value provides retrieval caching and rate-limit storage.
- Existing embeddings were copied and verified without re-embedding.
- Public document upload/status routes and cross-encoder reranking are
  disabled. Local Docker Compose retains the full ingestion workflow.
- Free Render instances sleep when inactive; initial requests can take
  a minute or longer.
- Model-provider quotas and availability can interrupt chat responses.
  Free Redis-compatible storage is ephemeral.

This is a project demo, not a production-availability guarantee.

## Evaluation

DocuRAG was evaluated on 80 development questions and a frozen
20-question held-out test. Development and test question IDs and
generation-source document IDs are disjoint.

### Selected configuration

| Setting | Value |
| --- | --- |
| Chunker | Recursive |
| Chunk size | 1,200 characters |
| Chunk overlap | 200 characters |
| Retrieval | Vector search |
| Returned chunks | 5 |
| Retrieval candidates | 20 |
| Reranking | Disabled |
| Answer model | gemini-3.5-flash-lite |
| Generation fallback | Disabled |

### Development evaluation

Completed 320 answer judgments across keyword, vector, hybrid and
hybrid-with-reranking retrieval on 80 development questions.

A separate vector-retrieval experiment compared baseline chunking
against token chunk sizes 256, 512 and 1024:

| Chunking | Hit@5 | MRR@5 | Precision@5 |
| --- | ---: | ---: | ---: |
| Baseline recursive | 1.000 | 0.981 | 0.410 |
| Token 256 | 1.000 | 0.972 | 0.408 |
| Token 512 | 1.000 | 0.959 | 0.375 |
| Token 1024 | 1.000 | 0.963 | 0.362 |

Retained baseline chunking because the alternatives did not improve
these retrieval metrics. These results do not establish statistical
superiority.

Reports:

- [Development answer evaluation](results/day5/baseline/answer-metrics.md)
- [Chunk-size comparison](results/day5/chunk-sizes/metrics.dev80.md)

### Held-out retrieval evaluation

The selected configuration retrieved relevant evidence for 19 of
20 held-out questions.

| Mode | Hit@5 | MRR@5 | Pooled Recall@5 | Precision@5 |
| --- | ---: | ---: | ---: | ---: |
| Vector | 0.950 | 0.950 | 0.938 | 0.450 |

[Held-out retrieval report](results/day5/heldout/retrieval-metrics.md)

### Held-out answer evaluation

Completed all 20 held-out answer judgments.

| Mode | Correctness | Relevancy | Faithfulness | Faithfulness N | Context precision | Abstention rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | 0.900 | 0.900 | 1.000 | 18 | 0.450 | 0.100 |

All 20 questions had non-empty retrieved context. Two answers abstained.
Faithfulness excludes abstentions and covers 18 answers.

[Held-out answer report](results/day5/heldout/answer-metrics.md)

### Methodology and limitations

- Relevance labels are assistant-reviewed, not human-verified.
- Partial answer support counts as relevant.
- Recall uses reviewed pools, not exhaustive corpus labels.
  Development chunk-size and held-out retrieval pools differ.
- Answer evaluation uses a custom same-model LLM judge, not RAGAS.
  Scores do not establish independently verified accuracy.
- Six judge-calibration sanity checks passed, including three
  deliberately incorrect answers. This does not establish judge accuracy.
- Chunk-size timings measure database retrieval only. Held-out API
  search timings have a different scope and are not directly comparable.
- These small evaluation sets do not establish production reliability.

See [DECISIONS.md](DECISIONS.md) for configuration selection and
retry/parser policy changes.

### Reproduce saved reports

From the project root:

```bash
uv run python scripts/report_answers.py
uv run python scripts/report_answers.py --split test
uv run python scripts/report_chunk_sizes.py
uv run python scripts/report_heldout_retrieval.py
```

These reporting commands use saved artifacts and make no model API calls.

Frozen datasets: `data/eval/v1/`

Evaluation results: `results/day5/`