# DocuRAG Design Decisions

This file records implementation choices, their reasons, and measured results.

## 1. Project structure and tooling

- Use a src layout to keep application code separate from scripts and
  repository files.
- Use uv for dependency management and commit uv.lock for reproducible installs.
- Use Ruff and pre-commit for consistent formatting and lint checks.
- Commit .env.example with placeholders; exclude .env and credentials.

## 2. Database and background processing

- Use PostgreSQL with pgvector to store documents, chunks, embeddings,
  metadata, and full-text search vectors together.
- Use SQLAlchemy's async interface for database access and Alembic for
  versioned schema migrations.
- Run ingestion through Celery with Redis so document processing does not
  block upload requests.
- Share an uploads volume between the API and worker so both can access
  saved source files.
- Use a unique content hash to detect duplicate uploads.

## 3. Parsing and chunking

- Support PDF, Markdown, HTML, and TXT documents.
- Preserve source filename, page number where available, and section
  heading where available for later citations.
- Provide recursive character and token-based chunkers through configuration.
- The current corpus uses recursive chunking with a 1,200-character limit
  and 200-character overlap.
- Keep token counts in chunk metadata for inspection and context budgeting.
- Compare chunking configurations during evaluation rather than assuming
  the current configuration is optimal.

## 4. Embeddings

- Use Gemini gemini-embedding-001 with 768-dimensional embeddings.
- Record the embedding provider, model, and dimensions with stored chunks.
- Restrict vector retrieval to the matching embedding space.
- Batch embedding requests and retry transient failures with backoff.

## 5. Retrieval indexes

- Use an HNSW index with vector_cosine_ops for approximate cosine search.
  HNSW supports retrieval without an index-training step.
- Use a GIN index on the English tsvector column for PostgreSQL full-text
  search.
- Rank keyword matches with ts_rank. This implementation is PostgreSQL
  full-text retrieval, not BM25.

## 6. Hybrid retrieval and reranking

- Retrieve vector and keyword candidates separately.
- Merge rankings using Reciprocal Rank Fusion, avoiding direct comparison
  of incompatible vector and keyword scores.
- Use BAAI/bge-reranker-base to rerank the top 20 fused candidates and
  return the top 5.
- Expose component scores through the debug GET /search endpoint.
- Support source filename and document creation-date filters.
  Creation dates describe uploaded records, not source publication dates.

## 7. Day 3 retrieval comparison

Evaluated 10 hand-written questions with top_k=5 and 20 candidates per
retrieval method.

| Mode | Hit@5 | MRR@5 | Mean request latency (ms) |
| --- | ---: | ---: | ---: |
| Keyword | 0.200 | 0.200 | 12.4 |
| Vector | 1.000 | 0.933 | 598.2 |
| Hybrid | 1.000 | 0.875 | 595.2 |
| Hybrid + reranking | 1.000 | 0.900 | 6828.4 |

### Decision

Choose vector retrieval as the intended initial default for chat because
it achieved the highest MRR@5 in this check.

Keep hybrid retrieval and reranking available for further experiments.
Reranking improved hybrid MRR@5 from 0.875 to 0.900 but did not outperform
vector retrieval and increased measured latency substantially.

The search request schema currently defaults to keyword mode. The chat
implementation should explicitly select vector mode.

### Limitations

- Labels were assistant-reviewed against pooled returned chunks and were
  not independently human-verified.
- This is a small development check, not a held-out benchmark.
- Pooled labels do not identify every relevant chunk in the corpus.
- Latency may include model loading and other cold-start effects.
- Repeat measurements after warm-up and evaluate a larger verified set
  before making broader performance claims.

## 8. Filter verification

Verified through GET /search:

- An upload-date range containing the ConfigMap document returned two chunks.
- A date range excluding that document returned no chunks.
- A reversed date range returned HTTP 422.
- Source filtering restricted results to configuration--configmap.html.

## 9. Next decisions

Record these after implementation and evaluation:

- Context budget and prompt structure.
- Streaming event format and citation validation.
- Conversation query rewriting.
- Generation timeouts and fallback behavior.
- Evaluation configuration and deployment choices.