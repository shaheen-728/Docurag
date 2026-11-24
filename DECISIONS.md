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

## Generation, streaming and citations

- Use vector retrieval for chat based on the Day 3 development results.
- Pass answer rules as system instructions and retrieved context as data.
- Require numbered citations and an explicit insufficient-context response.
- Stream text through POST /chat using SSE events:
  token, citations, done, and error.
- Return structured metadata for sources cited in the answer.
- Rewrite follow-up questions into standalone retrieval queries.
  Conversation history is supplied by the client; it is not persisted.
- Verified a three-turn conversation with correct source references.
- Verified refusal when the selected document cannot answer the question.
- Reject questions exceeding 1,000 characters with HTTP 422.
- Apply retrieval, rewrite, and generation timeouts.
- Configure gemini-2.5-flash-lite as generation fallback.
- Verified fallback switching with simulated failures. Never switch models
  after answer text has streamed.
- Correlate Gemini rewrite, embedding, and generation calls by request ID.
- Log provider-reported generation tokens, latency, and estimated
  standard paid-tier generation cost.

### Limitations

- Embedding responses did not return token usage; embedding tokens and
  cost remain unknown, so combined API cost is not reported.
- Usage accounting currently covers Gemini, not OpenAI.
- Citation validation checks source numbers, not claim support.
- System instructions reduce prompt-injection risk but do not guarantee
  resistance.
- Streamed answers remain provisional until the done event.
- Recorded latency examples are individual requests, not p95 measurements.

## Development retrieval comparison — 80 questions

Vector retrieval is the provisional default: Hit@5 1.000,
MRR@5 0.973, pooled Recall@5 0.984, mean client latency 870.4 ms.

Hybrid did not improve retrieval quality in this run.
Hybrid with reranking achieved the highest MRR@5 (0.988),
but mean latency increased to 5701.7 ms and pooled Recall@5
fell to 0.946. Reranking remains optional.

Labels are assistant-reviewed, not human-verified.
Recall uses pooled judgments. Latency includes cold starts.
These are development results; the 20-question test remains held out.

Judge format retries provide validation feedback and a structural example.
Saved valid judgments are reused; the scoring rubric and model are unchanged.
The development run includes judgments collected before and after this
retry-policy change.

## Token chunk-size experiment preparation

Prepared 256, 512 and 1024-token variants with a fixed 50-token
overlap over the same 38 hash-verified baseline documents.

Chunk counts: 1076, 853 and 797 respectively.
All 80 development evidence quotes fit within a single chunk
in every variant. This is lexical evidence coverage, not retrieval recall.

Heading/page boundaries remain intact; short sections are not merged.
The original 1200-character baseline remains a separate configuration.

Chunk-size experiments use separate databases per variant.
This preserves baseline data, avoids document/chunk uniqueness conflicts,
and gives each variant independent HNSW and GIN indexes.
Retrieval code and settings remain shared across variants.

### chunk-size selection

Compared baseline recursive chunking (1,200 characters, 200-character
overlap) against token chunk sizes 256, 512 and 1024, each with
50-token overlap, on 80 development questions.

All variants achieved Hit@5 of 1.000. Baseline achieved the highest
MRR@5 (0.981) and Precision@5 (0.410). Retain baseline chunking.

Labels are assistant-reviewed, not human-verified. Recall uses
variant-specific pools. Latency measures database retrieval only.
These results do not establish statistical superiority or answer quality.

The held-out test remains pending.

### Judge parser metadata normalization

During held-out judging, allow removal of the redundant top-level
`additionalProperties: false` field. Required judgment fields, scores,
claims, source validation, and rejection of schema wrappers remain unchanged.
The judge model and rubric are unchanged. Previously saved judgments are reused.

## final configuration and evaluation status

### Selected configuration

- Chunker: recursive.
- Chunk size: 1,200 characters.
- Chunk overlap: 200 characters.
- Retrieval: vector search.
- top_k: 5.
- candidate_k: 20.
- Reranking: disabled.
- Answer model: gemini-3.5-flash-lite.
- Generation fallback: disabled.

### Selection rationale

Compared baseline chunking against token chunk sizes 256, 512 and 1024
on 80 development questions. Each token variant used 50-token overlap.

All variants achieved Hit@5 of 1.000. Baseline achieved the highest
MRR@5 (0.981) and Precision@5 (0.410). Retain baseline chunking;
the comparison does not justify replacing it.

Small latency differences from one fixed-order run do not establish
a latency winner.

### Held-out evaluation

The frozen test contains 20 questions. Development and test question IDs
and generation-source document IDs are disjoint.

With the selected configuration, held-out retrieval achieved:

- Hit@5: 0.950 — relevant evidence retrieved for 19/20 questions.
- MRR@5: 0.950.
- Pooled Recall@5: 0.938.
- Precision@5: 0.450.

Do not tune the selected configuration against these held-out results.

All 20 held-out answers have been generated using the development model
and prompt. Held-out judging remains incomplete because of provider
availability failures. Final held-out answer-quality scores are pending.

### Evaluation limitations

Relevance labels are assistant-reviewed, not human-verified.
Partial answer support counts as relevant.

Recall uses reviewed pools rather than exhaustive corpus labels.
The development chunk-size and held-out retrieval pools differ.

Answer evaluation uses a custom same-model LLM judge, not RAGAS.
Six calibration cases passed; this is a sanity check, not proof of
judge accuracy.

Chunk-size latency measures database retrieval only. Held-out API
search latency has a different scope and is not directly comparable.

### Deferred judge provider failures

Allow continuing to other questions after transient provider retries
are exhausted. Deferred questions remain unscored and the run is reported
as partial until all judgments are saved. The model, rubric, and saved
successful judgments remain unchanged.

### Judge transport-error compatibility

Recognize httpx2 transport errors and operation timeouts as transient
judge failures. These use the existing bounded retry and optional
deferral policy. Validation errors remain distinct; model, rubric,
answers and successful judgments are unchanged.

Held-out answer evaluation is complete: 20/20 judgments.

| Mode | Correctness | Relevancy | Faithfulness | Faithfulness N | Context precision | Abstention rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | 0.900 | 0.900 | 1.000 | 18 | 0.450 | 0.100 |

All 20 questions had non-empty retrieved context. Two answers abstained.
Faithfulness excludes abstentions and covers 18 answers.

Scores use a custom same-model LLM judge and assistant-reviewed relevance
labels. They do not establish independently verified accuracy.
Retry and parser-policy changes are documented in DECISIONS.md.

## free public API deployment

Deploy the API on Render Free, PostgreSQL/pgvector on Neon Free,
and caching/rate-limit storage on Render Free Key Value.

Use Dockerfile.deploy with UID 10001. Require verified database TLS
and configure the system CA bundle through PGSSLROOTCERT.

Disable document routes and cross-encoder reranking in the public demo.
Keep the full ingestion workflow available through local Docker Compose.

Copied and fingerprint-verified 38 completed documents and 991 chunks.
Existing embeddings were reused; cloud storage paths were cleared.

Verified the deployed health endpoint, keyword retrieval, streaming
chat with citations and a completion event, retrieval cache hits,
and absence of document routes from the public API schema.

Public API documentation:
https://docurag-demo.onrender.com/docs

Free-tier sleep behavior, ephemeral cache storage, and model-provider
quota/availability limits are documented in README.
