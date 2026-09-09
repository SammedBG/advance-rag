# Advanced RAG Platform

Production-grade Retrieval-Augmented Generation (RAG) platform with hybrid retrieval, query understanding, query transformation, reranking, context engineering, grounded generation, agentic routing, MCP tools, evaluation, observability, authentication, authorization, caching, and production deployment.

---

## 1. Project Vision

The goal of this project is to build a real-world, production-oriented RAG platform rather than a simple "upload document and ask a question" chatbot.

The system should be able to:

- Ingest documents from multiple sources
- Parse and clean documents
- Understand document structure
- Generate parent/child chunks
- Preserve document metadata
- Generate dense embeddings
- Maintain a sparse BM25 index
- Perform hybrid retrieval
- Fuse retrieval results using Reciprocal Rank Fusion (RRF)
- Apply metadata and ACL filtering
- Rerank retrieved documents
- Select the most useful context
- Compress context within a token budget
- Construct grounded prompts
- Generate answers using an LLM
- Return citations
- Validate citations
- Validate answer grounding
- Route requests through an agent
- Use MCP/API tools when retrieval alone is insufficient
- Evaluate retrieval and generation quality
- Provide observability
- Enforce authentication and authorization
- Support production deployment
- Support testing and CI/CD

The architecture should remain modular so individual components can be replaced without rewriting the entire system.

---

# 2. High-Level Architecture

```text
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │  RAG AGENT  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             RAG           MCP          APIs
              │            │            │
              ▼            ▼            ▼
       Advanced RAG    External Tools   External
         Pipeline       / Systems       Services
              │
              ▼
     Query Understanding
              ↓
      Query Transformation
              ↓
    Dense Retrieval + Sparse Retrieval
              ↓
          RRF Fusion
              ↓
       ACL / Metadata Filtering
              ↓
          Re-ranking
              ↓
      Context Selection
              ↓
      Context Compression
              ↓
       Prompt Construction
              ↓
             LLM
              ↓
    Grounding / Validation
              ↓
       Answer + Citations
```

---

# 3. Ingestion Architecture

Document ingestion is separated from query-time retrieval.

```text
DOCUMENT SOURCES
      │
      ▼
Document Loading
      │
      ▼
Parsing / Cleaning
      │
      ▼
Chunking
      │
      ▼
Metadata Extraction
      │
      ▼
Embeddings
      │
      ├─────────────────────┐
      ▼                     ▼
   Qdrant                  BM25
(Dense Index)          (Sparse Index)
```

The ingestion pipeline is responsible for transforming raw documents into searchable knowledge.

---

# 4. Technology Stack

## Backend

* Python 3.10+
* FastAPI
* Pydantic
* Pydantic Settings
* Uvicorn

## RAG

* Sentence Transformers
* Dense embeddings
* Qdrant
* BM25
* Reciprocal Rank Fusion
* Cross-Encoder reranking

## LLM

Current provider:

```text
Groq
```

Current model:

```text
openai/gpt-oss-120b
```

The LLM layer must remain provider-independent so another provider can be introduced later.

## Agent

* LangGraph

## Tools

* MCP
* External APIs

## Data

* Qdrant
* PostgreSQL
* Redis

## Observability

Target stack:

* Structured logging
* Prometheus
* Grafana
* OpenTelemetry
* Distributed tracing

## Infrastructure

* Docker
* Docker Compose
* CI/CD

Kubernetes is intentionally not part of the initial architecture.

---

# 5. Repository Structure

The target repository structure is:

```text
advance-rag/
│
├── app/
│   │
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── indexing.py
│   │   ├── search.py
│   │   ├── chat.py
│   │   ├── health.py
│   │   └── auth.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── service.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   ├── metadata.py
│   │   ├── chunker.py
│   │   ├── tokenizer.py
│   │   ├── pipeline.py
│   │   └── indexer.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── chunk.py
│   │   ├── search.py
│   │   └── response.py
│   │
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── chunk_repository.py
│   │   ├── document_repository.py
│   │   └── user_repository.py
│   │
│   ├── query/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── understanding.py
│   │   └── transformation.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   ├── rrf.py
│   │   ├── reranker.py
│   │   └── filters.py
│   │
│   ├── context/
│   │   ├── __init__.py
│   │   ├── parent_expander.py
│   │   ├── selector.py
│   │   └── compressor.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── prompt_builder.py
│   │   ├── service.py
│   │   ├── citation.py
│   │   └── grounding.py
│   │
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── service.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── container.py
│   │   ├── embedding.py
│   │   ├── qdrant.py
│   │   ├── redis.py
│   │   └── postgres.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── retrieval.py
│   │   ├── reranking.py
│   │   ├── context.py
│   │   ├── generation.py
│   │   ├── citations.py
│   │   ├── grounding.py
│   │   └── end_to_end.py
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── authentication.py
│   │   ├── authorization.py
│   │   └── acl.py
│   │
│   └── core/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── exceptions.py
│       └── telemetry.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── retrieval/
│   ├── generation/
│   ├── agent/
│   └── e2e/
│
├── evaluation/
│   ├── datasets/
│   ├── results/
│   └── reports/
│
├── scripts/
│   ├── index_documents.py
│   ├── rebuild_indexes.py
│   └── run_evaluation.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── retrieval.md
│   ├── evaluation.md
│   └── deployment.md
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── pyproject.toml
```

---

# 6. Core Design Principle

The project is built in layers.

```text
Foundation
    ↓
Ingestion
    ↓
Retrieval
    ↓
Advanced Retrieval
    ↓
Context Engineering
    ↓
Generation
    ↓
Agent
    ↓
Evaluation
    ↓
Production
```

We do not introduce production infrastructure before the RAG pipeline itself works.

We also do not introduce agents simply because an agent framework is available.

Every component must have a reason to exist.

---

# 7. Phase 1 — Project Foundation

## Objective

Create the backend foundation.

Implement:

* FastAPI application
* Configuration
* Environment variables
* Dependency management
* Logging
* Error handling
* Health endpoint
* Service container

Initial configuration:

```env
APP_NAME=advanced-rag
APP_ENV=development
LOG_LEVEL=INFO

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
LLM_API_KEY=

REDIS_URL=redis://localhost:6379/0

POSTGRES_URL=

MCP_SERVER_URL=http://127.0.0.1:8001/mcp
```

Health endpoint:

```text
GET /health
```

Expected:

```json
{
  "status": "ok"
}
```

---

# 8. Phase 2 — Document Ingestion

## Objective

Convert documents into structured internal representations.

Pipeline:

```text
File
 ↓
Loader
 ↓
Cleaner
 ↓
Metadata Extractor
 ↓
Chunker
```

Supported initial sources:

```text
.txt
.md
```

Later:

```text
PDF
DOCX
HTML
URLs
Confluence
Notion
Google Drive
SharePoint
```

---

# 9. Document Model

Every document should contain:

```text
document_id
source
source_type
title
content
metadata
```

The document ID must be deterministic.

For example:

```text
AI ENGINEER
```

for:

```text
AI ENGINEER.txt
```

The document identity must remain stable across re-indexing.

---

# 10. Document Cleaning

Cleaning should:

* Normalize line endings
* Remove unnecessary trailing whitespace
* Normalize excessive blank lines
* Preserve meaningful document content
* Never silently delete useful information

The cleaner must not perform aggressive semantic transformations.

---

# 11. Metadata Extraction

Metadata should eventually include:

```text
document_id
title
source
source_type
author
created_at
updated_at
department
tenant_id
access_control
tags
```

Metadata is important for:

* Filtering
* ACL enforcement
* Retrieval
* Citations
* Auditing
* Multi-tenancy

---

# 12. Phase 3 — Advanced Chunking

The chunking strategy is:

```text
Document
   ↓
Logical Sections
   ↓
Parent Chunks
   ↓
Child Chunks
```

Parent chunks preserve broader context.

Child chunks are optimized for retrieval.

Example:

```text
Parent
 ├── Child 1
 ├── Child 2
 └── Child 3
```

The retrieval system searches child chunks.

The context system can expand a retrieved child back to its parent.

---

# 13. Chunk Requirements

Chunks must contain:

```text
chunk_id
document_id
content
title
heading_path
chunk_index
chunk_type
parent_id
token_count
metadata
```

Chunk types:

```text
parent
child
```

---

# 14. Deterministic Chunk IDs

Chunk IDs must be deterministic.

Do NOT use:

```python
uuid4()
```

for indexed chunk identity.

Use deterministic UUID generation based on stable document/chunk identity.

Reason:

```text
First indexing
      ↓
chunk ID A

Second indexing
      ↓
same chunk ID A
```

Therefore Qdrant upsert replaces the existing point instead of creating duplicates.

This is critical for production indexing.

---

# 15. Chunking Strategy

The current target configuration is:

```text
max_tokens = 500
overlap_tokens = 75
```

The chunker should support:

### Markdown headings

```text
# AI Engineer
## Skills
### Projects
```

### Plain-text section headings

```text
AI ENGINEER:
SKILLS:
EXPERIENCE:
PROJECTS:
```

### Documents without headings

The entire document must remain searchable.

Never produce an empty index merely because the document does not follow a recognized structure.

---

# 16. Phase 4 — Embeddings

Generate dense vectors for child chunks.

Current embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Pipeline:

```text
Child Chunk
    ↓
Embedding Service
    ↓
Vector
    ↓
Qdrant
```

Embedding service must expose:

```text
embed(text)
embed_batch(texts)
dimension
```

The embedding implementation must remain isolated so the model can later be changed.

---

# 17. Phase 5 — Qdrant Dense Retrieval

Qdrant stores:

```text
vector
+
payload
```

Payload should contain:

```text
document_id
chunk_id
chunk_type
parent_id
title
content
heading_path
chunk_index
token_count
metadata
```

Dense retrieval:

```text
User Query
    ↓
Embedding
    ↓
Vector Search
    ↓
Top K
```

Only child chunks should participate in dense retrieval.

Parent chunks are context objects, not primary retrieval targets.

---

# 18. Phase 6 — Sparse Retrieval

Use BM25 for lexical retrieval.

BM25 is particularly useful for:

* Exact technical terms
* Product names
* Error messages
* Function names
* Configuration keys
* IDs
* Rare words

Pipeline:

```text
Query
 ↓
Tokenization
 ↓
BM25
 ↓
Top K
```

The initial implementation may use an in-memory BM25 index.

Production implementation must persist or deterministically rebuild the sparse index.

Important:

```text
BM25 must contain ALL indexed child chunks.
```

It must not be replaced with only the most recently indexed document.

---

# 19. Phase 7 — Hybrid Retrieval

Dense and sparse retrieval solve different problems.

```text
                QUERY
                  │
          ┌───────┴───────┐
          ▼               ▼
       Dense             BM25
     Retrieval          Retrieval
          │               │
          └───────┬───────┘
                  ▼
                 RRF
                  ↓
             Candidate Set
```

Dense retrieval provides semantic matching.

BM25 provides lexical matching.

Hybrid retrieval should be the default retrieval strategy.

---

# 20. Phase 8 — Reciprocal Rank Fusion

RRF combines ranked lists.

Formula:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

Current default:

```text
k = 60
```

The system should preserve:

```text
dense_rank
sparse_rank
```

for debugging and evaluation.

---

# 21. Phase 9 — Query Understanding

Before retrieval, analyze the query.

Example:

```text
"How do I configure authentication in FastAPI?"
```

Possible structured representation:

```text
intent:
procedure

keywords:
authentication
configure
FastAPI

technical_terms:
FastAPI
authentication

entities:
FastAPI
```

The query understanding layer should eventually identify:

* Intent
* Keywords
* Technical terms
* Entities
* Filters
* Language

---

# 22. Phase 10 — Query Transformation

Transform the original query into retrieval-friendly queries.

Example:

```text
Original:
"How do I configure authentication in FastAPI?"
```

Possible retrieval queries:

```text
FastAPI authentication configuration

FastAPI authentication setup

configure FastAPI authentication
```

The transformation layer should preserve the original query.

The original query is always used for final answer generation.

---

# 23. Multi-Query Retrieval

For transformed queries:

```text
Query 1
   ↓
Dense + Sparse

Query 2
   ↓
Dense + Sparse

Query 3
   ↓
Dense + Sparse
```

Then combine all candidate results using global RRF.

Duplicate chunks must receive accumulated ranking scores rather than appearing multiple times.

---

# 24. Phase 11 — Metadata Filtering

Retrieval should support metadata filters.

Examples:

```text
department = engineering
document_type = handbook
project = rag
tenant_id = company_a
```

Filtering should happen as early as possible.

Example:

```text
Query
 ↓
Query Understanding
 ↓
Metadata Filter
 ↓
Dense + Sparse Retrieval
```

---

# 25. Phase 12 — ACL Filtering

Security must be enforced before context reaches the LLM.

Example:

```text
User
 ↓
Authentication
 ↓
Authorization
 ↓
ACL
 ↓
Retrieval
 ↓
Context
 ↓
LLM
```

A user must never receive information from documents they are not authorized to access.

ACL metadata should eventually support:

```text
tenant_id
user_id
roles
groups
permissions
visibility
```

ACL filtering must happen at retrieval time.

Do not rely on the LLM to hide unauthorized information.

---

# 26. Phase 13 — Reranking

After hybrid retrieval:

```text
Top 20-50 candidates
        ↓
Cross Encoder
        ↓
Top 5-10
```

Current model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Reranker input:

```text
(query, chunk)
```

Reranker output:

```text
chunk_id
score
dense_rank
sparse_rank
```

Reranking is used to improve relevance before context construction.

---

# 27. Phase 14 — Context Selection

Retrieval results are not automatically sent to the LLM.

Context selection determines which results are actually useful.

Current target:

```text
maximum context tokens = 3000
```

Selection should consider:

* Relevance score
* Token cost
* Diversity
* Duplicate information
* Parent/child relationships
* Query coverage

---

# 28. Phase 15 — Parent Expansion

When a child chunk is retrieved:

```text
Child
 ↓
parent_id
 ↓
Parent Chunk
```

The system can retrieve the parent for additional context.

Example:

```text
Retrieved child:
"JWT tokens expire after 30 minutes."

Parent:
"Authentication Configuration

JWT tokens expire after 30 minutes.
Refresh tokens are valid for 7 days.
..."
```

This improves contextual completeness.

---

# 29. Phase 16 — Context Compression

Selected context can be compressed before being sent to the LLM.

Pipeline:

```text
Retrieved Context
       ↓
Context Selection
       ↓
Context Compression
       ↓
Final Context
       ↓
LLM
```

Current target:

```text
maximum compressed tokens = 2000
```

Compression must preserve:

* Facts
* Technical terminology
* Relationships
* Citation/source identity
* Important constraints

Compression must not invent information.

---

# 30. Phase 17 — Prompt Construction

Prompt builder receives:

```text
Original Query
+
Compressed Context
```

The prompt must explicitly instruct the LLM:

1. Use the provided documentation.
2. Do not invent unsupported information.
3. If the answer is not present, say so.
4. Cite the relevant context.
5. Keep citations aligned with source numbers.

Example:

```text
[1] FastAPI Authentication Documentation
...

[2] Security Configuration Documentation
...
```

Answer:

```text
FastAPI authentication can be configured using ...

[1]
```

---

# 31. Phase 18 — LLM Generation

LLM service must be provider-independent.

Interface:

```text
generate(prompt)
```

Current provider:

```text
Groq
```

Current model:

```text
openai/gpt-oss-120b
```

Generation response should contain:

```text
answer
model
input_tokens
output_tokens
total_tokens
```

---

# 32. Phase 19 — Citation Validation

After generation:

```text
LLM Answer
    ↓
Citation Validator
    ↓
Valid / Invalid
```

Validate:

* Citation format
* Citation number
* Citation range
* Missing citations

Example:

```text
Context count = 3

Valid:
[1]
[2]
[3]

Invalid:
[4]
[7]
```

Citation validation is separate from factual grounding.

---

# 33. Phase 20 — Grounding Validation

Grounding determines whether the answer is supported by retrieved context.

Initial implementation can use lexical overlap as a baseline.

Production implementation should evolve toward stronger evaluation methods.

Possible future approaches:

```text
Claim extraction
      ↓
Evidence matching
      ↓
Entailment / LLM judge
      ↓
Grounding score
```

Output:

```text
score
grounded
matched_terms
unmatched_terms
```

---

# 34. Phase 21 — No-Answer Behavior

If useful context cannot be found:

```text
I could not find this information in the provided documentation.
```

The system must NOT hallucinate.

Response:

```json
{
  "answer": "I could not find this information in the provided documentation.",
  "citations": [],
  "grounding": {
    "grounded": false
  }
}
```

This behavior is a fundamental RAG safety mechanism.

---

# 35. Phase 22 — Agentic RAG

Only after the core RAG pipeline works should the agent layer be introduced.

Architecture:

```text
                 USER QUERY
                     │
                     ▼
                RAG AGENT
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       RAG PATH              TOOL PATH
          │                     │
          ▼                     ▼
     Hybrid Search          MCP / API
          │                     │
          ▼                     ▼
      Generation            Tool Result
          │                     │
          └──────────┬──────────┘
                     ▼
                  Answer
```

---

# 36. LangGraph

LangGraph should orchestrate decisions.

Example graph:

```text
START
  ↓
ROUTE
  ├── RAG
  │    ↓
  │  RETRIEVE
  │    ↓
  │  CONTEXT
  │    ↓
  │  GENERATE
  │    ↓
  │  VALIDATE
  │
  └── MCP
       ↓
    TOOL CALL
       ↓
    RESULT
       ↓
     ANSWER
       ↓
      END
```

The graph should remain small and understandable.

Do not create unnecessary agent loops.

---

# 37. MCP

MCP provides a standardized interface for external tools.

Example:

```text
RAG Agent
    ↓
MCP Client
    ↓
MCP Server
    ↓
External Tool
```

Possible future tools:

```text
Database lookup
Ticket system
Internal API
Monitoring system
GitHub
Documentation service
CRM
```

MCP is for external capabilities.

It is not a replacement for the RAG pipeline.

---

# 38. APIs

FastAPI exposes the system.

Initial APIs:

```text
GET  /health

POST /index

GET  /search

POST /chat
```

Future APIs:

```text
POST /documents
DELETE /documents/{id}

GET /documents

POST /query

GET /evaluation

GET /metrics
```

---

# 39. Target Search Response

A search response should expose enough information for debugging and observability.

Example:

```json
{
  "query": "how does authentication work?",
  "answer": "...",
  "citations": [1, 2],
  "citation_validation": {
    "valid": true,
    "invalid_citations": [],
    "missing_citations": false
  },
  "grounding": {
    "score": 0.91,
    "grounded": true
  },
  "results": [],
  "generation": {
    "model": "openai/gpt-oss-120b",
    "input_tokens": 1200,
    "output_tokens": 250,
    "total_tokens": 1450
  },
  "context_stats": {
    "contexts_selected": 5,
    "contexts_compressed": 4,
    "selected_tokens": 2800,
    "compressed_tokens": 1800
  }
}
```

---

# 40. Phase 23 — PostgreSQL

PostgreSQL becomes the system-of-record database.

Use it for:

```text
Users
Documents
Document versions
Tenants
ACLs
Metadata
Indexing status
Queries
Evaluation results
Audit logs
```

Do NOT use PostgreSQL as the primary vector database.

Qdrant remains responsible for vector retrieval.

---

# 41. Document Lifecycle

Production document lifecycle:

```text
UPLOADED
   ↓
PROCESSING
   ↓
PARSED
   ↓
CHUNKED
   ↓
EMBEDDED
   ↓
INDEXED
   ↓
READY
```

Failure:

```text
PROCESSING
   ↓
FAILED
```

Store errors for diagnosis.

---

# 42. Document Versioning

Documents should eventually support:

```text
document_id
version
content_hash
created_at
updated_at
status
```

Content hash allows the system to detect whether a document actually changed.

If the document has not changed:

```text
Do not reprocess unnecessarily.
```

---

# 43. Stale Chunk Deletion

Deterministic IDs prevent duplicates.

However, deterministic IDs alone do NOT remove stale chunks.

Example:

```text
Version 1:
10 chunks

Version 2:
7 chunks
```

The old 3 chunks must be deleted.

Production indexing must therefore support:

```text
Document version
      ↓
Determine current chunk IDs
      ↓
Compare with existing IDs
      ↓
Delete stale chunks
      ↓
Upsert current chunks
```

This is required for correct re-indexing.

---

# 44. Redis

Redis is used for:

```text
Query caching
LLM response caching
Embedding caching
Rate limiting
Short-lived state
Distributed coordination
```

Example:

```text
Query
 ↓
Cache lookup
 ↓
Hit → return
 ↓
Miss
 ↓
RAG pipeline
 ↓
Store result
```

Cache keys must include relevant dimensions such as:

```text
tenant
user/ACL context
query
index/version
model
```

Security must never be bypassed by a shared cache.

---

# 45. Authentication

Production API requires authentication.

Possible mechanism:

```text
JWT
```

Flow:

```text
User
 ↓
Login
 ↓
Access Token
 ↓
FastAPI
 ↓
Authentication
 ↓
Authorization
 ↓
RAG
```

Authentication answers:

```text
Who are you?
```

Authorization answers:

```text
What are you allowed to access?
```

---

# 46. Multi-Tenancy

The system should eventually support multiple organizations.

Example:

```text
Tenant A
 ├── Documents
 ├── Users
 └── ACLs

Tenant B
 ├── Documents
 ├── Users
 └── ACLs
```

Every document and retrieval request should carry:

```text
tenant_id
```

Cross-tenant data leakage must be impossible through:

* Qdrant filters
* PostgreSQL queries
* Redis keys
* API authorization
* Context construction

---

# 47. Observability

Production RAG needs visibility into every stage.

Trace:

```text
Request
 ↓
Agent
 ↓
Query Understanding
 ↓
Transformation
 ↓
Dense Retrieval
 ↓
BM25
 ↓
RRF
 ↓
Reranking
 ↓
Context Selection
 ↓
Compression
 ↓
LLM
 ↓
Validation
```

Record latency for each stage.

---

# 48. Metrics

Important metrics:

## API

```text
request_count
request_latency
error_count
```

## Retrieval

```text
dense_latency
bm25_latency
rrf_latency
reranker_latency
retrieval_count
```

## Generation

```text
llm_latency
input_tokens
output_tokens
total_tokens
```

## RAG Quality

```text
retrieval_recall
precision
MRR
nDCG
context_relevance
answer_relevance
faithfulness
citation_accuracy
grounding_score
```

---

# 49. Logging

Use structured logs.

Example:

```json
{
  "timestamp": "...",
  "level": "INFO",
  "request_id": "...",
  "query_id": "...",
  "stage": "reranking",
  "latency_ms": 42,
  "candidate_count": 20
}
```

Never log:

```text
API keys
passwords
tokens
private credentials
sensitive user data
```

---

# 50. Evaluation

Evaluation is a first-class part of the project.

The system should not be considered production-ready simply because answers "look good."

Evaluation must measure each stage.

---

# 51. Retrieval Evaluation

Metrics:

```text
Recall@K
Precision@K
MRR
nDCG@K
Hit Rate
```

Example:

```text
Recall@5
Recall@10
MRR@10
nDCG@10
```

Compare:

```text
Dense
BM25
Hybrid
Hybrid + Reranker
```

This tells us whether each retrieval layer actually improves the system.

---

# 52. Reranking Evaluation

Compare:

```text
Hybrid
vs
Hybrid + Reranker
```

Measure:

```text
MRR
nDCG
Recall
Precision
```

The reranker must demonstrate measurable improvement.

---

# 53. Context Evaluation

Measure:

```text
Context relevance
Context precision
Context recall
Context redundancy
Token efficiency
```

Important question:

```text
Did we send the right information to the LLM?
```

---

# 54. Generation Evaluation

Measure:

```text
Answer relevance
Faithfulness
Groundedness
Completeness
Citation correctness
Citation completeness
```

Important question:

```text
Did the LLM answer correctly using the retrieved evidence?
```

---

# 55. End-to-End Evaluation

Complete pipeline:

```text
Question
 ↓
Retrieval
 ↓
Reranking
 ↓
Context
 ↓
LLM
 ↓
Answer
```

Evaluate:

```text
End-to-end correctness
Groundedness
Citation accuracy
Latency
Token cost
```

---

# 56. Evaluation Dataset

Create a dataset containing:

```text
question
expected_answer
relevant_document_ids
relevant_chunk_ids
difficulty
category
```

Example:

```json
{
  "question": "What is hybrid retrieval?",
  "expected_answer": "...",
  "relevant_document_ids": ["rag-doc"],
  "relevant_chunk_ids": ["chunk-1", "chunk-4"],
  "difficulty": "medium",
  "category": "retrieval"
}
```

---

# 57. Testing Strategy

Testing is divided into:

```text
Unit Tests
Integration Tests
Retrieval Tests
Generation Tests
Agent Tests
End-to-End Tests
```

---

# 58. Unit Tests

Every core component should be independently testable.

Examples:

```text
Cleaner
Chunker
Tokenizer
Metadata extractor
Query understanding
Query transformation
RRF
Reranker
Context selector
Compressor
Citation validator
Grounding validator
```

---

# 59. Integration Tests

Test:

```text
FastAPI
+
Qdrant
+
BM25
+
LLM
```

Examples:

```text
Index document
 ↓
Search
 ↓
Retrieve context
 ↓
Generate answer
```

---

# 60. Retrieval Regression Tests

Maintain fixed queries.

After changing retrieval:

```text
Run benchmark
 ↓
Compare metrics
 ↓
Detect regression
```

Never blindly change retrieval algorithms without measuring their effect.

---

# 61. End-to-End Tests

Example:

```text
Upload document
 ↓
Index
 ↓
Ask question
 ↓
Retrieve
 ↓
Generate
 ↓
Validate
 ↓
Return answer
```

Expected:

```text
HTTP 200
Answer exists
Citations valid
Grounding acceptable
```

---

# 62. Error Handling

Every layer must have controlled failures.

Examples:

```text
Invalid document
Embedding failure
Qdrant unavailable
BM25 failure
LLM timeout
MCP unavailable
Database failure
Redis failure
```

Do not expose raw internal stack traces to users in production.

Return structured errors.

---

# 63. Timeouts

External dependencies require explicit timeouts.

Examples:

```text
Qdrant timeout
LLM timeout
MCP timeout
PostgreSQL timeout
Redis timeout
```

A single unavailable external service must not hang the entire API indefinitely.

---

# 64. Retries

Retries should only be applied to transient failures.

Example:

```text
LLM 503
 ↓
Retry
 ↓
Retry
 ↓
Fail gracefully
```

Do not retry:

```text
400
401
403
invalid input
```

Use exponential backoff.

---

# 65. Rate Limiting

Production API should protect:

```text
/chat
/search
/index
```

Rate limits may vary by:

```text
tenant
user
endpoint
API key
```

Redis can provide distributed rate limiting.

---

# 66. Docker Architecture

Local production-like environment:

```text
Docker Compose
│
├── rag-api
├── qdrant
├── postgres
└── redis
```

Optional:

```text
├── prometheus
└── grafana
```

MCP can run separately when required.

---

# 67. Local Development

Start infrastructure:

```powershell
docker compose up -d
```

Start API:

```powershell
uvicorn app.main:app --reload
```

Check:

```text
GET /health
```

---

# 68. Environment Separation

Support:

```text
development
testing
staging
production
```

Never hard-code production credentials.

Use:

```text
.env
```

locally.

Use secret management in production.

---

# 69. CI/CD

Pipeline:

```text
Git Push
   ↓
CI
   ↓
Lint
   ↓
Unit Tests
   ↓
Integration Tests
   ↓
Build Docker Image
   ↓
Security Scan
   ↓
Evaluation Tests
   ↓
Deploy
```

Production deployment should only occur after automated validation.

---

# 70. Deployment

Initial production architecture:

```text
                    Internet
                       │
                       ▼
                  Load Balancer
                       │
                       ▼
                 FastAPI Service
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Qdrant        PostgreSQL       Redis
        │
        ▼
   Vector Storage

FastAPI
   │
   ├── Groq
   ├── MCP
   └── External APIs
```

Multiple API instances can run behind the load balancer.

---

# 71. Stateless API

The FastAPI application should remain as stateless as possible.

Persistent state belongs in:

```text
PostgreSQL
Qdrant
Redis
```

Do not depend on process-local Python memory for production-critical data.

This is especially important for:

```text
BM25
ChunkRepository
sessions
caches
document state
```

---

# 72. Production BM25 Requirement

The current in-memory BM25 implementation is acceptable for development.

Production requires one of:

```text
Persistent sparse index
```

or:

```text
Deterministic rebuild from PostgreSQL/document storage
```

The important requirement is:

```text
FastAPI restart
      ↓
BM25 still available
```

and:

```text
multiple API replicas
      ↓
same sparse index
```

---

# 73. Production Parent Storage

The current parent repository is in-memory.

Development:

```text
ChunkRepository
```

Production:

```text
PostgreSQL
```

or another persistent metadata store.

Parent expansion must continue working after:

```text
FastAPI restart
```

and across:

```text
multiple API replicas
```

---

# 74. Production Indexing

Indexing should eventually be asynchronous.

Instead of:

```text
POST /index
      ↓
Process everything
      ↓
Return
```

use:

```text
POST /documents
      ↓
Create indexing job
      ↓
Queue
      ↓
Worker
      ↓
Load
      ↓
Chunk
      ↓
Embed
      ↓
Index
      ↓
READY
```

This prevents long indexing operations from blocking API requests.

---

# 75. Background Workers

Future architecture:

```text
FastAPI
   │
   ▼
Job Queue
   │
   ▼
Worker
   │
   ├── Document processing
   ├── Chunking
   ├── Embedding
   └── Indexing
```

Redis can initially support queue/state coordination.

---

# 76. Security Requirements

Production checklist:

```text
[ ] Authentication
[ ] Authorization
[ ] Tenant isolation
[ ] ACL filtering
[ ] Input validation
[ ] Rate limiting
[ ] Secrets management
[ ] HTTPS
[ ] Secure headers
[ ] Audit logging
[ ] No secret leakage
[ ] No cross-tenant cache leakage
[ ] No unauthorized context retrieval
```

---

# 77. RAG Security Principle

The most important rule:

```text
Unauthorized data must never reach the LLM.
```

Correct:

```text
User
 ↓
ACL filter
 ↓
Retriever
 ↓
Authorized context
 ↓
LLM
```

Incorrect:

```text
User
 ↓
Retriever
 ↓
Unauthorized context
 ↓
LLM
 ↓
"Please don't mention private data"
```

The second approach is not a security boundary.

---

# 78. Performance Optimization

After correctness:

```text
Measure
 ↓
Profile
 ↓
Optimize
```

Potential optimizations:

```text
Embedding batching
Query embedding cache
LLM response cache
Parallel dense/BM25 retrieval
Parallel transformed queries
Qdrant indexing optimization
Reranker batching
Connection pooling
Async APIs
```

Do not optimize prematurely.

---

# 79. Cost Optimization

Track:

```text
Embedding tokens
LLM input tokens
LLM output tokens
Requests
Reranking volume
```

Potential optimizations:

```text
Cache embeddings
Cache repeated queries
Reduce candidate count
Compress context
Use cheaper models for simple tasks
Avoid unnecessary agent calls
Avoid unnecessary query transformations
```

---

# 80. RAG Request Lifecycle

The complete production request should look like:

```text
USER
 │
 ▼
Authentication
 │
 ▼
Authorization
 │
 ▼
RAG AGENT
 │
 ▼
Query Understanding
 │
 ▼
Query Transformation
 │
 ▼
ACL / Metadata Filters
 │
 ├───────────────┐
 ▼               ▼
Dense           BM25
 │               │
 └───────┬───────┘
         ▼
        RRF
         │
         ▼
     Reranking
         │
         ▼
   Parent Expansion
         │
         ▼
   Context Selection
         │
         ▼
  Context Compression
         │
         ▼
   Prompt Construction
         │
         ▼
        LLM
         │
         ▼
 Citation Validation
         │
         ▼
 Grounding Validation
         │
         ▼
      Response
```

---

# 81. Production Ingestion Lifecycle

```text
Document Upload
      ↓
Authentication
      ↓
Authorization
      ↓
Create Document Record
      ↓
Calculate Content Hash
      ↓
Check Existing Version
      ↓
Indexing Job
      ↓
Load
      ↓
Clean
      ↓
Metadata Extraction
      ↓
Structure-Aware Chunking
      ↓
Parent / Child Creation
      ↓
Embedding
      ↓
Qdrant Upsert
      ↓
Sparse Index Update
      ↓
Delete Stale Chunks
      ↓
Mark Document READY
```

---

# 82. Current Development Roadmap

The project will be implemented in this order.

```text
PHASE 1 — RAG CORE
├── Ingestion
├── Chunking
├── Embeddings
├── Dense Retrieval
├── Sparse Retrieval
├── RRF
└── Reranking

PHASE 2 — ADVANCED RAG
├── Query Understanding
├── Query Transformation
├── Metadata Filtering
├── ACL Filtering
├── Context Selection
├── Context Compression
└── Prompt Construction

PHASE 3 — GENERATION
├── LLM
├── Citations
├── Grounding
└── Answer Validation

PHASE 4 — AGENTIC RAG
├── LangGraph
├── Routing
├── MCP
└── External Tools

PHASE 5 — EVALUATION
├── Retrieval Evaluation
├── Reranking Evaluation
├── Context Evaluation
├── Answer Evaluation
├── Citation Evaluation
├── Grounding Evaluation
└── End-to-End Evaluation

PHASE 6 — PRODUCTION
├── PostgreSQL
├── Redis
├── Authentication
├── Authorization / ACL
├── Persistent BM25
├── Persistent Parent Storage
├── Async Indexing
├── Observability
├── Error Handling
├── Testing
├── Docker
└── CI/CD
```

---

# 83. Definition of Done

The project is NOT considered production-ready when:

```text
"The chatbot returns answers."
```

It is production-ready when:

```text
[ ] Documents can be reliably indexed
[ ] Re-indexing is idempotent
[ ] Stale chunks are removed
[ ] Dense retrieval works
[ ] Sparse retrieval works
[ ] Hybrid retrieval works
[ ] RRF improves candidate ranking
[ ] Reranking improves relevance
[ ] Query transformation is measurable
[ ] Metadata filtering works
[ ] ACL filtering is enforced
[ ] Parent expansion is persistent
[ ] Context selection respects token budgets
[ ] Context compression preserves evidence
[ ] LLM answers are grounded
[ ] Citations are validated
[ ] Hallucination behavior is controlled
[ ] Agent routing works
[ ] MCP tools work
[ ] Evaluation dataset exists
[ ] Retrieval metrics are measured
[ ] Generation metrics are measured
[ ] Regression tests exist
[ ] Authentication works
[ ] Authorization works
[ ] Tenant isolation works
[ ] Redis caching is secure
[ ] PostgreSQL persistence works
[ ] BM25 survives restart
[ ] API is observable
[ ] Errors are handled
[ ] Timeouts exist
[ ] Retries exist where appropriate
[ ] Rate limiting exists
[ ] Docker deployment works
[ ] CI/CD works
[ ] Secrets are managed securely
[ ] Production deployment has been tested
```

---

# 84. Engineering Principles

## Principle 1 — Correctness before complexity

Do not add another component until the current layer works.

---

## Principle 2 — Measure before changing

Any retrieval or generation optimization should be evaluated.

---

## Principle 3 — Persistent state must be persistent

Do not depend on Python process memory for production-critical state.

---

## Principle 4 — Security before generation

ACL filtering must happen before context reaches the LLM.

---

## Principle 5 — Deterministic indexing

The same document should produce stable IDs.

---

## Principle 6 — No hallucination by design

If evidence is unavailable:

```text
Say that the information was not found.
```

---

## Principle 7 — Components must be replaceable

Examples:

```text
Qdrant → another vector DB
Groq → another LLM provider
Sentence Transformers → another embedding model
BM25 → another sparse retrieval engine
Redis → another cache
```

The architecture should not tightly couple business logic to infrastructure.

---

## Principle 8 — Agent is orchestration, not magic

LangGraph decides which capability to use.

It does not replace:

```text
retrieval
ranking
context engineering
generation
evaluation
```

---

## Principle 9 — Production means reproducibility

The system must behave consistently across:

```text
restart
multiple processes
multiple machines
multiple API replicas
```

---

## Principle 10 — Every optimization must have a reason

Do not add technologies because they are popular.

Add them because they solve a measurable problem.

---

# 85. Final Target Architecture

```text
                              ┌──────────────┐
                              │     USER     │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │ Load Balancer│
                              └──────┬───────┘
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │      Gateway         │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
             Authentication                    RAG Agent
                    │                               │
                    ▼                               ▼
             Authorization                 Query Understanding
                    │                               │
                    ▼                               ▼
                 ACLs                      Query Transformation
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                           Hybrid Retrieval
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
                    Qdrant                     BM25
                 Dense Search              Sparse Search
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                                   RRF
                                    │
                                    ▼
                                Reranker
                                    │
                                    ▼
                           Parent Expansion
                                    │
                                    ▼
                          Context Selection
                                    │
                                    ▼
                         Context Compression
                                    │
                                    ▼
                           Prompt Builder
                                    │
                                    ▼
                                  LLM
                                    │
                                    ▼
                         Citation Validation
                                    │
                                    ▼
                         Grounding Validation
                                    │
                                    ▼
                               Response


          ┌──────────────────────────────────────────────┐
          │              Persistent Services             │
          │                                              │
          │ PostgreSQL    Qdrant    Redis                │
          │                                              │
          └──────────────────────────────────────────────┘


          ┌──────────────────────────────────────────────┐
          │              External Capabilities           │
          │                                              │
          │ MCP Servers     APIs     External Systems    │
          │                                              │
          └──────────────────────────────────────────────┘


          ┌──────────────────────────────────────────────┐
          │              Observability                   │
          │                                              │
          │ Logs     Metrics     Traces     Evaluation   │
          │                                              │
          └──────────────────────────────────────────────┘


          ┌──────────────────────────────────────────────┐
          │              Deployment                      │
          │                                              │
          │ Docker → CI/CD → Staging → Production       │
          │                                              │
          └──────────────────────────────────────────────┘
```

---

# 86. Current Project Status

The project has already progressed beyond the foundation.

Implemented/partially implemented:

```text
[x] FastAPI foundation
[x] Configuration
[x] Document loading
[x] Document cleaning
[x] Metadata extraction
[x] Structure-aware chunking
[x] Parent/child chunks
[x] Deterministic chunk IDs
[x] Embedding service
[x] Qdrant integration
[x] BM25 retrieval
[x] RRF
[x] Query understanding
[x] Query transformation
[x] Hybrid retrieval
[x] Cross-encoder reranking
[x] Parent expansion
[x] Context selection
[x] Context compression
[x] Prompt construction
[x] Groq generation
[x] Citation validation
[x] Grounding validation baseline
[x] LangGraph agent
[x] MCP client/service
[x] Basic API endpoints
```

Important current limitations:

```text
[ ] BM25 persistence
[ ] BM25 multi-document aggregation
[ ] Persistent parent storage
[ ] Stale chunk deletion
[ ] PostgreSQL
[ ] Redis
[ ] Authentication
[ ] Authorization
[ ] Production ACL enforcement
[ ] Async indexing
[ ] Full evaluation framework
[ ] Production observability
[ ] Comprehensive automated tests
[ ] Docker production setup
[ ] CI/CD
```

---

# 87. Immediate Next Engineering Priorities

Before adding more features, stabilize the current RAG core.

Priority order:

```text
1. Verify ingestion output
2. Verify chunk correctness
3. Verify deterministic indexing
4. Verify Qdrant retrieval
5. Fix BM25 persistence/aggregation design
6. Verify hybrid retrieval
7. Verify reranking
8. Verify parent expansion
9. Verify context selection/compression
10. Verify generation
11. Verify citations
12. Verify grounding
13. Add evaluation dataset
14. Measure retrieval quality
15. Improve retrieval
16. Then productionize
```

The system should be debugged layer-by-layer rather than modifying multiple components simultaneously.

---

# 88. Final Goal

The final product is not simply a chatbot.

It is a:

```text
Production-grade AI Knowledge Platform
```

with:

```text
Reliable ingestion
+
Advanced retrieval
+
Hybrid search
+
Reranking
+
Context engineering
+
Grounded generation
+
Citations
+
Agentic routing
+
MCP tools
+
Evaluation
+
Security
+
Observability
+
Persistence
+
Scalability
+
CI/CD
```

The architecture should remain understandable, testable, measurable, and replaceable throughout its evolution from local development to production.
