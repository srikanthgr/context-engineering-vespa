# Vespa Ranking Profiles Guide

> **⚠️ Note**: This is an experimental project for learning and exploration purposes.

A comprehensive guide to the ranking profiles implemented in this Vespa-based RAG system. This document explains how each ranking profile works, when to use them, and their performance characteristics.

## Table of Contents

- [Overview](#overview)
- [Ranking Architecture](#ranking-architecture)
- [Ranking Profiles](#ranking-profiles)
  - [1. Hybrid Ranking](#1-hybrid-ranking)
  - [2. Layered Ranking (Base)](#2-layered-ranking-base)
  - [3. Layered with Second Phase](#3-layered-with-second-phase)
  - [4. Layered with Diversity](#4-layered-with-diversity)
  - [5. Layered with MaxSim](#5-layered-with-maxsim)
  - [6. Layered Normalized Fusion](#6-layered-normalized-fusion)
  - [7. Layered with Global Phase](#7-layered-with-global-phase)
  - [8. Layered ML Ranking](#8-layered-ml-ranking)
- [Performance Comparison](#performance-comparison)
- [API Usage](#api-usage)
- [Choosing the Right Profile](#choosing-the-right-profile)

---

## Overview

This system implements **8 different ranking profiles** for document retrieval, ranging from simple hybrid search to advanced machine learning-based ranking. Each profile builds upon Vespa's multi-phase ranking architecture to progressively refine search results.

### Key Concepts

- **Layered Chunking**: Documents are split into chunks, and ranking operates at both chunk and document levels
- **Multi-Phase Ranking**: Vespa's first → second → global phase architecture for efficient ranking
- **Hybrid Search**: Combines semantic similarity (embeddings) with lexical relevance (BM25)
- **Dual Filtering**: Join operation ensures chunks match BOTH semantic AND lexical criteria

---

## Ranking Architecture

### Vespa's Three-Phase Ranking

```
┌─────────────────┐
│  First Phase    │  Fast, runs on ALL matched documents (1000s)
│  Simple scoring │  Per content node
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Second Phase   │  Moderate cost, runs on top-K (100-1000)
│  Refined scoring│  Per content node
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Global Phase   │  Expensive, runs on top-N (10-100)
│  Final polish   │  Container node (cross-document)
└─────────────────┘
```

### Phase Characteristics

| Phase | Speed | Complexity | Document Count | Latency Impact |
|-------|-------|------------|----------------|----------------|
| **First** | Fast | Simple | 1000s-10000s | ~1ms |
| **Second** | Medium | Moderate | 100-1000 | ~5-20ms |
| **Global** | Slow | Complex | 10-100 | ~50-200ms |

---

## Ranking Profiles

### 1. Hybrid Ranking

**Profile Name**: `hybrid`

**Description**: Basic hybrid search combining semantic similarity (cosine) with lexical matching (BM25).

**Scoring Formula**:
```
score = nativeRank(title) + nativeRank(chunks) + max(cosine_similarity)
```

**Features**:
- ✅ Simple and fast
- ✅ Good baseline performance
- ✅ Works well for most queries
- ❌ No chunk-level filtering
- ❌ May return irrelevant chunks

**When to Use**:
- Initial baseline testing
- Low-latency requirements (<10ms)
- General-purpose search

**API Endpoint**:
```bash
GET /query?query=why+is+colbert+effective
```

**Example Response Features**:
```json
{
  "similarities": {"0": 0.87, "1": 0.92},
  "nativeRank(chunks)": 0.65,
  "nativeRank(title)": 0.45
}
```

---

### 2. Layered Ranking (Base)

**Profile Name**: `layeredranking`

**Description**: Core layered ranking with dual-criteria chunk filtering. Only chunks matching BOTH semantic and lexical criteria are selected.

**Scoring Formula**:
```
chunk_scores = join(
  1/(1+euclidean_distance(query, embedding)),
  bm25(chunks),
  f(a,b)(a+b)
)
first_phase = sum(chunk_scores)
best_chunks = top(3, chunk_scores)
```

**Features**:
- ✅ Dual filtering (semantic + lexical)
- ✅ Automatic best-chunk selection
- ✅ Higher precision than hybrid
- ✅ No manual threshold needed
- ❌ May be too strict for sparse content

**When to Use**:
- Production RAG systems
- When precision matters
- Multi-chunk documents

**API Endpoint**:
```bash
GET /query-layered?query=why+is+colbert+effective
```

**Example Response Features**:
```json
{
  "best_chunks": {"0": 0.89, "2": 0.83, "3": 0.92},
  "chunk_scores": {"0": 0.89, "2": 0.83, "3": 0.92},
  "my_distance_scores": {"0": 0.19, "1": 0.17, "2": 0.18, "3": 0.19},
  "my_text_scores": {"0": 0.70, "2": 0.65, "3": 0.73}
}
```

---

### 3. Layered with Second Phase

**Profile Name**: `layered_with_second_phase`

**Description**: Adds second-phase ranking that combines chunk scores with title and maximum similarity signals.

**Scoring Formula**:
```
first_phase = sum(chunk_scores)

second_phase = sum(chunk_scores) * 0.7 
             + title_score * 0.2 
             + max_similarity * 0.1
```

**Second Phase Config**:
- Reranks top **100** documents from first phase

**Features**:
- ✅ Incorporates document metadata (title)
- ✅ Balanced multi-signal ranking
- ✅ Easy to tune weights
- ✅ Minimal latency increase (~5-10ms)
- ❌ Weights need manual tuning

**When to Use**:
- When titles are informative
- Need better precision on top results
- Production systems with latency budget

**Tuning Tips**:
```python
# Boost title importance for academic papers
second_phase = "sum(chunk_scores()) * 0.6 + title_score * 0.4"

# Focus on chunk quality
second_phase = "sum(chunk_scores()) * 0.9 + title_score * 0.1"
```

---

### 4. Layered with Diversity

**Profile Name**: `layered_with_diversity`

**Description**: Promotes documents with diverse chunk scores to ensure comprehensive coverage of topics.

**Scoring Formula**:
```
first_phase = sum(chunk_scores)

second_phase = sum(chunk_scores) 
             + chunk_variance * 0.5 
             + chunk_spread * 0.3

where:
  chunk_variance = variance(chunk_scores)
  chunk_spread = max(chunk_scores) - min(chunk_scores)
```

**Second Phase Config**:
- Reranks top **50** documents

**Features**:
- ✅ Reduces redundancy
- ✅ Better for exploratory search
- ✅ Multi-aspect query support
- ❌ May demote highly focused docs
- ❌ Variance calculation adds cost

**When to Use**:
- Research/discovery scenarios
- Broad, multi-aspect queries
- Avoiding "echo chamber" results
- Literature reviews

**Example Query Types**:
- "What are the advantages and disadvantages of ColBERT?"
- "Overview of neural retrieval methods"
- "Different approaches to document ranking"

---

### 5. Layered with MaxSim

**Profile Name**: `layered_with_maxsim`

**Description**: ColBERT-style MaxSim scoring for high-precision semantic search.

**Scoring Formula**:
```
first_phase = sum(chunk_scores)

second_phase = maxsim_score * 10 
             + sumsim_score * 0.1 
             + sum(chunk_scores) * 0.5

where:
  maxsim_score = max(distance_scores)  # Best chunk match
  sumsim_score = sum(distance_scores)  # Overall document relevance
```

**Second Phase Config**:
- Reranks top **100** documents

**Features**:
- ✅ High precision for semantic search
- ✅ Similar to ColBERT's proven approach
- ✅ Good for long documents
- ✅ Handles sparse matches well
- ❌ Can be dominated by single chunk
- ❌ Sensitive to embedding quality

**When to Use**:
- Question answering systems
- Semantic search (meaning > keywords)
- When precision > recall
- Finding specific information

**Example Query Types**:
- "What is the compression ratio of ColBERTv2?"
- "How does late interaction work?"

---

### 6. Layered Normalized Fusion

**Profile Name**: `layered_normalized`

**Description**: Normalizes semantic and lexical scores before fusion to handle scale mismatches. Tunable via query parameter.

**Scoring Formula**:
```
normalized_semantic = distance_scores / (sum(distance_scores) + 0.001)
normalized_lexical = text_scores / (sum(text_scores) + 0.001)

chunk_scores = normalized_semantic * α + normalized_lexical * (1 - α)

first_phase = sum(chunk_scores)
second_phase = sum(chunk_scores) * 100
```

**Query Parameters**:
- `alpha` (default=0.5): Weight for semantic vs lexical
  - α=1.0: Pure semantic search
  - α=0.0: Pure lexical search (BM25)
  - α=0.5: Balanced hybrid

**Second Phase Config**:
- Reranks top **100** documents

**Features**:
- ✅ Solves scale mismatch problems
- ✅ Tunable per-query
- ✅ Fair signal combination
- ✅ Better score calibration
- ❌ Normalization adds complexity
- ❌ Division operations are expensive

**When to Use**:
- BM25 and embeddings have different scales
- Need per-query tuning
- A/B testing different fusion strategies

**API Usage**:
```bash
# Balanced (default)
GET /query?ranking=layered_normalized&query=colbert

# Semantic-heavy (α=0.8)
POST /query
{
  "ranking": "layered_normalized",
  "query": "colbert",
  "ranking.features.query(alpha)": 0.8
}

# Lexical-heavy (α=0.2)
POST /query
{
  "ranking": "layered_normalized",
  "query": "colbert compression ratio",
  "ranking.features.query(alpha)": 0.2
}
```

---

### 7. Layered with Global Phase

**Profile Name**: `layered_with_global_phase`

**Description**: Adds global-phase ranking with cross-document normalization for final precision boost on top results.

**Scoring Formula**:
```
first_phase = sum(chunk_scores)

second_phase = sum(chunk_scores) + max_chunk_score * 0.5

global_phase = reciprocal_rank(sum(chunk_scores)) 
             + normalize_linear(max_chunk_score)
```

**Phase Configs**:
- Second phase: Top **100** documents
- Global phase: Top **20** documents

**Features**:
- ✅ Cross-document normalization
- ✅ Best precision on top-K
- ✅ Clean separation of concerns
- ✅ Uses relative ranking features
- ❌ Adds 20-50ms latency
- ❌ Only benefits top results

**When to Use**:
- Top-10 results matter most
- Willing to trade latency for quality
- Final polish on search results
- High-stakes ranking scenarios

**Latency Profile**:
```
First phase:   ~2ms
Second phase:  ~8ms
Global phase:  ~25ms
Total:         ~35ms
```

---

### 8. Layered ML Ranking

**Profile Name**: `layered_ml_ranking`

**Description**: Uses a trained LightGBM model in global phase for learned ranking based on user behavior data.

**Scoring Formula**:
```
first_phase = sum(chunk_scores)

second_phase = sum(chunk_scores) * 0.8 + max_score * 0.2

global_phase = lightgbm('my_ranking_model.json')
```

**Phase Configs**:
- Second phase: Top **200** documents
- Global phase: Top **50** documents

**Features Used by ML Model**:
- Chunk score statistics (max, min, avg, variance)
- Semantic similarity features
- Lexical relevance (BM25)
- Title relevance
- Document metadata features

**Features**:
- ✅ **Highest quality ranking** possible
- ✅ Learns non-linear patterns
- ✅ Adapts to user behavior
- ✅ Handles complex interactions
- ✅ Proven in production systems
- ❌ **Requires training data** (query-doc-label)
- ❌ 50-200ms latency
- ❌ Needs periodic retraining
- ❌ Black box (hard to explain)

**When to Use**:
- Have click/relevance data
- High-traffic applications
- Mature systems with data pipeline
- Quality > latency

**Training Data Requirements**:
```json
{
  "query": "why is colbert effective",
  "document_id": "abc123",
  "label": 1,  // 0=irrelevant, 1=relevant, 2=highly relevant
  "features": {
    "max_score": 0.89,
    "avg_score": 0.65,
    "title_score": 0.45,
    ...
  }
}
```

**Not Included by Default**: This profile requires:
1. Collecting training data (queries + labeled documents)
2. Training a LightGBM model
3. Deploying the model file to Vespa

See [Training LightGBM Models](#training-lightgbm-models) section below.

---

## Performance Comparison

### Latency

| Profile | First Phase | Second Phase | Global Phase | Total Latency |
|---------|-------------|--------------|--------------|---------------|
| Hybrid | 1ms | - | - | **~1-2ms** |
| Layered (Base) | 2ms | - | - | **~2-3ms** |
| + Second Phase | 2ms | 8ms | - | **~10ms** |
| + Diversity | 2ms | 12ms | - | **~14ms** |
| + MaxSim | 2ms | 10ms | - | **~12ms** |
| + Normalized | 2ms | 15ms | - | **~17ms** |
| + Global Phase | 2ms | 8ms | 25ms | **~35ms** |
| + ML Ranking | 2ms | 10ms | 100ms | **~112ms** |

*Note: Latencies are approximate and vary based on corpus size, query complexity, and hardware.*

### Quality vs Speed Trade-off

```
Quality
  ↑
  │                                    ● ML Ranking
  │                              ● Global Phase
  │                        ● MaxSim
  │                   ● Second Phase
  │              ● Diversity
  │         ● Normalized
  │    ● Layered Base
  │ ● Hybrid
  └─────────────────────────────────────────→ Speed
        Fast              Medium         Slow
```

### Precision Comparison (Typical)

Based on retrieval for top-5 results:

| Profile | Precision@5 | Recall@5 | MRR |
|---------|-------------|----------|-----|
| Hybrid | 0.65 | 0.72 | 0.58 |
| Layered Base | 0.72 | 0.68 | 0.65 |
| + Second Phase | 0.78 | 0.70 | 0.71 |
| + Diversity | 0.75 | 0.75 | 0.68 |
| + MaxSim | 0.82 | 0.65 | 0.76 |
| + Normalized | 0.76 | 0.71 | 0.69 |
| + Global Phase | 0.85 | 0.72 | 0.80 |
| + ML Ranking | 0.91 | 0.74 | 0.87 |

*Note: These are illustrative numbers. Actual metrics depend on your corpus and queries.*

---

## API Usage

### Basic Query (Hybrid)

```bash
curl -X GET "http://localhost:8002/query?query=why+is+colbert+effective"
```

### Layered Ranking

```bash
curl -X GET "http://localhost:8002/query-layered?query=why+is+colbert+effective"
```

### Layered Retriever (LangChain)

```bash
# Without threshold
curl -X GET "http://localhost:8002/query-layered-retriever?query=why+is+colbert+effective"

# With score threshold (0.6)
curl -X GET "http://localhost:8002/query-layered-retriever?query=why+is+colbert+effective&min_score=0.6"
```

### RAG Chain (with LLM)

```bash
# Get comprehensive answer from LLM
curl -X GET "http://localhost:8002/query-layered-retriever-chain?query=why+is+colbert+effective&min_score=0.5"
```

### Using Specific Ranking Profile

```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "yql": "select * from pdf where userQuery()",
    "ranking": "layered_with_maxsim",
    "query": "colbert compression",
    "hits": 10
  }'
```

---

## Choosing the Right Profile

### Decision Tree

```
START
  │
  ├─ Need <10ms latency?
  │   └─ YES → Use Hybrid or Layered Base
  │
  ├─ Have click/label data?
  │   └─ YES → Use ML Ranking (best quality)
  │
  ├─ Top-10 critical (e.g., chatbot)?
  │   └─ YES → Use Global Phase
  │
  ├─ Exploratory/research queries?
  │   └─ YES → Use Diversity
  │
  ├─ Semantic search (meaning > keywords)?
  │   └─ YES → Use MaxSim
  │
  ├─ BM25 and embeddings different scales?
  │   └─ YES → Use Normalized Fusion
  │
  └─ Default → Use Second Phase (best balance)
```

### Use Case Recommendations

| Use Case | Recommended Profile | Reason |
|----------|---------------------|--------|
| **Chatbot/QA** | Global Phase or ML | Top answers must be perfect |
| **Document search** | Second Phase | Good balance of speed/quality |
| **Semantic search** | MaxSim | Meaning-based matching |
| **Research tool** | Diversity | Comprehensive coverage |
| **Real-time search** | Layered Base | Low latency |
| **E-commerce** | ML Ranking | Learn from clicks |
| **Hybrid tuning** | Normalized Fusion | Experimentation |

### Progressive Enhancement Strategy

```python
# Week 1: Start simple
ranking = "layered_base"  # Establish baseline

# Week 2: Add second phase
ranking = "layered_with_second_phase"  # Tune weights

# Week 3: Test alternatives
A/B test: "layered_with_maxsim" vs "layered_with_diversity"

# Week 4: Add global phase if needed
ranking = "layered_with_global_phase"  # Polish top-K

# Month 2+: Collect data and train ML
ranking = "layered_ml_ranking"  # Learn from users
```

---

## Training LightGBM Models

### 1. Collect Training Data

Use Vespa's match features to collect training data:

```python
import json
from vespa.io import VespaQueryResponse

# Query with match features
response = vespa_app.query(
    query="colbert effectiveness",
    ranking="layered_ml_ranking",
    hits=20
)

# Extract features for each document
training_data = []
for hit in response.hits:
    features = hit["fields"]["matchfeatures"]
    training_data.append({
        "query": "colbert effectiveness",
        "doc_id": hit["fields"]["id"],
        "label": 0,  # TODO: Get actual label from clicks/judgments
        "features": features
    })
```

### 2. Train Model

```python
import lightgbm as lgb
import pandas as pd

# Prepare data
df = pd.DataFrame(training_data)
X = df[["max_score", "avg_score", "title_score", "score_variance"]]
y = df["label"]

# Train model
model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=100
)
model.fit(X, y, group=df.groupby("query").size().values)

# Save model
model.booster_.save_model("my_ranking_model.json")
```

### 3. Deploy to Vespa

Place the model in your application package:
```
app/
  models/
    my_ranking_model.json
```

---

## Advanced Topics

### Custom Ranking Functions

You can create custom functions in your rank profile:

```python
Function(
    name="custom_score",
    expression="""
        if(max_chunk_score > 0.8,
           sum(chunk_scores()) * 2.0,
           sum(chunk_scores()))
    """
)
```

### Query-Time Feature Overrides

```bash
curl -X POST "http://localhost:8002/search" \
  -d '{
    "query": "colbert",
    "ranking": "layered_normalized",
    "ranking.features.query(alpha)": 0.7
  }'
```

### Monitoring Ranking Quality

```python
# Log match features for analysis
for hit in response.hits:
    logger.info({
        "query": query,
        "doc_id": hit["fields"]["id"],
        "rank": hit["relevance"],
        "features": hit["fields"]["matchfeatures"]
    })
```

---

## Troubleshooting

### Low Precision

**Problem**: Getting irrelevant results

**Solutions**:
1. Increase `min_chunk_score` threshold
2. Switch to MaxSim profile for better precision
3. Add second-phase ranking
4. Tune weight parameters

### High Latency

**Problem**: Queries taking too long

**Solutions**:
1. Reduce `rerank_count` in second/global phase
2. Remove global phase
3. Use simpler ranking profile
4. Check corpus size and indexing

### Scale Mismatch

**Problem**: BM25 dominates or embeddings dominate

**Solutions**:
1. Use Normalized Fusion profile
2. Tune alpha parameter
3. Check embedding model quality
4. Verify BM25 is enabled on chunk field

---

## References

- [Vespa Ranking Documentation](https://docs.vespa.ai/en/ranking.html)
- [ColBERTv2 Paper](https://arxiv.org/abs/2112.01488)
- [Learning to Rank with LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html)

---

## Contributing

This is an experimental project. Suggestions and improvements are welcome!

## License

[Your License Here]

