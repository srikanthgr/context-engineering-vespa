# Vespa Layered Ranking with LangChain

A comprehensive implementation demonstrating Vespa's **Layered Ranking** feature for RAG (Retrieval Augmented Generation) applications, with side-by-side comparison to traditional hybrid ranking.

## Table of Contents

- [Overview](#overview)
- [What is Layered Ranking?](#what-is-layered-ranking)
- [Architecture](#architecture)
- [Key Observations](#key-observations)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Ranking Profiles Explained](#ranking-profiles-explained)
- [Performance Comparison](#performance-comparison)
- [Advanced Topics](#advanced-topics)

## Additional Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[OBSERVATIONS.md](OBSERVATIONS.md)** - Technical deep dive and debugging insights
- **[HYBRID_VS_LAYERED_COMPARISON.md](HYBRID_VS_LAYERED_COMPARISON.md)** - Detailed comparison: Hybrid + Python filtering vs Layered ranking
- **[retriever_comparison.py](retriever_comparison.py)** - Practical implementation of both approaches as LangChain retrievers

## Overview

This application provides a FastAPI-based service that:
- Loads PDF documents using LangChain
- Chunks documents intelligently
- Embeds chunks using Sentence Transformers (E5 model)
- Stores in Vespa with streaming mode
- Provides two ranking approaches:
  - **Hybrid Ranking**: Traditional semantic + lexical search
  - **Layered Ranking**: Advanced chunk-level scoring with best-chunk selection

## What is Layered Ranking?

Layered ranking is a Vespa feature that optimizes RAG applications by:

1. **Scoring at the chunk level**: Each chunk within a document gets individual scores
2. **Selecting top chunks**: Uses `top(N, scores)` to identify the N best chunks
3. **Document-level ranking**: Documents ranked by the sum of all chunk scores
4. **Efficient context**: Only the most relevant chunks are identified for LLM consumption

### Benefits

- ✅ **Better LLM Quality**: Send only the most relevant chunks to your LLM
- ✅ **Reduced Latency**: Less data transferred over the network
- ✅ **Smaller Context Windows**: Optimize token usage
- ✅ **Cost Savings**: Fewer tokens sent to LLM APIs
- ✅ **Hybrid Matching**: Combines semantic similarity AND keyword relevance

## Architecture

```
┌─────────────┐
│   PDFs      │
└──────┬──────┘
       │ LangChain Loader
       ↓
┌─────────────────────┐
│  Chunked Documents  │
└──────┬──────────────┘
       │ E5 Embeddings
       ↓
┌─────────────────────┐
│  Vespa (Streaming)  │
│  ┌───────────────┐  │
│  │ Hybrid Rank   │  │
│  │ Profile       │  │
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │ Layered Rank  │  │
│  │ Profile       │  │
│  └───────────────┘  │
└─────────────────────┘
       │
       ↓
┌─────────────────────┐
│   FastAPI Service   │
│  /query (hybrid)    │
│  /query-layered     │
└─────────────────────┘
```

## Key Observations

### 1. Chunk Selection Differences

**Hybrid Ranking** returns ALL chunks from matched documents:
```json
"similarities": {
  "0": 0.8033,
  "1": 0.8129,
  "2": 0.8369,
  "3": 0.8344
}
```
All 4 chunks present, ranked by semantic similarity only.

**Layered Ranking** returns ONLY chunks matching BOTH semantic AND lexical criteria:
```json
"best_chunks": {
  "0": 0.8903,
  "2": 0.8375,
  "3": 0.9197
}
```
Chunk 1 is missing because it didn't match the query keywords!

### 2. Relevance Score Interpretation

**Hybrid Ranking**: Score typically 0-1 range
```
score = nativeRank(title) + nativeRank(chunks) + max(similarities)
```
Takes the **maximum** similarity across chunks.

**Layered Ranking**: Score can be > 1 (cumulative)
```
score = sum(chunk_scores) where each chunk_score = distance_score + bm25_score
```
**Sums** all qualifying chunk scores - more relevant chunks = higher score!

### 3. Hybrid vs Pure Semantic Matching

Layered ranking's `join` operation is key:
```
chunk_scores = join(my_distance_scores, my_text_scores, f(a,b)(a+b))
```

This means chunks must have:
- ✅ Semantic similarity (vector distance) **AND**
- ✅ Lexical match (BM25 keyword presence)

Pure semantic systems may retrieve contextually similar but lexically unrelated chunks. Layered ranking ensures both dimensions align.

### 4. Document Re-ranking

In our tests with query "why is colbert effective?":

| Approach | Top Result | Relevance | Chunks Returned |
|----------|-----------|-----------|-----------------|
| Hybrid | Page 13 | 1.033 | 4 chunks (all) |
| Layered | Page 29 | 2.647 | 3 chunks (filtered) |

Page 29 scored higher because its chunks had **both** high semantic similarity and strong keyword matches.

## Installation

### Prerequisites

```bash
# Python 3.8+
# Docker (for Vespa)
```

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or use uv (recommended)
uv pip install -r requirements.txt
```

### Required Python Packages

```
pyvespa>=0.40.0
fastapi>=0.104.0
uvicorn>=0.24.0
langchain>=0.1.0
langchain-text-splitters>=0.3.4
pypdf>=5.1.0
sentence-transformers>=2.2.0
```

## Usage

### 1. Start the Application

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8005
```

The application will:
1. Start Vespa in Docker container
2. Deploy the schema with both rank profiles
3. Wait for document feeding
4. Expose API endpoints

### 2. Feed Documents

```bash
POST http://localhost:8005/feed-pdf

Body:
{
  "url": "https://arxiv.org/pdf/2112.01488",
  "authors": "Omar Khattab, Matei Zaharia"
}
```

### 3. Query with Hybrid Ranking

```bash
GET http://localhost:8005/query?query=why is colbert effective
```

### 4. Query with Layered Ranking

```bash
GET http://localhost:8005/query-layered?query=why is colbert effective
```

## API Endpoints

### `POST /feed-pdf`

Feed a PDF document into Vespa.

**Request Body:**
```json
{
  "url": "https://example.com/paper.pdf",
  "authors": "Author One, Author Two"
}
```

**Response:**
```json
{
  "status": "success",
  "chunks_fed": 45,
  "groupname": "jo-bergum"
}
```

### `GET /query`

Search using hybrid ranking profile.

**Parameters:**
- `query` (required): Search query text

**Example:**
```bash
curl "http://localhost:8005/query?query=what%20is%20colbert"
```

**Response:**
```json
{
  "root": {
    "children": [
      {
        "id": "...",
        "relevance": 1.033,
        "fields": {
          "matchfeatures": {
            "similarities": {...},
            "nativeRank(chunks)": 0.154,
            "closest(embedding)": {...}
          },
          "title": "ColBERTv2...",
          "chunks": ["...", "...", "...", "..."]
        }
      }
    ]
  }
}
```

### `GET /query-layered`

Search using layered ranking profile with chunk-level scoring.

**Parameters:**
- `query` (required): Search query text

**Example:**
```bash
curl "http://localhost:8005/query-layered?query=what%20is%20colbert"
```

**Response:**
```json
{
  "root": {
    "children": [
      {
        "id": "...",
        "relevance": 2.647,
        "fields": {
          "matchfeatures": {
            "best_chunks": {
              "0": 0.890,
              "2": 0.837,
              "3": 0.919
            },
            "chunk_scores": {...},
            "my_distance": {...},
            "my_distance_scores": {...},
            "my_text_scores": {...}
          },
          "title": "ColBERTv2...",
          "chunks": ["...", "...", "...", "..."]
        }
      }
    ]
  }
}
```

## Ranking Profiles Explained

### Hybrid Ranking Profile

```python
rank_profile = RankProfile(
    name="hybrid",
    inputs=[("query(q)", "tensor(x[384])")],
    functions=[
        Function(
            name="similarities",
            expression="cosine_similarity(query(q), attribute(embedding), x)",
        )
    ],
    first_phase="nativeRank(title) + nativeRank(chunks) + reduce(similarities, max, chunk)",
    match_features=[
        "closest(embedding)",
        "similarities",
        "nativeRank(chunks)",
        "nativeRank(title)",
        "elementSimilarity(chunks)",
    ],
)
```

**How it works:**
1. Computes cosine similarity between query and each chunk embedding
2. Takes the **maximum** similarity score across all chunks
3. Adds native BM25 scores from title and chunks fields
4. Returns ALL chunks from matched documents

**Best for:**
- Exploratory search
- When you want all context from matched documents
- Simpler scoring logic

### Layered Ranking Profile

```python
rank_profile = RankProfile(
    name="layeredranking",
    inputs=[("query(q)", "tensor(x[384])")],
    functions=[
        Function(
            name="my_distance",
            expression="euclidean_distance(query(q), attribute(embedding), x)",
        ),
        Function(
            name="my_distance_scores",
            expression="1 / (1+my_distance)",
        ),
        Function(
            name="my_text_scores",
            expression="elementwise(bm25(chunks), chunk, float)",
        ),
        Function(
            name="chunk_scores",
            expression="join(my_distance_scores, my_text_scores, f(a,b)(a+b))",
        ),
        Function(
            name="best_chunks",
            expression="top(3, chunk_scores)",
        ),
    ],
    first_phase="sum(chunk_scores())",
    match_features=[
        "my_distance",
        "my_distance_scores",
        "my_text_scores",
        "chunk_scores",
        "best_chunks",
    ],
    summary_features=["best_chunks"]
)
```

**How it works:**

1. **Distance Calculation**: `euclidean_distance(query(q), attribute(embedding), x)`
   - Computes Euclidean distance between query embedding and each chunk
   - Returns `tensor(chunk{})` with one distance per chunk

2. **Distance Scoring**: `1 / (1+my_distance)`
   - Converts distance to similarity score (closer = higher score)
   - Returns `tensor(chunk{})` with normalized scores

3. **Text Scoring**: `elementwise(bm25(chunks), chunk, float)`
   - Computes BM25 score for each chunk based on keyword matches
   - Returns `tensor(chunk{})` - **only for chunks with keyword matches**

4. **Hybrid Scoring**: `join(my_distance_scores, my_text_scores, f(a,b)(a+b))`
   - ⚠️ **Critical**: `join` only keeps chunks present in BOTH tensors
   - Chunks without BM25 scores are excluded
   - Adds distance and text scores element-wise

5. **Best Chunks Selection**: `top(3, chunk_scores)`
   - Selects the 3 highest-scoring chunks
   - Returns sparse tensor with only top 3

6. **Document Ranking**: `sum(chunk_scores())`
   - Documents ranked by cumulative score of all qualifying chunks
   - More relevant chunks = higher document score

**Best for:**
- RAG applications
- When LLM context window is limited
- Ensuring both semantic AND lexical relevance
- Cost optimization (fewer tokens to LLM)

## Performance Comparison

### Query: "why is colbert effective?"

#### Hybrid Ranking Results
```json
{
  "relevance": 1.033,
  "title": "ColBERTv2: Effective and Efficient Retrieval...",
  "page": 13,
  "chunks_returned": 4,
  "matchfeatures": {
    "similarities": {
      "0": 0.803,
      "1": 0.813,
      "2": 0.837,  // Highest similarity
      "3": 0.834
    }
  }
}
```
**Score breakdown:**
- Takes max similarity: 0.837
- Plus native ranking scores
- **Total: 1.033**

#### Layered Ranking Results
```json
{
  "relevance": 2.647,
  "title": "ColBERTv2: Effective and Efficient Retrieval...",
  "page": 29,
  "chunks_returned": 3,  // Chunk 1 filtered out
  "matchfeatures": {
    "best_chunks": {
      "0": 0.890,  // distance + bm25
      "2": 0.837,  // distance + bm25
      "3": 0.920   // distance + bm25
    },
    "my_distance_scores": {
      "0": 0.189,
      "1": 0.179,  // Has distance score
      "2": 0.184,
      "3": 0.192
    },
    "my_text_scores": {
      "0": 0.701,
      // Chunk 1 missing - no keyword match!
      "2": 0.654,
      "3": 0.728
    }
  }
}
```
**Score breakdown:**
- Chunk 0: 0.890 = (0.189 distance + 0.701 bm25)
- Chunk 2: 0.837 = (0.184 distance + 0.654 bm25)
- Chunk 3: 0.920 = (0.192 distance + 0.728 bm25)
- Chunk 1: **Excluded** (no BM25 score)
- **Total: 0.890 + 0.837 + 0.920 = 2.647**

### Key Insights

| Aspect | Hybrid | Layered |
|--------|--------|---------|
| **Scoring Method** | Max of similarities | Sum of chunk scores |
| **Chunk Filtering** | None (all chunks) | Requires semantic + lexical |
| **Score Range** | Typically 0-1 | Cumulative (can be >> 1) |
| **Chunks Returned** | All | Only qualifying chunks |
| **Best For** | Exploration | RAG/LLM feeding |
| **Precision** | Moderate | High (dual criteria) |

## Advanced Topics

### Understanding the `join` Operation

The `join` operation in Vespa is crucial to layered ranking:

```
join(tensor_A, tensor_B, f(a,b)(a+b))
```

**Behavior:**
- Only includes dimensions (chunks) present in **BOTH** tensors
- Applies the lambda function `f(a,b)(a+b)` to matching elements
- This is why chunks without BM25 scores disappear!

**Example:**
```
tensor_A (distance_scores): {0: 0.2, 1: 0.18, 2: 0.19, 3: 0.21}
tensor_B (text_scores):     {0: 0.7,          2: 0.65, 3: 0.73}
                                   // Chunk 1 missing!

Result (chunk_scores):      {0: 0.9,          2: 0.84, 3: 0.94}
```

### Tensor Dimension Compatibility

**Critical Insight**: Query and document embeddings have different tensor structures:

- **Query**: `tensor(x[384])` - 1D vector with 384 dimensions
- **Document**: `tensor(chunk{}, x[384])` - 2D tensor (multiple chunks × 384 dimensions)

**Distance functions must handle this:**
```python
# ✅ Correct - computes distance per chunk
euclidean_distance(query(q), attribute(embedding), x)
# Returns: tensor(chunk{})

# ❌ Wrong - dimension mismatch
euclidean_distance(query(q), attribute(embedding))
```

The third parameter `x` tells Vespa to compute distance along the `x` dimension for each chunk.

### Customizing Top-K Chunks

To select more or fewer chunks, modify the `best_chunks` function:

```python
# Top 5 chunks
Function(
    name="best_chunks",
    expression="top(5, chunk_scores)",
)

# Top 10 chunks
Function(
    name="best_chunks",
    expression="top(10, chunk_scores)",
)
```

### Alternative Scoring Strategies

**Use dot product instead of euclidean distance:**
```python
Function(
    name="chunk_vector_scores",
    expression="reduce(query(q) * attribute(embedding), sum, x)",
)
```

**Use cosine similarity:**
```python
Function(
    name="chunk_vector_scores",
    expression="cosine_similarity(query(q), attribute(embedding), x)",
)
```

**Weighted combination:**
```python
Function(
    name="chunk_scores",
    expression="join(my_distance_scores, my_text_scores, f(a,b)(0.7*a + 0.3*b))",
)
```
This weights semantic similarity (70%) higher than lexical match (30%).

## Limitations and Known Issues

### 1. `select-elements-by` Not Supported in Pyvespa

**The Problem:**

The official Vespa documentation shows how to automatically return only selected chunks:

```
field myChunks type array<string> {
    indexing: index | summary
    summary {
        select-elements-by: best_chunks
    }
}
```

This would tell Vespa to return ONLY the chunks identified by the `best_chunks` function, reducing network transfer.

However, **pyvespa currently doesn't support this syntax**:

```python
Field(
    name="chunks",
    type="array<string>",
    indexing=["index", "summary"],
    summary="select-elements-by: best_chunks"  # ❌ Causes deployment error
)
```

**Error:** `Failed parsing schema: Encountered " "{" "{"" at line X`

**Current Behavior:**

All chunks are returned in the response, even though only some are selected as "best":

```json
{
  "chunks": [
    "chunk 0 content...",
    "chunk 1 content...",
    "chunk 2 content...",
    "chunk 3 content..."
  ],
  "matchfeatures": {
    "best_chunks": {
      "0": 0.890,
      "2": 0.837,
      "3": 0.920
    }
  }
}
```

All 4 chunks are transferred, but only chunks 0, 2, and 3 are marked as "best".

**The Workaround: Manual Python Filtering**

You must filter chunks in your Python code using the `best_chunks` indices:

```python
def _extract_best_chunks(self, fields: dict) -> List[str]:
    """
    Extract only the chunks identified as 'best' by Vespa.
    
    This mimics what native Vespa would do with:
    summary { select-elements-by: best_chunks }
    """
    match_features = fields.get("matchfeatures", {})
    best_chunks_dict = match_features.get("best_chunks", {})
    
    if not best_chunks_dict:
        # Fallback if best_chunks not in match_features
        return fields["chunks"]
    
    all_chunks = fields["chunks"]
    
    # Build list of (chunk_content, score) tuples
    selected = []
    for idx_str, score in best_chunks_dict.items():
        idx = int(idx_str)
        if 0 <= idx < len(all_chunks):
            selected.append((all_chunks[idx], score))
    
    # Sort by score descending
    selected.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the chunk text
    return [chunk for chunk, _ in selected]

# Usage in retriever
for hit in response.hits:
    fields = hit["fields"]
    
    # Get only the best chunks
    best_chunks = self._extract_best_chunks(fields)
    
    # Use for LLM context
    page_content = " ### ".join(best_chunks)
```

**What This Solves:**

| Aspect | Native select-elements-by | Python Workaround |
|--------|--------------------------|-------------------|
| **Chunk Selection** | ✅ Correct | ✅ Correct (same result) |
| **Network Transfer** | ✅ Only best chunks | ❌ All chunks transferred |
| **Processing** | ✅ Done in Vespa | ❌ Done in Python |
| **Memory Usage** | ✅ Minimal | ❌ Higher |
| **Latency** | ✅ Lower | ⚠️ Slightly higher |

**Impact Analysis:**

For a typical query returning 5 documents with 10 chunks each (500 chars/chunk):

- **With select-elements-by (native)**: 5 docs × 3 chunks × 500 chars = ~7.5KB transferred
- **Without (pyvespa workaround)**: 5 docs × 10 chunks × 500 chars = ~25KB transferred
- **3.3x more data over the network**

**When This Matters:**

Low Impact (workaround is fine):
- ✅ Small documents (few chunks)
- ✅ Low query volume
- ✅ Fast network
- ✅ Development/testing

High Impact (significant overhead):
- ❌ Large documents (many chunks)
- ❌ High query volume (1000s QPS)
- ❌ Distributed deployment
- ❌ Limited bandwidth

**Alternative: Use Native Vespa Schema Files**

If network optimization is critical, bypass pyvespa for schema definition:

```python
# Create native .sd file
with open("schemas/pdf.sd", "w") as f:
    f.write("""
    schema pdf {
        document pdf {
            field chunks type array<string> {
                indexing: index | summary
            }
        }
        
        field chunks type array<string> {
            indexing: index | summary
            summary {
                select-elements-by: best_chunks
            }
        }
        
        rank-profile layeredranking {
            function best_chunks() {
                expression: top(3, chunk_scores)
            }
        }
    }
    """)

# Deploy using schema directory
package = ApplicationPackage(
    name="myapp",
    schema_dir="schemas/"
)
```

This gives you the full power of native Vespa schema syntax, including `select-elements-by`.

**Bottom Line:**

The Python workaround **solves the chunk selection problem** (you get the right chunks for your LLM) but **doesn't solve the network efficiency problem** (all chunks still transferred). For most RAG applications, this is acceptable - the quality improvement from layered ranking's dual-criteria filtering outweighs the network overhead.

### 2. Parameter Name Consistency

Three places must use the same query parameter name:

1. YQL nearestNeighbor: `nearestNeighbor(embedding, q)`
2. Rank profile input: `inputs=[("query(q)", ...)]`
3. Query body: `"input.query(q)": f'embed(...)'`

Mismatch causes: `Expected 'query(X)' to be a tensor, but it is a string`

### 3. Streaming Mode Group Name

Documents must be fed with a `groupname` parameter for streaming mode:
```python
vespa_app.feed_data_point(
    schema="pdf",
    data_id=doc_id,
    fields=doc_fields,
    groupname="jo-bergum"  # Required!
)
```

And queried with the same:
```python
vespa_app.query(
    yql="...",
    groupname="jo-bergum",  # Must match!
    ...
)
```

## Troubleshooting

### Error: "Field 'X' does not exist"
- Ensure fieldset only references document fields, not computed fields
- Add `is_document_field=False` to computed fields like embeddings

### Error: "unknown type named 'array'"
- Use explicit types: `array<string>`, not `array`
- Use explicit types: `map<string,string>`, not `map`

### Error: "Address already in use"
- Check if another service is using the port
- Kill existing process: `lsof -ti:8005 | xargs kill -9`
- Change port in `uvicorn.run()`

### Error: Tensor dimension mismatch
- Ensure distance functions specify the dimension to compute over
- Use: `euclidean_distance(query(q), attribute(embedding), x)`
- Not: `euclidean_distance(query(q), attribute(embedding))`

## References

- [Vespa Layered Ranking Blog Post](https://blog.vespa.ai/introducing-layered-ranking-for-rag-applications/)
- [Vespa Tensor User Guide](https://docs.vespa.ai/en/tensor-user-guide.html)
- [Pyvespa Documentation](https://pyvespa.readthedocs.io/)
- [LangChain Documentation](https://python.langchain.com/)

## Contributing

Contributions welcome! Areas of interest:
- Supporting `select-elements-by` in pyvespa
- Additional ranking strategies
- Performance benchmarks
- Multi-modal embeddings

## License

MIT License - feel free to use and modify for your projects.

---

**Built with ❤️ using Vespa, LangChain, and FastAPI**

