# Layered Chunking: How Dual-Criteria Filtering Cuts 40% of Irrelevant Context in RAG Pipelines

---

## Table of Contents

- [Why Your RAG Pipeline Is Quietly Failing](#why-your-rag-pipeline-is-quietly-failing)
- [The Real Cost of "Good Enough" Retrieval](#the-real-cost-of-good-enough-retrieval)
- [The Problem: Semantic Similarity Lies](#the-problem-semantic-similarity-lies)
- [The Fix: Require Both Signals](#the-fix-require-both-signals)
- [Architecture: How It Works in Vespa](#architecture-how-it-works-in-vespa)
  - [The Ranking Profile](#the-ranking-profile)
  - [Document-Level Ranking](#document-level-ranking)
- [The Retriever: Connecting to LangChain](#the-retriever-connecting-to-langchain)
- [Advanced Ranking Profiles](#advanced-ranking-profiles)
  - [Second-Phase Re-ranking](#second-phase-re-ranking)
  - [Diversity-Aware Ranking](#diversity-aware-ranking)
  - [Normalized Score Fusion](#normalized-score-fusion)
- [Benchmarks](#benchmarks)
  - [Ranking Profile Comparison](#ranking-profile-comparison)
- [Gotchas and Lessons Learned](#gotchas-and-lessons-learned)
- [When to Use Layered Chunking](#when-to-use-layered-chunking)
- [Getting Started](#getting-started)

---

## Why Your RAG Pipeline Is Quietly Failing

Here's something that doesn't show up in your demo but ruins production: **most RAG systems retrieve chunks that are semantically adjacent to the answer but not actually the answer.**

You've tuned your embeddings. You've picked a chunking strategy. You've wired up a vector database. Your retrieval "works" — the top-K chunks come back, they look vaguely related, and the LLM generates something plausible. Ship it.

Except the LLM is hallucinating details. Or hedging with "based on the provided context, it appears that..." Or confidently synthesizing an answer from a data table that has nothing to do with the user's question. The retrieval step looked fine. The chunks were semantically similar. But *similar isn't the same as relevant.*

This is the central failure mode of embedding-only retrieval: **cosine similarity captures topical proximity, not informational relevance.** A chunk that lives in the same embedding neighborhood as your query — because it's from the same paper section, uses similar vocabulary, or discusses adjacent concepts — will score highly even if it contains zero useful information for the actual question asked.

And it's not an edge case. In our testing across information retrieval research papers, **roughly 1 in 5 chunks selected by embedding similarity alone were false positives** — topically adjacent but informationally irrelevant.

## The Real Cost of "Good Enough" Retrieval

Why does this matter? Three reasons that compound on each other:

**1. LLM output quality is gated on input quality.** This is the most important and least discussed constraint in RAG system design. A frontier model with three precisely relevant chunks will outperform the same model with five chunks where two are noise. The irrelevant chunks don't just waste tokens — they actively confuse the model. They introduce contradictory signals, dilute the relevant context, and give the model material to hallucinate from. Garbage in, confident garbage out.

**2. Token costs scale with retrieval sloppiness.** Every irrelevant chunk you send to the LLM costs real money. At scale, the difference between sending 3 relevant chunks and 5 chunks (with 2 being noise) is a 40% increase in input tokens per query — with no quality benefit. Across millions of queries, that's significant spend for worse results.

**3. User trust erodes silently.** When a RAG system returns a plausible-sounding but subtly wrong answer because the LLM synthesized from irrelevant context, the user doesn't file a bug report. They just stop trusting the system. By the time you notice in your metrics, the damage is done.

The fix isn't a better embedding model. It's not a smarter chunking strategy. **It's requiring chunks to prove their relevance through multiple independent signals before they ever reach the LLM.**

That's what layered chunking does.

---

## The Problem: Semantic Similarity Lies

Let's make this concrete. Consider a query against a corpus of IR research papers: *"why is colbert effective?"*

A standard retrieval pipeline computes cosine similarity between the query embedding and each chunk embedding, then picks the top-K. Here's what actually happens with a real 4-chunk document from the ColBERTv2 paper:

| Chunk | Content Summary | Cosine Similarity | Actually Relevant? |
|-------|----------------|-------------------|-------------------|
| 0 | Token clustering analysis | 0.803 | Yes |
| 1 | Random sample queries table | 0.813 | **No** |
| 2 | ColBERT evaluation metrics | 0.837 | Yes |
| 3 | Comparison with baselines | 0.834 | Yes |

Chunk 1 scores *higher* than chunk 0 on semantic similarity. It lives in the same embedding neighborhood because it's from the same paper section. But it's a table of query examples — it says nothing about why ColBERT is effective.

A similarity-only retriever selects chunks 1, 2, and 3. The LLM gets a table of random queries where it should have gotten analysis content. This is the topical proximity trap in action — chunk 1 is *about ColBERT*, it's *from the ColBERT paper*, but it answers a completely different question.

Now multiply this across every query your system handles. One in five chunks is a false positive like this. Your LLM is working with polluted context on 20% of its inputs, and you can't see it because the retrieval metrics look fine.

## The Fix: Require Both Signals

Layered chunking scores each chunk on two independent axes:

1. **Semantic score**: How close is the chunk embedding to the query embedding?
2. **Lexical score**: Does the chunk contain the actual query terms (via BM25)?

The key insight is in how these scores are combined. Rather than adding them for all chunks, a `join` operation computes the combined score **only for chunks that have both signals**. Chunks with high embedding similarity but zero keyword overlap are dropped entirely.

Here's the same document under layered chunking:

| Chunk | Semantic Score | BM25 Score | Combined | Selected? |
|-------|---------------|------------|----------|-----------|
| 0 | 0.189 | 0.701 | 0.890 | Yes |
| 1 | 0.179 | — (no keywords) | **dropped** | No |
| 2 | 0.184 | 0.654 | 0.838 | Yes |
| 3 | 0.192 | 0.728 | 0.920 | Yes |

Chunk 1 has no BM25 score because it doesn't contain "colbert" or "effective." The join operation excludes it from the result tensor entirely. The LLM now gets three chunks that are both semantically and lexically relevant.

## Architecture: How It Works in Vespa

The implementation uses Vespa's tensor algebra to perform chunk-level scoring and filtering server-side. Here's the data model:

```
Document
├── id, title, url, authors (metadata)
├── chunks: array<string>          ← text chunks from PDF pages
└── embedding: tensor(chunk{}, x[384])  ← one E5 embedding per chunk
```

The `embedding` field is a 2D tensor: the `chunk{}` dimension indexes over chunks (variable count), and `x[384]` holds the 384-dimensional E5-small-v2 embedding for each chunk. Vespa computes these at feed time from the `chunks` field using its built-in Hugging Face embedder.

### The Ranking Profile

The layered ranking profile defines five functions that form a pipeline:

```python
RankProfile(
    name="layeredranking",
    inputs=[("query(q)", "tensor(x[384])")],
    functions=[
        # 1. Euclidean distance from query to each chunk embedding
        Function("my_distance",
            "euclidean_distance(query(q), attribute(embedding), x)"),

        # 2. Convert distances to similarity scores (0-1 range)
        Function("my_distance_scores",
            "1 / (1 + my_distance)"),

        # 3. BM25 score per chunk (only chunks with keyword matches)
        Function("my_text_scores",
            "elementwise(bm25(chunks), chunk, float)"),

        # 4. Join: combine scores, DROP chunks missing from either tensor
        Function("chunk_scores",
            "join(my_distance_scores, my_text_scores, f(a,b)(a+b))"),

        # 5. Select top 3 chunks by combined score
        Function("best_chunks",
            "top(3, chunk_scores)"),
    ],
    first_phase="sum(chunk_scores())",
)
```

Let's trace through each step.

**Step 1 — Distance computation.** `euclidean_distance(query(q), attribute(embedding), x)` computes the distance along the `x` dimension separately for each chunk. The query is `tensor(x[384])` (1D), the document embedding is `tensor(chunk{}, x[384])` (2D). The third argument `x` tells Vespa which dimension to reduce over. The result is `tensor(chunk{})` — one distance value per chunk.

**Step 2 — Score normalization.** `1 / (1 + distance)` converts unbounded distances into a 0-1 similarity score. Every chunk gets a semantic score.

**Step 3 — BM25 per chunk.** `elementwise(bm25(chunks), chunk, float)` distributes the field-level BM25 score to individual chunks. Critically, **only chunks containing query terms get a score**. If a chunk has none of the query keywords, it simply doesn't appear in this tensor.

**Step 4 — The join.** This is where filtering happens. `join(my_distance_scores, my_text_scores, f(a,b)(a+b))` performs an inner join on the `chunk{}` dimension. Only chunks present in **both** tensors get a combined score. Chunks with semantic similarity but no keyword match are silently excluded.

```
distance_scores: {0: 0.189, 1: 0.179, 2: 0.184, 3: 0.192}  ← all 4 chunks
text_scores:     {0: 0.701,           2: 0.654, 3: 0.728}    ← only 3 chunks

join result:     {0: 0.890,           2: 0.838, 3: 0.920}    ← chunk 1 gone
```

**Step 5 — Top-K selection.** `top(3, chunk_scores)` selects the highest-scoring chunks from those that survived the join.

### Document-Level Ranking

The first-phase expression `sum(chunk_scores())` ranks documents by the cumulative score of all qualifying chunks. This is a deliberate design choice for RAG:

- A document with **one great chunk** scores lower than a document with **three good chunks**
- For RAG, you want the document with the most relevant content overall, not just the best single passage
- A document scoring 2.65 (three qualifying chunks) outranks one scoring 0.95 (one great chunk)

This is the opposite of hybrid ranking, which uses `reduce(similarities, max, chunk)` and rewards documents based only on their single best chunk.

## The Retriever: Connecting to LangChain

The `VespaStreamingLayeredRetriever` integrates with LangChain as a drop-in retriever:

```python
class VespaStreamingLayeredRetriever(BaseRetriever):
    app: Vespa
    user: str
    pages: int = 5
    chunks_per_page: int = 3
    min_chunk_score: float = 0.0

    def _get_relevant_documents(self, query: str) -> List[Document]:
        response = self.app.query(
            yql=("select id, url, title, page, authors, chunks from pdf "
                 "where userQuery() or "
                 "({targetHits:20}nearestNeighbor(embedding,q))"),
            groupname=self.user,
            ranking="layeredranking",
            query=query,
            hits=self.pages,
            body={
                "presentation.format.tensors": "short-value",
                "input.query(q)": f'embed(e5, "query: {query} ")',
            },
        )
        return self._parse_response(response)
```

The query uses both `userQuery()` (keyword matching) and `nearestNeighbor` (embedding search). Vespa's layered ranking profile then scores each chunk on both signals and returns the `best_chunks` indices in `matchfeatures`.

Chunk extraction reads the `best_chunks` tensor from match features:

```python
def _get_best_chunks(self, hit_fields: dict) -> List[tuple]:
    match_features = hit_fields["matchfeatures"]
    best_chunks = match_features["best_chunks"]
    chunks = hit_fields["chunks"]

    chunks_with_scores = []
    for idx_str, score in best_chunks.items():
        idx = int(idx_str)
        if idx < len(chunks):
            chunks_with_scores.append((chunks[idx], score))

    return sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)
```

The `best_chunks` value is a dict like `{"0": 0.890, "2": 0.838, "3": 0.920}` — chunk indices as keys, combined scores as values. The retriever maps these back to the original chunk text.

## Advanced Ranking Profiles

The base layered profile can be extended with second-phase re-ranking for more sophisticated scoring:

### Second-Phase Re-ranking

```python
second_phase=SecondPhaseRanking(
    expression="sum(chunk_scores()) * 0.7 + title_score * 0.2 + max_similarity * 0.1",
    rerank_count=100
)
```

This re-ranks the top 100 documents from the first phase, incorporating title relevance and peak similarity as additional signals. The first phase acts as a fast filter; the second phase refines ordering.

### Diversity-Aware Ranking

```python
Function("chunk_spread",
    "reduce(my_distance_scores, max, chunk) - reduce(my_distance_scores, min, chunk)")

second_phase=SecondPhaseRanking(
    expression="sum(chunk_scores()) * 0.7 + chunk_spread * 2.0 + avg_chunk_score * 0.3",
    rerank_count=50
)
```

The `chunk_spread` function measures how varied the chunk scores are within a document. Documents with diverse chunk scores cover the topic from multiple angles — useful when you want the LLM to synthesize a comprehensive answer.

### Normalized Score Fusion

```python
Function("normalized_semantic",
    "my_distance_scores / (reduce(my_distance_scores, sum, chunk) + 0.001)")
Function("normalized_lexical",
    "my_text_scores / (reduce(my_text_scores, sum, chunk) + 0.001)")
Function("chunk_scores",
    "join(normalized_semantic, normalized_lexical, f(a,b)(a * 0.5 + b * 0.5))")
```

This normalizes both score distributions before combining them, preventing one signal from dominating. The `0.001` epsilon avoids division by zero. Useful when your semantic and lexical score ranges differ significantly.

## Benchmarks

We tested across 100 queries on a corpus of information retrieval research papers (5 PDFs, ~150 pages, ~600 chunks total):

| Metric | Hybrid (all chunks) | Hybrid + Python filter | Layered Ranking |
|--------|---------------------|----------------------|-----------------|
| Precision@3 | 0.67 | 0.78 | **0.85** |
| Recall@3 | **0.85** | 0.72 | 0.79 |
| MRR | 0.58 | — | **0.65** |
| Avg latency | 70ms | 72ms | **66ms** |
| App CPU | 45% | 45% | **12%** |
| False positive chunks | ~20% | ~12% | **~8%** |

Key takeaways:

- **Precision jumped from 0.67 to 0.85** — fewer irrelevant chunks in the top 3
- **Latency dropped 6%** — Vespa's C++ tensor operations are faster than Python post-processing
- **App server CPU dropped 73%** — scoring and filtering moved to Vespa
- **Recall traded off slightly** (0.85 → 0.79) — the dual-criteria filter is stricter, which can exclude borderline-relevant chunks that lack exact keywords

The recall tradeoff is real but manageable. For RAG applications where precision matters more than recall (you'd rather send 3 great chunks than 5 mediocre ones to the LLM), layered chunking wins convincingly.

### Ranking Profile Comparison

Different profiles suit different use cases:

| Profile | Latency | P@5 | Best For |
|---------|---------|-----|----------|
| Hybrid (baseline) | 1-2ms | 0.65 | Fast prototyping |
| Layered | 2-3ms | 0.72 | Production RAG |
| + Second Phase | 10ms | 0.78 | Balanced quality/speed |
| + MaxSim | 12ms | 0.82 | Semantic-heavy queries |
| + Diversity | 14ms | 0.75 | Exploratory questions |
| + Normalized | ~10ms | 0.78 | Mixed score distributions |

## Gotchas and Lessons Learned

### 1. Query parameter names must match in three places

The same name (e.g., `q`) must appear in the rank profile input declaration, the YQL `nearestNeighbor` clause, and the request body:

```python
# Rank profile
inputs=[("query(q)", "tensor(x[384])")]

# YQL
"nearestNeighbor(embedding, q)"
#                          ^ must match

# Request body
"input.query(q)": 'embed(e5, "...")'
#            ^ must match
```

A mismatch produces a confusing error: *"Expected 'query(q)' to be a tensor, but it is a string."* Vespa treats the unresolved embed expression as a literal string.

### 2. Distance functions need the dimension argument

When computing distance between a 1D query tensor and a 2D chunk tensor, you must specify which dimension to reduce over:

```python
# Wrong — dimension mismatch error
"euclidean_distance(query(q), attribute(embedding))"

# Correct — compute distance along x, separately per chunk
"euclidean_distance(query(q), attribute(embedding), x)"
```

### 3. PyVespa doesn't support `select-elements-by` (yet)

In native Vespa schema files, you can configure a field to only return elements selected by a ranking function:

```
field chunks type array<string> {
    summary { select-elements-by: best_chunks }
}
```

This would eliminate network transfer of unselected chunks. PyVespa doesn't support this syntax, so the current workaround is to transfer all chunks and filter client-side. The filtering still happens server-side for *scoring* purposes — you just pay extra on the wire.

### 4. The join can be too strict for sparse content

If your documents have very short chunks or your queries use uncommon terms, the BM25 side of the join may return empty tensors, dropping all chunks. Monitor for queries that return zero results and consider a fallback:

```python
if not documents:
    # Fall back to similarity-only retrieval
    response = self.app.query(ranking="hybrid", ...)
```

## When to Use Layered Chunking

**Use it when:**
- You're building a production RAG pipeline and precision matters
- Your documents have multiple chunks per page (the dual-criteria filter shines with 3+ chunks)
- You want to reduce LLM token costs by sending fewer, better chunks
- Your queries contain specific terms (not just vibes-based semantic queries)

**Skip it when:**
- You're prototyping and iteration speed matters more than quality
- Your documents are single-chunk (nothing to filter)
- Your queries are purely semantic ("find me something about machine learning feelings")
- You need maximum recall and can't afford to miss borderline-relevant chunks

## Getting Started

The full implementation is available with a FastAPI service, LangChain integration, and evaluation tools. The core migration from a standard hybrid retriever is minimal:

1. Add the `layeredranking` rank profile to your schema
2. Change `ranking="hybrid"` to `ranking="layeredranking"` in your query
3. Read chunk indices from `matchfeatures["best_chunks"]` instead of `matchfeatures["similarities"]`
4. Remove any Python threshold filtering — the join handles it

That's four lines of change for a 40% reduction in irrelevant chunks.

## The Bigger Picture

RAG systems have a quality ceiling, and it's not the LLM — it's what you feed it. We've spent enormous effort on better embedding models, smarter chunking strategies, and more capable language models. But the weakest link in most pipelines is the gap between "semantically similar" and "actually relevant."

Layered chunking closes that gap by treating retrieval as an evidence problem: a chunk should only reach the LLM if it can demonstrate relevance through multiple independent signals. The `join` operation is the mechanism, but the principle is what matters. Don't trust a single similarity score. Require corroboration.

The 40% reduction in false positives isn't just a number — it's the difference between an LLM that confidently synthesizes from relevant context and one that hallucinates from noise. Your users can tell the difference, even if your eval metrics can't.

---

*Built with [Vespa](https://vespa.ai), [PyVespa](https://pyvespa.readthedocs.io/), [LangChain](https://python.langchain.com/), and E5-small-v2 embeddings.*
