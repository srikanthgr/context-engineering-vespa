# Technical Observations: Layered Ranking Implementation

This document captures the key technical insights and observations discovered during the implementation and debugging of Vespa's layered ranking feature.

## Table of Contents
- [Query Parameter Consistency](#query-parameter-consistency)
- [Tensor Dimension Compatibility](#tensor-dimension-compatibility)
- [The Join Operation Behavior](#the-join-operation-behavior)
- [Chunk Filtering Mechanism](#chunk-filtering-mechanism)
- [Relevance Score Interpretation](#relevance-score-interpretation)
- [Match Features vs Summary Features](#match-features-vs-summary-features)
- [PyVespa Limitations](#pyvespa-limitations)

---

## Query Parameter Consistency

### The Problem
One of the most subtle bugs we encountered was query parameter name mismatches across three different locations.

### The Three Critical Places

**1. Rank Profile Input Declaration:**
```python
inputs = [("query(q)", "tensor(x[384])")]
```

**2. YQL nearestNeighbor Query:**
```python
yql="... nearestNeighbor(embedding,q)"
                                   ^^^ must match rank profile
```

**3. Request Body Parameter:**
```python
body={
    "input.query(q)": f'embed(e5, "{q}")'
              ^^^ must match rank profile
}
```

### Error When Mismatched

```
vespa.exceptions.VespaError: Expected 'query(q)' to be a tensor, 
but it is a string. The full string value is 'embed(e5, "...")'.
```

### Why This Happens
When Vespa can't find the declared tensor input in the rank profile, it treats the embed expression as a literal string instead of evaluating it.

### Solution
**Use consistent naming everywhere**. Convention is to use `query(q)` but you can use any name (e.g., `query(embedding)`) as long as it's consistent.

---

## Tensor Dimension Compatibility

### The Challenge
Query embeddings and document chunk embeddings have fundamentally different tensor structures.

### Tensor Structures

**Query Embedding:**
```python
tensor(x[384])
```
- 1-dimensional
- Single vector of 384 floats
- Example: `{x:0: 0.1, x:1: 0.2, ..., x:383: 0.9}`

**Document Chunk Embeddings:**
```python
tensor(chunk{}, x[384])
```
- 2-dimensional
- Multiple chunks, each with 384 floats
- Example: `{chunk:0: {x:0: 0.1, ...}, chunk:1: {x:0: 0.3, ...}}`

### Distance Function Requirements

**Wrong Approach (Dimension Mismatch):**
```python
# ❌ This fails!
expression="euclidean_distance(query(q), attribute(embedding))"
```
Error: `euclidean_distance expects both arguments to have the 'x' dimension with same size, but input types were tensor() and tensor(chunk{},x[384])`

**Correct Approach:**
```python
# ✅ Specify the dimension to compute over
expression="euclidean_distance(query(q), attribute(embedding), x)"
                                                               ^^^
```

### How It Works
The third parameter `x` tells Vespa:
1. Compute distance along the `x` dimension
2. Do this separately for each chunk
3. Return `tensor(chunk{})` with one distance value per chunk

### Alternative: Dot Product
```python
expression="reduce(query(q) * attribute(embedding), sum, x)"
```
- Multiplies query with each chunk embedding element-wise along `x`
- Reduces by summing over the `x` dimension
- Returns `tensor(chunk{})` with similarity scores

---

## The Join Operation Behavior

### Critical Understanding
The `join` operation is the secret sauce that makes layered ranking filter chunks.

### Function Signature
```python
join(tensor_A, tensor_B, f(a,b)(expression))
```

### Key Behavior
**Only dimensions present in BOTH tensors are included in the result.**

### Example

```python
# Input tensors
distance_scores = tensor(chunk{}) {
    chunk:0: 0.19,
    chunk:1: 0.18,
    chunk:2: 0.20,
    chunk:3: 0.21
}

text_scores = tensor(chunk{}) {
    chunk:0: 0.70,
    // chunk:1 MISSING - no BM25 match!
    chunk:2: 0.65,
    chunk:3: 0.73
}

# Join operation
chunk_scores = join(distance_scores, text_scores, f(a,b)(a+b))

# Result
chunk_scores = tensor(chunk{}) {
    chunk:0: 0.89,   // 0.19 + 0.70
    // chunk:1 GONE!
    chunk:2: 0.85,   // 0.20 + 0.65
    chunk:3: 0.94    // 0.21 + 0.73
}
```

### Why Chunk 1 Disappeared
1. `my_text_scores` uses `elementwise(bm25(chunks), chunk, float)`
2. BM25 only returns scores for chunks with keyword matches
3. If chunk 1 has no query keywords, it's not in `my_text_scores`
4. The `join` operation excludes it from the result

### This is a Feature!
It ensures chunks must satisfy **BOTH**:
- Semantic similarity (embeddings)
- Lexical relevance (keywords)

---

## Chunk Filtering Mechanism

### Hybrid vs Layered Behavior

**Hybrid Ranking:**
```python
first_phase="nativeRank(title) + nativeRank(chunks) + reduce(similarities, max, chunk)"
```
- All chunks are considered
- Takes the **maximum** similarity
- No filtering occurs

**Layered Ranking:**
```python
functions=[
    Function("my_distance_scores", "..."),      # tensor(chunk{}) - ALL chunks
    Function("my_text_scores", "bm25(...)"),     # tensor(chunk{}) - MATCHING chunks only
    Function("chunk_scores", "join(...)"),       # tensor(chunk{}) - INTERSECTION
]
first_phase="sum(chunk_scores())"
```
- Only chunks in the intersection are summed
- **Implicit filtering** via the join operation

### Real Example Data

Query: "why is colbert effective?"

**Document with 4 chunks:**

| Chunk | Semantic Match | Has Keywords | Included? |
|-------|----------------|--------------|-----------|
| 0 | ✅ (0.189) | ✅ (0.701) | ✅ Yes |
| 1 | ✅ (0.179) | ❌ None | ❌ No |
| 2 | ✅ (0.184) | ✅ (0.654) | ✅ Yes |
| 3 | ✅ (0.192) | ✅ (0.728) | ✅ Yes |

**Chunk 1 Analysis:**
- Has semantic similarity (embeddings matched)
- Likely contains related concepts
- But lacks the specific keywords "colbert" or "effective"
- Gets filtered out by the join operation

---

## Relevance Score Interpretation

### Why Layered Ranking Scores > 1

**Hybrid Ranking Score:**
```
score = max(similarities) + nativeRank(title) + nativeRank(chunks)
      ≈ 0.84 + 0.04 + 0.15
      = 1.03
```
Typically stays in 0-2 range.

**Layered Ranking Score:**
```
score = sum(chunk_scores)
      = chunk_0 + chunk_2 + chunk_3
      = 0.89 + 0.84 + 0.92
      = 2.65
```
Can be much larger!

### Why This Makes Sense

**More relevant chunks = higher cumulative score**

Consider two documents:
- **Doc A**: 1 highly relevant chunk (score 0.95)
- **Doc B**: 3 moderately relevant chunks (scores 0.75, 0.70, 0.72)

**Hybrid ranking:**
- Doc A: 0.95 (takes max)
- Doc B: 0.75 (takes max)
- **Doc A wins**

**Layered ranking:**
- Doc A: 0.95
- Doc B: 0.75 + 0.70 + 0.72 = 2.17
- **Doc B wins**

Doc B has more total relevant content - this is often what you want for RAG!

### When Higher Scores Are Better

✅ **RAG applications**: More context for LLM
✅ **Comprehensive answers**: Documents covering topic from multiple angles
✅ **Tutorial content**: Step-by-step explanations

### When Lower Scores Might Be Better

⚠️ **Pinpoint accuracy**: Need one specific fact
⚠️ **Concise answers**: Minimal context preferred

Solution: Use `reduce(chunk_scores, max, chunk)` for pinpoint, `sum(chunk_scores)` for comprehensive.

---

## Match Features vs Summary Features

### Two Different Feature Types in Vespa

**match_features:**
- Returned in query results for each matched document
- Used for debugging and analysis
- Accessible via `response['fields']['matchfeatures']`

**summary_features:**
- Also returned in results
- Traditionally used for different purposes
- Accessible via `response['fields']['summaryfeatures']`

### In Layered Ranking

```python
rank_profile = RankProfile(
    name="layeredranking",
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

### Why We Include Both

**Without match_features:**
```json
{
  "relevance": 2.647,
  // No debugging information!
}
```

**With match_features:**
```json
{
  "relevance": 2.647,
  "matchfeatures": {
    "my_distance": {"0": 4.28, "1": 4.59, "2": 4.44, "3": 4.22},
    "my_distance_scores": {"0": 0.189, "1": 0.179, "2": 0.184, "3": 0.192},
    "my_text_scores": {"0": 0.701, "2": 0.654, "3": 0.728},
    "chunk_scores": {"0": 0.890, "2": 0.838, "3": 0.920},
    "best_chunks": {"0": 0.890, "2": 0.838, "3": 0.920}
  }
}
```

Now you can see:
- Why chunk 1 is missing (no text score)
- How each component contributes to the final score
- Which chunks were selected as "best"

### The Missing Piece

Ideally, we'd also have:
```python
summary="select-elements-by: best_chunks"
```

This would return only the top chunks in the `chunks` field. However, this is **not yet supported in pyvespa**.

---

## PyVespa Limitations

### 1. `select-elements-by` Syntax

**Official Vespa Schema (.sd file):**
```
field chunks type array<string> {
    indexing: index | summary
    summary {
        select-elements-by: best_chunks
    }
}
```

**PyVespa Equivalent:**
```python
Field(
    name="chunks",
    type="array<string>",
    indexing=["index", "summary"],
    summary="select-elements-by: best_chunks"  # ❌ Causes schema parsing error!
)
```

**Error:**
```
RuntimeError: Deployment failed, code: 400
message: 'Invalid application: Failed parsing schema from 'pdf.sd': 
Encountered " "{" "{"" at line X, column Y.'
```

### 2. Document vs Computed Fields

**Problem:**
```python
fieldsets=[
    FieldSet(name="default", fields=["title", "chunks", "embedding"])
]
```

**Error:**
```
Invalid application: For schema 'pdf': 
Field 'chunks' in fieldset 'default' does not exist.
```

**Why:**
- `chunks` is a computed field (not in the document struct)
- Embeddings are derived from chunks via `indexing: input chunks | embed e5`
- Fieldsets can only reference document fields

**Solution:**
```python
Field(
    name="chunks",
    type="array<string>",
    indexing=["index", "summary"],
    is_document_field=False  # ✅ Mark as computed
)

fieldsets=[
    FieldSet(name="default", fields=["title"])  # ✅ Only document fields
]
```

### 3. Type Specifications

**Wrong:**
```python
Field(name="authors", type="array")           # ❌
Field(name="metadata", type="map")            # ❌
```

**Correct:**
```python
Field(name="authors", type="array<string>")   # ✅
Field(name="metadata", type="map<string,string>")  # ✅
```

Vespa requires explicit element types for collections.

### 4. Rank Profile Returns

**Wrong:**
```python
def create_rank_profile():
    return RankProfile(...)  # ❌ Returns single object
```

**Correct:**
```python
def create_rank_profile():
    return [RankProfile(...)]  # ✅ Returns list

# Usage
for rank_profile in create_rank_profile():
    schema.add_rank_profile(rank_profile)
```

---

## Best Practices Learned

### 1. Always Use match_features for Debugging
Include ALL intermediate functions in match_features during development:
```python
match_features=[
    "my_distance",
    "my_distance_scores",
    "my_text_scores",
    "chunk_scores",
    "best_chunks",
]
```

### 2. Name Query Parameters Consistently
Pick a convention and stick with it everywhere:
- ✅ Use `q` everywhere
- ✅ Or use `embedding` everywhere
- ❌ Don't mix them!

### 3. Understand Your Tensor Dimensions
Before writing ranking expressions:
1. Print out tensor shapes
2. Understand which dimensions you're reducing over
3. Test with small examples

### 4. Validate join Operations
When using `join`, explicitly verify:
- What dimensions are in tensor A?
- What dimensions are in tensor B?
- What do you expect in the result?

### 5. Test with Verbose Queries
Start with queries that will match many chunks, then try edge cases:
```python
# Good starting query (matches many chunks)
"what is colbert and how does it work"

# Edge case (might match few chunks)
"xylophone manufacturing"
```

---

## Future Improvements

### 1. PyVespa Support for select-elements-by
This would enable automatic filtering of returned chunks:
```python
Field(
    name="chunks",
    type="array<string>",
    summary=Summary(select_elements_by="best_chunks")  # Future API?
)
```

### 2. Configurable Scoring Weights
Make it easy to tune semantic vs lexical balance:
```python
chunk_scores = join(
    my_distance_scores, 
    my_text_scores, 
    f(a,b)(config.semantic_weight*a + config.lexical_weight*b)
)
```

### 3. Second-Phase Re-ranking
Add cross-encoder re-ranking:
```python
second_phase=SecondPhaseRanking(
    expression="cross_encoder_score()",
    rerank_count=10
)
```

### 4. Dynamic Top-K Selection
Adjust number of chunks based on query complexity:
```python
expression="top(if(query(complex), 5, 3), chunk_scores)"
```

---

## Conclusion

These observations highlight that layered ranking is:
1. **Powerful**: Combines multiple relevance signals
2. **Subtle**: Many interacting components require careful setup
3. **Flexible**: Can be adapted for different use cases
4. **Production-Ready**: Once configured, it's robust and efficient

The key insight is that **the join operation** creates implicit filtering that ensures chunks satisfy multiple criteria - this is what makes layered ranking superior for RAG applications.

---

**Last Updated**: November 2025  
**Contributors**: Based on debugging sessions and real-world implementation experience

