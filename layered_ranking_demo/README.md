# Layered Ranking Demo for Vespa RAG Applications

This demo showcases **Layered Ranking** - a powerful feature introduced in Vespa 8.530 that allows you to select the best chunks from documents after ranking, rather than returning all chunks.

## What is Layered Ranking?

Traditional RAG systems face a dilemma:
- **Chunk-level documents**: Good for retrieving only relevant chunks, but loses document context and requires metadata duplication
- **Multi-chunk documents**: Preserves context, but returns ALL chunks from each document, leading to large context windows with irrelevant information

**Layered ranking solves this** by:
1. Ranking documents at the document level (first layer)
2. Ranking chunks within each document (second layer)
3. Selecting only the top N chunks from each document to return

This gives you the best of both worlds: document-level context for ranking, but only the most relevant chunks for the LLM context window.

## Key Benefits

✅ **Optimal Context Window Usage**: Only send the most relevant chunks to your LLM  
✅ **Reduced Bandwidth**: Don't transfer entire documents when only a few chunks are needed  
✅ **Better Quality**: Focus the LLM's attention on the most relevant information  
✅ **Scalability**: Works efficiently even with very large documents  

## How It Works

### Traditional Ranking (Without Layered Ranking)
```
Document 1 (Score: 0.95)
  ├─ Chunk 1 (Score: 0.8)  ← Returned
  ├─ Chunk 2 (Score: 0.3)   ← Returned (but not very relevant!)
  └─ Chunk 3 (Score: 0.7)  ← Returned

Document 2 (Score: 0.85)
  ├─ Chunk 1 (Score: 0.9)  ← Returned
  ├─ Chunk 2 (Score: 0.2)  ← Returned (not relevant!)
  └─ Chunk 3 (Score: 0.1)  ← Returned (not relevant!)
```

**Result**: 6 chunks sent to LLM, but only 2 are highly relevant.

### Layered Ranking (With Chunk Selection)
```
Document 1 (Score: 0.95)
  ├─ Chunk 1 (Score: 0.8)  ← Selected (top 3)
  ├─ Chunk 2 (Score: 0.3)
  └─ Chunk 3 (Score: 0.7)  ← Selected (top 3)

Document 2 (Score: 0.85)
  ├─ Chunk 1 (Score: 0.9)  ← Selected (top 3)
  ├─ Chunk 2 (Score: 0.2)
  └─ Chunk 3 (Score: 0.1)
```

**Result**: 3 chunks sent to LLM, all highly relevant!

## Installation

1. Install dependencies:
```bash
pip install pyvespa sentence-transformers numpy
```

2. Make sure Docker is running (Vespa runs in Docker)

## Running the Demo

```bash
python layered_ranking_demo/main.py
```

The demo will:
1. Create sample documents with multiple chunks each
2. Deploy two Vespa instances (one with layered ranking, one without)
3. Run the same queries on both
4. Show the difference in results

## Code Structure

### Schema with Layered Ranking

The key is in the rank profile and document summary:

```python
# Rank profile calculates scores per chunk and selects top N
rank_profile = RankProfile(
    name="layered_ranking",
    functions=[
        Function(name="chunk_scores", expression="..."),
        Function(name="best_chunks", expression="top(3, chunk_scores())")
    ],
    first_phase="sum(chunk_scores())"
)

# Document summary uses select-elements-by to filter chunks
doc_summary = DocumentSummary(
    name="best_chunks_only",
    summaries=[
        Summary(
            name="chunks",
            source="chunks",
            select_elements_by="best_chunks"  # ← This is the magic!
        )
    ]
)
```

### Schema without Layered Ranking

```python
# Rank profile calculates scores but doesn't select chunks
rank_profile = RankProfile(
    name="traditional_ranking",
    functions=[
        Function(name="chunk_scores", expression="...")
    ],
    first_phase="sum(chunk_scores())"
)

# Document summary returns ALL chunks
doc_summary = DocumentSummary(
    name="all_chunks",
    summaries=[
        Summary(name="chunks", source="chunks")  # ← Returns everything
    ]
)
```

## Understanding the Output

When you run the demo, you'll see:

1. **Traditional Ranking Results**:
   - All chunks from each document are returned
   - More chunks = larger context window
   - May include less relevant chunks

2. **Layered Ranking Results**:
   - Only top 3 chunks per document are returned
   - Smaller, more focused context window
   - Only the most relevant chunks

## Real-World Use Cases

- **Large Documents**: When documents have hundreds of chunks, you don't want to send all of them
- **Token Optimization**: LLM APIs charge by token - fewer chunks = lower costs
- **Quality Improvement**: Focus LLM attention on the most relevant information
- **Bandwidth Savings**: Especially important for high query rates

## References

- [Vespa Blog: Introducing Layered Ranking for RAG Applications](https://blog.vespa.ai/introducing-layered-ranking-for-rag-applications/)
- [Vespa Documentation](https://docs.vespa.ai/)

## Example Output

```
Query: 'machine learning neural networks'

WITHOUT LAYERED RANKING (Traditional)
───────────────────────────────────────
📄 Result 1: Python Machine Learning Guide
   Relevance Score: 2.3456
   Number of Chunks Returned: 5
   Chunks:
      1. Machine learning is a subset of artificial intelligence...
      2. Python is the most popular programming language...
      3. Libraries like scikit-learn provide powerful tools...
      4. TensorFlow and PyTorch are deep learning frameworks...
      5. Data preprocessing is crucial for successful...

WITH LAYERED RANKING (Selects Best Chunks)
───────────────────────────────────────────
📄 Result 1: Python Machine Learning Guide
   Relevance Score: 2.3456
   Number of Chunks Returned: 3
   Chunks:
      1. Machine learning is a subset of artificial intelligence...
      2. TensorFlow and PyTorch are deep learning frameworks...
      3. Feature engineering can significantly improve model...
```

Notice how layered ranking returns only 3 chunks instead of 5, focusing on the most relevant ones!

