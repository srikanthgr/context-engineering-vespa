# Quick Start Guide - Layered Ranking Demo

Get up and running with Vespa layered ranking in 5 minutes!

## Installation

```bash
# Clone and navigate to project
cd langchain_streaming_mode

# Install dependencies
pip install pyvespa fastapi uvicorn langchain langchain-text-splitters pypdf sentence-transformers

# Ensure Docker is running
docker ps
```

## Start the Service

```bash
python main.py
```

Wait for the service to start. You'll see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8005
```

## Feed a Document

```bash
curl -X POST "http://localhost:8005/feed-pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://arxiv.org/pdf/2112.01488",
    "authors": "Omar Khattab, Matei Zaharia"
  }'
```

Expected response:
```json
{
  "status": "success",
  "chunks_fed": 45,
  "groupname": "jo-bergum"
}
```

## Query Comparison

### 1. Hybrid Ranking (Traditional)

```bash
curl -G "http://localhost:8005/query" \
  --data-urlencode "query=why is colbert effective"
```

**What you get:**
- All chunks from matched documents
- Simpler scoring (max similarity)
- Match features showing similarities for all chunks

### 2. Layered Ranking (Advanced)

```bash
curl -G "http://localhost:8005/query-layered" \
  --data-urlencode "query=why is colbert effective"
```

**What you get:**
- Only chunks matching BOTH semantic and lexical criteria
- Detailed match features:
  - `best_chunks`: Top 3 chunks with their scores
  - `chunk_scores`: Combined semantic + lexical scores
  - `my_distance`: Euclidean distances per chunk
  - `my_distance_scores`: Normalized distance scores
  - `my_text_scores`: BM25 keyword scores
- Higher cumulative relevance scores

## Example Comparison

### Query: "what is token clustering in colbert"

**Hybrid Result:**
```json
{
  "relevance": 0.95,
  "chunks": 4,
  "matchfeatures": {
    "similarities": {
      "0": 0.78,
      "1": 0.81,
      "2": 0.85,
      "3": 0.79
    }
  }
}
```

**Layered Result:**
```json
{
  "relevance": 2.34,
  "chunks": 3,
  "matchfeatures": {
    "best_chunks": {
      "0": 0.89,
      "2": 0.76,
      "3": 0.69
    },
    "my_distance_scores": {
      "0": 0.21,
      "1": 0.19,
      "2": 0.20,
      "3": 0.18
    },
    "my_text_scores": {
      "0": 0.68,
      // Chunk 1 missing!
      "2": 0.56,
      "3": 0.51
    }
  }
}
```

**Notice:**
- Chunk 1 is excluded in layered ranking (no keyword match)
- Layered ranking has higher cumulative score
- More detailed scoring breakdown

## Key Observations

### Why Chunk 1 Disappeared?

The `join` operation requires chunks to exist in BOTH tensors:
- ✅ Chunk has semantic similarity (embedding match)
- ✅ Chunk has lexical match (BM25 keyword match)
- ❌ Missing either = chunk excluded

This is a **feature**, not a bug! It ensures chunks are relevant in multiple dimensions.

### Why Higher Relevance Scores?

Layered ranking **sums** all chunk scores:
```
relevance = chunk_0_score + chunk_2_score + chunk_3_score
          = 0.89 + 0.76 + 0.69
          = 2.34
```

Hybrid ranking takes the **max** similarity plus native ranks:
```
relevance = max(similarities) + nativeRank(title) + nativeRank(chunks)
          = 0.85 + 0.05 + 0.05
          = 0.95
```

More relevant chunks = higher score in layered ranking!

## Use Cases

### When to Use Hybrid Ranking

- ✅ Exploratory search
- ✅ You want all context from documents
- ✅ Simpler scoring model
- ✅ User-facing search interfaces

### When to Use Layered Ranking

- ✅ RAG applications with LLMs
- ✅ Limited context windows
- ✅ Need both semantic AND keyword relevance
- ✅ Cost optimization (fewer tokens)
- ✅ Higher precision requirements

## Common Queries

### Feed Multiple PDFs

```bash
# ColBERT paper
curl -X POST "http://localhost:8005/feed-pdf" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/pdf/2112.01488", "authors": "Omar Khattab, Matei Zaharia"}'

# BERT paper
curl -X POST "http://localhost:8005/feed-pdf" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/pdf/1810.04805", "authors": "Jacob Devlin"}'

# GPT-2 paper
curl -X POST "http://localhost:8005/feed-pdf" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", "authors": "Alec Radford"}'
```

### Compare Results

```bash
# Save hybrid results
curl -G "http://localhost:8005/query" \
  --data-urlencode "query=how does attention work" \
  > hybrid_results.json

# Save layered results
curl -G "http://localhost:8005/query-layered" \
  --data-urlencode "query=how does attention work" \
  > layered_results.json

# Compare
diff hybrid_results.json layered_results.json
```

### Python Client Example

```python
import requests
import json

# Feed a document
feed_response = requests.post(
    "http://localhost:8005/feed-pdf",
    json={
        "url": "https://arxiv.org/pdf/2112.01488",
        "authors": "Omar Khattab, Matei Zaharia"
    }
)
print(f"Fed {feed_response.json()['chunks_fed']} chunks")

# Query with hybrid ranking
hybrid_results = requests.get(
    "http://localhost:8005/query",
    params={"query": "what is late interaction"}
).json()

# Query with layered ranking
layered_results = requests.get(
    "http://localhost:8005/query-layered",
    params={"query": "what is late interaction"}
).json()

# Compare top results
print("Hybrid relevance:", hybrid_results['root']['children'][0]['relevance'])
print("Layered relevance:", layered_results['root']['children'][0]['relevance'])

# Analyze best chunks
best_chunks = layered_results['root']['children'][0]['fields']['matchfeatures']['best_chunks']
print(f"Best chunks selected: {list(best_chunks.keys())}")
```

## Troubleshooting

### Port Already in Use
```bash
# Find and kill process
lsof -ti:8005 | xargs kill -9

# Or change port in main.py
uvicorn.run(app, host="0.0.0.0", port=8006)  # Different port
```

### Docker Not Running
```bash
# Start Docker daemon
open -a Docker  # macOS

# Or start Docker service
sudo systemctl start docker  # Linux
```

### PDF Download Fails
- Check URL is accessible
- Try downloading manually first
- Ensure sufficient disk space
- Check firewall settings

### No Results Returned
- Verify documents were fed successfully
- Check groupname matches in query
- Try simpler queries first
- Inspect Vespa logs: `docker logs <container_id>`

## Next Steps

1. **Read the full README**: `README.md` for comprehensive documentation
2. **Experiment with scoring**: Modify rank profile functions
3. **Adjust chunk selection**: Change `top(3, ...)` to select more/fewer chunks
4. **Try different embeddings**: Replace E5 with other models
5. **Add reranking**: Implement second-phase ranking with cross-encoders

## Performance Tips

### For Faster Feeding

```python
# Batch feed documents
for pdf in pdf_list:
    asyncio.create_task(feed_pdf(pdf))
```

### For Better Accuracy

- Use larger embedding models (768d instead of 384d)
- Tune chunk size (currently 512 chars)
- Adjust chunk overlap (currently 50 chars)
- Experiment with different distance metrics

### For Lower Latency

- Reduce `targetHits` in YQL
- Use smaller embedding models
- Enable result caching
- Deploy Vespa with more resources

## Resources

- **Main README**: `README.md`
- **Vespa Docs**: https://docs.vespa.ai
- **Blog Post**: https://blog.vespa.ai/introducing-layered-ranking-for-rag-applications/
- **LangChain Docs**: https://python.langchain.com

---

Happy searching! 🚀

