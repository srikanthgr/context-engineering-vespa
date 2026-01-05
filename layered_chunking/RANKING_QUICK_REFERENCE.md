# Ranking Profiles Quick Reference

## TL;DR - Which Profile Should I Use?

| Your Need | Use This Profile | Latency | Quality |
|-----------|------------------|---------|---------|
| 🚀 **Fastest** | `hybrid` | 1-2ms | ⭐⭐⭐ |
| 🎯 **Best Balance** | `layered_with_second_phase` | ~10ms | ⭐⭐⭐⭐ |
| 🔍 **Highest Precision** | `layered_with_maxsim` | ~12ms | ⭐⭐⭐⭐⭐ |
| 🌈 **Most Diverse** | `layered_with_diversity` | ~14ms | ⭐⭐⭐⭐ |
| 🎓 **Best for Learning** | `layered_ml_ranking` | ~112ms | ⭐⭐⭐⭐⭐⭐ |
| 👑 **Top-10 Perfection** | `layered_with_global_phase` | ~35ms | ⭐⭐⭐⭐⭐ |
| ⚖️ **Fair Hybrid** | `layered_normalized` | ~17ms | ⭐⭐⭐⭐ |
| 📊 **Simple Baseline** | `layered_base` | 2-3ms | ⭐⭐⭐⭐ |

---

## Profile Comparison at a Glance

```
SPEED  ←────────────────────────────────────────→  QUALITY

Hybrid          Layered     Second    MaxSim   Global    ML
(1ms)           (3ms)      Phase     (12ms)   Phase    Ranking
                           (10ms)              (35ms)   (112ms)
  │               │          │          │        │         │
  └─ Baseline     └─ Prod    └─ Best   └─ QA   └─ Top-K └─ Ultimate
                    Default    Balance   System  Polish   Quality
```

---

## API Endpoints

```bash
# 1. Hybrid (fastest)
GET /query?query=colbert

# 2. Layered base
GET /query-layered?query=colbert

# 3. Layered retriever (with threshold)
GET /query-layered-retriever?query=colbert&min_score=0.6

# 4. RAG chain (with LLM)
GET /query-layered-retriever-chain?query=colbert&min_score=0.5
```

---

## Scoring Formulas (Simplified)

| Profile | First Phase | Second Phase | Global Phase |
|---------|-------------|--------------|--------------|
| **Hybrid** | `chunks + title + max(sim)` | - | - |
| **Layered** | `sum(chunk_scores)` | - | - |
| **+ Second** | `sum(chunk_scores)` | `↑ + title + max_sim` | - |
| **+ Diversity** | `sum(chunk_scores)` | `↑ + variance + spread` | - |
| **+ MaxSim** | `sum(chunk_scores)` | `max(sim)*10 + sum(sim)` | - |
| **+ Normalized** | `sum(normalized_scores)` | `↑ * 100` | - |
| **+ Global** | `sum(chunk_scores)` | `↑ + max_chunk` | `reciprocal_rank + normalize` |
| **+ ML** | `sum(chunk_scores)` | `↑ + max` | `lightgbm(model)` |

---

## Decision Tree (5 seconds)

```
Need <10ms?          → hybrid or layered_base
Have ML model?       → layered_ml_ranking
Top-10 critical?     → layered_with_global_phase
Semantic search?     → layered_with_maxsim
Exploratory?         → layered_with_diversity
Scale issues?        → layered_normalized
Default choice       → layered_with_second_phase ✓
```

---

## Feature Comparison

| Feature | Hybrid | Layered | +Second | +Diversity | +MaxSim | +Normalized | +Global | +ML |
|---------|--------|---------|---------|------------|---------|-------------|---------|-----|
| **Dual Filtering** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Title Boost** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Diversity** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Tunable** | ❌ | ❌ | ⚠️ | ❌ | ❌ | ✅ | ❌ | ✅ |
| **Cross-Doc** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Learns** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Production Ready** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |

Legend: ✅ Yes | ❌ No | ⚠️ Partial/Complex

---

## Common Use Cases

### Chatbot/QA System
```python
ranking = "layered_with_global_phase"  # or ML if you have data
min_chunk_score = 0.7  # High threshold for precision
```

### Document Search
```python
ranking = "layered_with_second_phase"
min_chunk_score = 0.5  # Balanced
```

### Research Tool
```python
ranking = "layered_with_diversity"
min_chunk_score = 0.4  # Lower threshold for coverage
```

### Real-time Search
```python
ranking = "layered_base"
min_chunk_score = 0.6
```

---

## Tuning Cheat Sheet

### Increase Precision (fewer but better results)
- ✅ Raise `min_chunk_score` (try 0.7-0.9)
- ✅ Use `layered_with_maxsim`
- ✅ Add second/global phase
- ✅ Reduce `chunks_per_page` (try 1-2)

### Increase Recall (more results)
- ✅ Lower `min_chunk_score` (try 0.3-0.5)
- ✅ Use `layered_base` or `layered_with_diversity`
- ✅ Increase `chunks_per_page` (try 5)
- ✅ Increase `pages` retrieved

### Reduce Latency
- ✅ Remove global phase
- ✅ Reduce `rerank_count` (50 → 20)
- ✅ Use simpler profile (hybrid/layered_base)

### Improve Top-K Quality
- ✅ Add global phase
- ✅ Increase `rerank_count` (100 → 200)
- ✅ Use `layered_with_maxsim`

---

## Troubleshooting One-Liners

| Problem | Solution |
|---------|----------|
| Too slow | Use `hybrid` or reduce `rerank_count` |
| Irrelevant results | Increase `min_chunk_score` or use MaxSim |
| Missing relevant docs | Lower `min_chunk_score` or use Diversity |
| BM25 dominates | Use Normalized with `alpha=0.7` |
| Embeddings dominate | Use Normalized with `alpha=0.3` |
| Top-3 wrong | Add global phase |

---

## Monitoring

### Log This for Each Query
```python
{
  "query": str,
  "profile": str,
  "latency_ms": float,
  "num_results": int,
  "top_scores": [float],
  "user_clicked": int  # Position of clicked result
}
```

### Key Metrics
- **MRR** (Mean Reciprocal Rank): Average 1/position of first relevant result
- **Precision@K**: % relevant in top-K results
- **Latency P95**: 95th percentile latency
- **CTR** (Click-Through Rate): % queries with click

---

## Further Reading

📖 Full guide: `RANKING_PROFILES_GUIDE.md`  
🔗 Vespa docs: https://docs.vespa.ai/en/ranking.html  
📄 ColBERT paper: https://arxiv.org/abs/2112.01488

