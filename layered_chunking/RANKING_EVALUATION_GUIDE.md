# Ranking Profile Evaluation Guide

A comprehensive guide to evaluating and comparing the 8 ranking profiles in this system.

## Table of Contents

- [Overview](#overview)
- [Evaluation Metrics](#evaluation-metrics)
- [Test Data Requirements](#test-data-requirements)
- [Evaluation Setup](#evaluation-setup)
- [Offline Evaluation](#offline-evaluation)
- [Online Evaluation (A/B Testing)](#online-evaluation-ab-testing)
- [Performance Benchmarking](#performance-benchmarking)
- [Evaluation Code](#evaluation-code)

---

## Overview

To properly evaluate ranking profiles, you need **three types of evaluations**:

1. **Offline Evaluation** - Test with labeled data before deployment
2. **Online Evaluation** - A/B test with real users
3. **Performance Evaluation** - Measure latency and scalability

---

## Evaluation Metrics

### 1. Relevance Metrics (Quality)

#### **Precision@K**
Percentage of relevant results in top-K

```python
def precision_at_k(results, relevant_docs, k):
    """
    Args:
        results: List of retrieved doc IDs in order
        relevant_docs: Set of known relevant doc IDs
        k: Number of top results to consider
    
    Returns:
        float: Precision score (0.0 to 1.0)
    """
    top_k = results[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return relevant_in_top_k / k if k > 0 else 0.0

# Example
results = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = {"doc1", "doc3", "doc4"}
print(f"P@3: {precision_at_k(results, relevant, 3)}")  # 2/3 = 0.667
print(f"P@5: {precision_at_k(results, relevant, 5)}")  # 3/5 = 0.600
```

**When to use**: Chatbots, QA systems where top results matter

---

#### **Recall@K**
Percentage of relevant docs found in top-K

```python
def recall_at_k(results, relevant_docs, k):
    """
    Returns:
        float: Recall score (0.0 to 1.0)
    """
    if len(relevant_docs) == 0:
        return 0.0
    
    top_k = results[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return relevant_in_top_k / len(relevant_docs)

# Example
results = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = {"doc1", "doc3", "doc6", "doc7"}  # 4 total relevant
print(f"R@5: {recall_at_k(results, relevant, 5)}")  # 2/4 = 0.500
```

**When to use**: Search systems, discovery tools

---

#### **Mean Reciprocal Rank (MRR)**
Average of 1/position of first relevant result

```python
def mean_reciprocal_rank(queries_results, queries_relevant):
    """
    Args:
        queries_results: Dict[query_id, List[doc_ids]]
        queries_relevant: Dict[query_id, Set[relevant_doc_ids]]
    
    Returns:
        float: MRR score
    """
    reciprocal_ranks = []
    
    for query_id, results in queries_results.items():
        relevant = queries_relevant[query_id]
        
        for position, doc_id in enumerate(results, start=1):
            if doc_id in relevant:
                reciprocal_ranks.append(1.0 / position)
                break
        else:
            reciprocal_ranks.append(0.0)  # No relevant found
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# Example
queries_results = {
    "q1": ["doc1", "doc2", "doc3"],  # First relevant at position 1
    "q2": ["doc4", "doc5", "doc6"],  # First relevant at position 2
    "q3": ["doc7", "doc8", "doc9"],  # No relevant found
}
queries_relevant = {
    "q1": {"doc1"},
    "q2": {"doc5"},
    "q3": {"doc10"},
}
print(f"MRR: {mean_reciprocal_rank(queries_results, queries_relevant)}")
# (1/1 + 1/2 + 0) / 3 = 0.50
```

**When to use**: QA, known-item search

---

#### **Normalized Discounted Cumulative Gain (NDCG@K)**
Position-aware metric with graded relevance

```python
import math

def dcg_at_k(relevances, k):
    """
    Args:
        relevances: List of relevance scores in retrieval order
        k: Cut-off position
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k(results, relevance_scores, k):
    """
    Args:
        results: List of retrieved doc IDs
        relevance_scores: Dict[doc_id, score] (0=irrelevant, 1=relevant, 2=highly relevant)
        k: Cut-off position
    
    Returns:
        float: NDCG score (0.0 to 1.0)
    """
    # Get relevance scores for retrieved docs
    retrieved_scores = [relevance_scores.get(doc_id, 0) for doc_id in results[:k]]
    
    # Calculate DCG
    dcg = dcg_at_k(retrieved_scores, k)
    
    # Calculate ideal DCG (best possible ordering)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    
    return dcg / idcg if idcg > 0 else 0.0

# Example
results = ["doc1", "doc2", "doc3", "doc4"]
relevance = {
    "doc1": 2,  # Highly relevant
    "doc2": 1,  # Relevant
    "doc3": 0,  # Not relevant
    "doc4": 1,  # Relevant
}
print(f"NDCG@3: {ndcg_at_k(results, relevance, 3)}")
```

**When to use**: Research systems, graded judgments available

---

### 2. Diversity Metrics

#### **Intra-List Diversity (ILD)**
Measures how different results are from each other

```python
def intra_list_diversity(results, embeddings):
    """
    Args:
        results: List of doc IDs
        embeddings: Dict[doc_id, np.array] - document embeddings
    
    Returns:
        float: Average pairwise distance (higher = more diverse)
    """
    from scipy.spatial.distance import cosine
    
    if len(results) < 2:
        return 0.0
    
    distances = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            doc1, doc2 = results[i], results[j]
            if doc1 in embeddings and doc2 in embeddings:
                dist = cosine(embeddings[doc1], embeddings[doc2])
                distances.append(dist)
    
    return sum(distances) / len(distances) if distances else 0.0
```

**When to use**: Evaluating `layered_with_diversity` profile

---

### 3. Coverage Metrics

#### **Topic Coverage**
How many topics/aspects are covered

```python
def topic_coverage(results, doc_topics, all_topics):
    """
    Args:
        results: List of doc IDs
        doc_topics: Dict[doc_id, Set[topic_ids]]
        all_topics: Set of all possible topics
    
    Returns:
        float: Percentage of topics covered
    """
    covered_topics = set()
    for doc_id in results:
        if doc_id in doc_topics:
            covered_topics.update(doc_topics[doc_id])
    
    return len(covered_topics) / len(all_topics) if all_topics else 0.0
```

---

### 4. Performance Metrics

```python
import time
import statistics

def measure_latency(query_func, query, n_runs=10):
    """Measure query latency"""
    latencies = []
    
    for _ in range(n_runs):
        start = time.time()
        query_func(query)
        latencies.append((time.time() - start) * 1000)  # ms
    
    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * len(latencies))],
        "p99_ms": sorted(latencies)[int(0.99 * len(latencies))],
    }
```

---

## Test Data Requirements

### 1. Query Test Set

Create diverse queries covering:

```python
TEST_QUERIES = {
    # Factual questions (high precision needed)
    "factual": [
        "What is the compression ratio of ColBERTv2?",
        "How does late interaction work in ColBERT?",
        "What is the MRR@10 score on MS MARCO?",
    ],
    
    # Conceptual questions (need comprehensive answers)
    "conceptual": [
        "Why is ColBERT effective for retrieval?",
        "How does ColBERT compare to dense retrieval?",
        "What are the advantages of late interaction?",
    ],
    
    # Broad exploratory (diversity important)
    "exploratory": [
        "neural retrieval methods",
        "document ranking approaches",
        "efficient search techniques",
    ],
    
    # Long-form (need multiple chunks)
    "long_form": [
        "Explain the ColBERT architecture including the training process, "
        "compression techniques, and evaluation results",
    ],
    
    # Specific technical (precision critical)
    "technical": [
        "HNSW algorithm complexity",
        "token-level embedding clustering",
        "residual compression implementation",
    ],
}
```

### 2. Relevance Judgments

For each query, label relevant documents:

```python
RELEVANCE_JUDGMENTS = {
    "What is the compression ratio of ColBERTv2?": {
        "doc_abc123": 2,  # Highly relevant (contains exact answer)
        "doc_def456": 1,  # Relevant (mentions compression)
        "doc_ghi789": 0,  # Not relevant
    },
    # ... more queries
}
```

**How to get judgments:**
1. Manual labeling (you or domain experts)
2. LLM-as-a-judge (GPT-4 evaluates relevance)
3. Crowdsourcing (MTurk, Scale AI)
4. User clicks (implicit feedback)

---

## Evaluation Setup

### Complete Evaluation Script

```python
"""
evaluation.py - Complete ranking profile evaluation
"""

import json
import time
import statistics
from typing import Dict, List, Set, Any
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class EvaluationResult:
    profile_name: str
    precision_at_3: float
    precision_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    latency_p50: float
    latency_p95: float
    total_results: int

class RankingEvaluator:
    def __init__(self, vespa_app, test_queries, relevance_judgments):
        self.vespa_app = vespa_app
        self.test_queries = test_queries
        self.relevance_judgments = relevance_judgments
        self.results = {}
    
    def evaluate_profile(
        self, 
        profile_name: str, 
        ranking_config: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate a single ranking profile"""
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {profile_name}")
        print(f"{'='*60}")
        
        all_precisions_3 = []
        all_precisions_5 = []
        all_recalls_10 = []
        all_reciprocal_ranks = []
        all_ndcgs = []
        latencies = []
        
        for query_type, queries in self.test_queries.items():
            print(f"\n  Query type: {query_type}")
            
            for query in queries:
                # Measure latency
                start_time = time.time()
                
                # Query Vespa
                retriever = VespaStreamingLayeredRetriever(
                    app=self.vespa_app,
                    user="jo-bergum",
                    **ranking_config
                )
                results = retriever._get_relevant_documents(query)
                
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
                
                # Get result IDs
                result_ids = [doc.id for doc in results]
                
                # Get relevance judgments
                relevant_docs = set()
                relevance_scores = {}
                if query in self.relevance_judgments:
                    for doc_id, score in self.relevance_judgments[query].items():
                        if score > 0:
                            relevant_docs.add(doc_id)
                        relevance_scores[doc_id] = score
                
                # Calculate metrics
                if relevant_docs:
                    p3 = self.precision_at_k(result_ids, relevant_docs, 3)
                    p5 = self.precision_at_k(result_ids, relevant_docs, 5)
                    r10 = self.recall_at_k(result_ids, relevant_docs, 10)
                    rr = self.reciprocal_rank(result_ids, relevant_docs)
                    ndcg = self.ndcg_at_k(result_ids, relevance_scores, 10)
                    
                    all_precisions_3.append(p3)
                    all_precisions_5.append(p5)
                    all_recalls_10.append(r10)
                    all_reciprocal_ranks.append(rr)
                    all_ndcgs.append(ndcg)
                    
                    print(f"    {query[:50]}... P@3={p3:.3f} MRR={rr:.3f} Latency={latency_ms:.1f}ms")
        
        # Aggregate results
        result = EvaluationResult(
            profile_name=profile_name,
            precision_at_3=statistics.mean(all_precisions_3),
            precision_at_5=statistics.mean(all_precisions_5),
            recall_at_10=statistics.mean(all_recalls_10),
            mrr=statistics.mean(all_reciprocal_ranks),
            ndcg_at_10=statistics.mean(all_ndcgs),
            latency_p50=statistics.median(latencies),
            latency_p95=sorted(latencies)[int(0.95 * len(latencies))],
            total_results=len(latencies)
        )
        
        self.results[profile_name] = result
        return result
    
    def precision_at_k(self, results, relevant, k):
        top_k = results[:k]
        return sum(1 for doc in top_k if doc in relevant) / k if k > 0 else 0.0
    
    def recall_at_k(self, results, relevant, k):
        if not relevant:
            return 0.0
        top_k = results[:k]
        return sum(1 for doc in top_k if doc in relevant) / len(relevant)
    
    def reciprocal_rank(self, results, relevant):
        for i, doc_id in enumerate(results, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0
    
    def ndcg_at_k(self, results, relevance_scores, k):
        import math
        
        def dcg(scores):
            return sum((2**score - 1) / math.log2(i + 2) for i, score in enumerate(scores))
        
        retrieved_scores = [relevance_scores.get(doc_id, 0) for doc_id in results[:k]]
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        
        dcg_score = dcg(retrieved_scores)
        idcg_score = dcg(ideal_scores)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0
    
    def compare_profiles(self):
        """Generate comparison report"""
        
        print("\n" + "="*80)
        print("RANKING PROFILE COMPARISON")
        print("="*80)
        
        # Create DataFrame
        data = []
        for profile_name, result in self.results.items():
            data.append({
                "Profile": profile_name,
                "P@3": f"{result.precision_at_3:.3f}",
                "P@5": f"{result.precision_at_5:.3f}",
                "R@10": f"{result.recall_at_10:.3f}",
                "MRR": f"{result.mrr:.3f}",
                "NDCG@10": f"{result.ndcg_at_10:.3f}",
                "Latency(p50)": f"{result.latency_p50:.1f}ms",
                "Latency(p95)": f"{result.latency_p95:.1f}ms",
            })
        
        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))
        
        # Find best performer for each metric
        print("\n" + "="*80)
        print("BEST PERFORMERS")
        print("="*80)
        
        metrics = ["precision_at_3", "precision_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
        for metric in metrics:
            best = max(self.results.items(), key=lambda x: getattr(x[1], metric))
            print(f"  {metric:20s}: {best[0]} ({getattr(best[1], metric):.3f})")
        
        # Find fastest
        fastest = min(self.results.items(), key=lambda x: x[1].latency_p50)
        print(f"  {'latency_p50':20s}: {fastest[0]} ({fastest[1].latency_p50:.1f}ms)")
        
        return df
    
    def plot_results(self, output_file="ranking_comparison.png"):
        """Create visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        profiles = list(self.results.keys())
        
        # Plot 1: Quality Metrics
        ax = axes[0, 0]
        metrics = {
            "P@3": [self.results[p].precision_at_3 for p in profiles],
            "MRR": [self.results[p].mrr for p in profiles],
            "NDCG@10": [self.results[p].ndcg_at_10 for p in profiles],
        }
        x = range(len(profiles))
        width = 0.25
        for i, (metric, values) in enumerate(metrics.items()):
            ax.bar([xi + i*width for xi in x], values, width, label=metric)
        ax.set_xlabel("Profile")
        ax.set_ylabel("Score")
        ax.set_title("Quality Metrics Comparison")
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels(profiles, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Latency
        ax = axes[0, 1]
        latencies_p50 = [self.results[p].latency_p50 for p in profiles]
        latencies_p95 = [self.results[p].latency_p95 for p in profiles]
        x = range(len(profiles))
        width = 0.35
        ax.bar([xi - width/2 for xi in x], latencies_p50, width, label='p50', color='skyblue')
        ax.bar([xi + width/2 for xi in x], latencies_p95, width, label='p95', color='orange')
        ax.set_xlabel("Profile")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 3: Quality vs Speed Trade-off
        ax = axes[1, 0]
        ndcgs = [self.results[p].ndcg_at_10 for p in profiles]
        lats = [self.results[p].latency_p50 for p in profiles]
        ax.scatter(lats, ndcgs, s=100)
        for i, profile in enumerate(profiles):
            ax.annotate(profile, (lats[i], ndcgs[i]), fontsize=8)
        ax.set_xlabel("Latency p50 (ms)")
        ax.set_ylabel("NDCG@10")
        ax.set_title("Quality vs Speed Trade-off")
        ax.grid(alpha=0.3)
        
        # Plot 4: Precision-Recall
        ax = axes[1, 1]
        precisions = [self.results[p].precision_at_5 for p in profiles]
        recalls = [self.results[p].recall_at_10 for p in profiles]
        ax.scatter(recalls, precisions, s=100)
        for i, profile in enumerate(profiles):
            ax.annotate(profile, (recalls[i], precisions[i]), fontsize=8)
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("Precision@5")
        ax.set_title("Precision-Recall Trade-off")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
        plt.close()
    
    def save_results(self, output_file="evaluation_results.json"):
        """Save results to JSON"""
        output = {}
        for profile_name, result in self.results.items():
            output[profile_name] = {
                "precision_at_3": result.precision_at_3,
                "precision_at_5": result.precision_at_5,
                "recall_at_10": result.recall_at_10,
                "mrr": result.mrr,
                "ndcg_at_10": result.ndcg_at_10,
                "latency_p50": result.latency_p50,
                "latency_p95": result.latency_p95,
            }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
```

---

## Running the Evaluation

### Usage Example

```python
from vespa.application import Vespa

# Load test data
TEST_QUERIES = {
    "factual": [
        "What is the compression ratio of ColBERTv2?",
        "How does late interaction work?",
    ],
    "conceptual": [
        "Why is ColBERT effective?",
    ],
}

RELEVANCE_JUDGMENTS = {
    "What is the compression ratio of ColBERTv2?": {
        "ee6cdd437411e98cc396ed6f84852eed6205dde3": 2,  # Highly relevant
        "e7f6c35b909c54acf5e01a2325dd5028c701a8fb": 1,  # Relevant
    },
    # ... more judgments
}

# Connect to Vespa
vespa_app = Vespa(url="http://localhost", port=8080)

# Initialize evaluator
evaluator = RankingEvaluator(vespa_app, TEST_QUERIES, RELEVANCE_JUDGMENTS)

# Define profiles to test
PROFILES_TO_TEST = {
    "hybrid": {
        "pages": 10,
        "chunks_per_page": 3,
        "min_chunk_score": 0.0
    },
    "layered_base": {
        "pages": 10,
        "chunks_per_page": 3,
        "min_chunk_score": 0.0
    },
    "layered_with_second_phase": {
        "pages": 10,
        "chunks_per_page": 3,
        "min_chunk_score": 0.5
    },
    "layered_with_maxsim": {
        "pages": 10,
        "chunks_per_page": 3,
        "min_chunk_score": 0.6
    },
    "layered_with_diversity": {
        "pages": 20,
        "chunks_per_page": 5,
        "min_chunk_score": 0.3
    },
}

# Run evaluations
for profile_name, config in PROFILES_TO_TEST.items():
    evaluator.evaluate_profile(profile_name, config)

# Compare and visualize
df = evaluator.compare_profiles()
evaluator.plot_results("ranking_comparison.png")
evaluator.save_results("evaluation_results.json")
```

---

## Online Evaluation (A/B Testing)

### Implementation

```python
import random
from datetime import datetime

class ABTestTracker:
    def __init__(self):
        self.events = []
    
    def assign_variant(self, user_id: str) -> str:
        """Assign user to A or B variant (50/50 split)"""
        # Use hash for consistent assignment
        return "A" if hash(user_id) % 2 == 0 else "B"
    
    def log_query(self, user_id: str, query: str, variant: str, results: List[str]):
        """Log query event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "variant": variant,
            "results": results,
            "event_type": "query"
        }
        self.events.append(event)
    
    def log_click(self, user_id: str, query: str, doc_id: str, position: int):
        """Log click event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "doc_id": doc_id,
            "position": position,
            "event_type": "click"
        }
        self.events.append(event)
    
    def analyze_results(self):
        """Analyze A/B test results"""
        from collections import defaultdict
        
        stats = defaultdict(lambda: {"queries": 0, "clicks": 0, "positions": []})
        
        # Aggregate by variant
        for event in self.events:
            if event["event_type"] == "query":
                stats[event["variant"]]["queries"] += 1
            elif event["event_type"] == "click":
                # Find variant for this user
                # (simplified - in real system, join with query events)
                stats[event["variant"]]["clicks"] += 1
                stats[event["variant"]]["positions"].append(event["position"])
        
        # Calculate metrics
        for variant in ["A", "B"]:
            s = stats[variant]
            ctr = s["clicks"] / s["queries"] if s["queries"] > 0 else 0
            avg_pos = sum(s["positions"]) / len(s["positions"]) if s["positions"] else 0
            
            print(f"\nVariant {variant}:")
            print(f"  Queries: {s['queries']}")
            print(f"  Clicks: {s['clicks']}")
            print(f"  CTR: {ctr:.3f}")
            print(f"  Avg Click Position: {avg_pos:.2f}")

# Usage in endpoint
ab_tracker = ABTestTracker()

@app.get("/search-ab")
async def search_with_ab_test(
    user_id: str,
    query: str
):
    variant = ab_tracker.assign_variant(user_id)
    
    # Use different profile based on variant
    if variant == "A":
        retriever = VespaStreamingLayeredRetriever(
            app=vespa_app,
            user="jo-bergum",
            min_chunk_score=0.5  # Control
        )
    else:
        retriever = VespaStreamingLayeredRetriever(
            app=vespa_app,
            user="jo-bergum",
            min_chunk_score=0.7  # Treatment (higher threshold)
        )
    
    results = retriever._get_relevant_documents(query)
    result_ids = [doc.id for doc in results]
    
    # Log query
    ab_tracker.log_query(user_id, query, variant, result_ids)
    
    return {
        "variant": variant,
        "results": results
    }
```

---

## Quick Start Checklist

### Minimum Viable Evaluation

To get started quickly:

1. **Create 20 test queries** (mix of types)
2. **Label top-5 results** for each query (binary: relevant/not relevant)
3. **Run evaluation script** on all profiles
4. **Compare P@3 and latency** (these two metrics tell you most of what you need)
5. **Pick winner** and deploy

### Comprehensive Evaluation

For production:

1. **Collect 100+ test queries** from real users
2. **Get graded relevance** (0-2 scale)
3. **Run full metrics** (P@K, R@K, MRR, NDCG)
4. **A/B test** top 2 candidates
5. **Monitor online metrics** (CTR, dwell time, user satisfaction)

---

## Tools & Libraries

```bash
# Install evaluation dependencies
pip install pandas matplotlib seaborn scikit-learn scipy
```

---

## Expected Results (Typical)

| Profile | P@3 | MRR | NDCG@10 | Latency(p50) | Best For |
|---------|-----|-----|---------|--------------|----------|
| Hybrid | 0.65 | 0.58 | 0.68 | 2ms | Baseline |
| Layered Base | 0.72 | 0.65 | 0.74 | 3ms | Production |
| + Second Phase | 0.78 | 0.71 | 0.80 | 10ms | Balanced |
| + Diversity | 0.75 | 0.68 | 0.77 | 14ms | Exploration |
| + MaxSim | 0.82 | 0.76 | 0.84 | 12ms | Precision |
| + Normalized | 0.76 | 0.69 | 0.78 | 17ms | Fair Hybrid |
| + Global Phase | 0.85 | 0.80 | 0.88 | 35ms | Top-K Quality |

*Your results will vary based on your corpus and queries!*

---

## Next Steps

1. **Start small**: Test with 10-20 queries first
2. **Iterate**: Use results to tune thresholds and weights
3. **Scale up**: Add more queries as you go
4. **Deploy**: A/B test top 2-3 candidates
5. **Monitor**: Track metrics in production

Good luck with your evaluations! 🎯

