"""
evaluate_rankings.py - Simple evaluation script for ranking profiles

Usage:
    python evaluate_rankings.py
"""

import json
import time
import statistics
from typing import Dict, List, Set
from vespa.application import Vespa
import sys
sys.path.append('.')
from main import VespaStreamingLayeredRetriever

# ============================================================================
# TEST DATA - ADD YOUR OWN QUERIES AND RELEVANCE JUDGMENTS
# ============================================================================

TEST_QUERIES = {
    "factual": [
        "What is the compression ratio of ColBERTv2?",
        "How does late interaction work in ColBERT?",
        "What is ColBERT's MRR@10 score on MS MARCO?",
    ],
    "conceptual": [
        "Why is ColBERT effective for retrieval?",
        "What are the advantages of late interaction?",
        "How does ColBERT handle compression?",
    ],
    "exploratory": [
        "neural retrieval methods overview",
        "dense retrieval architectures",
    ],
}

# Add your relevance judgments here
# Format: query -> {doc_id: relevance_score}
# relevance_score: 0=not relevant, 1=relevant, 2=highly relevant
RELEVANCE_JUDGMENTS = {
    "What is the compression ratio of ColBERTv2?": {
        "ee6cdd437411e98cc396ed6f84852eed6205dde3": 2,  # Page 29 - discusses compression
        "e7f6c35b909c54acf5e01a2325dd5028c701a8fb": 1,  # Page 13 - mentions results
    },
    "Why is ColBERT effective for retrieval?": {
        "ee6cdd437411e98cc396ed6f84852eed6205dde3": 2,
        "e7f6c35b909c54acf5e01a2325dd5028c701a8fb": 2,
        "72bcb9b78cb76d3d71d770baf4808c19ebe4fa47": 1,
    },
    # Add more as you label data...
}

# ============================================================================
# PROFILES TO TEST
# ============================================================================

PROFILES_TO_TEST = {
    "layered_base": {
        "pages": 5,
        "chunks_per_page": 3,
        "min_chunk_score": 0.0,
    },
    "layered_threshold_0.5": {
        "pages": 5,
        "chunks_per_page": 3,
        "min_chunk_score": 0.5,
    },
    "layered_threshold_0.7": {
        "pages": 5,
        "chunks_per_page": 3,
        "min_chunk_score": 0.7,
    },
}

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def precision_at_k(results: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Precision@K"""
    if k == 0:
        return 0.0
    top_k = results[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant) / k

def recall_at_k(results: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Recall@K"""
    if not relevant:
        return 0.0
    top_k = results[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant) / len(relevant)

def reciprocal_rank(results: List[str], relevant: Set[str]) -> float:
    """Calculate Reciprocal Rank"""
    for position, doc_id in enumerate(results, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0

def average_precision(results: List[str], relevant: Set[str]) -> float:
    """Calculate Average Precision"""
    if not relevant:
        return 0.0
    
    precisions = []
    num_hits = 0
    
    for i, doc_id in enumerate(results, start=1):
        if doc_id in relevant:
            num_hits += 1
            precisions.append(num_hits / i)
    
    return sum(precisions) / len(relevant) if relevant else 0.0

def evaluate_profile(
    vespa_app, 
    profile_name: str,
    config: Dict,
    test_queries: Dict[str, List[str]],
    relevance_judgments: Dict[str, Dict[str, int]]
):
    """Evaluate a single ranking profile"""
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {profile_name}")
    print(f"Config: {config}")
    print(f"{'='*70}")
    
    all_p3, all_p5, all_r10, all_rr, all_ap = [], [], [], [], []
    latencies = []
    num_results_per_query = []
    
    query_count = 0
    for query_type, queries in test_queries.items():
        print(f"\n  Query Type: {query_type}")
        
        for query in queries:
            query_count += 1
            
            # Skip if no relevance judgments
            if query not in relevance_judgments:
                print(f"    ⚠️  Skipping (no judgments): {query[:50]}...")
                continue
            
            # Create retriever
            retriever = VespaStreamingLayeredRetriever(
                app=vespa_app,
                user="jo-bergum",
                **config
            )
            
            # Measure latency
            start_time = time.time()
            try:
                results = retriever._get_relevant_documents(query)
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
            
            # Extract result IDs
            result_ids = [doc.id for doc in results]
            num_results_per_query.append(len(result_ids))
            
            # Get relevant docs for this query
            relevant_docs = {
                doc_id for doc_id, score in relevance_judgments[query].items() 
                if score > 0
            }
            
            # Calculate metrics
            if relevant_docs and result_ids:
                p3 = precision_at_k(result_ids, relevant_docs, 3)
                p5 = precision_at_k(result_ids, relevant_docs, 5)
                r10 = recall_at_k(result_ids, relevant_docs, 10)
                rr = reciprocal_rank(result_ids, relevant_docs)
                ap = average_precision(result_ids, relevant_docs)
                
                all_p3.append(p3)
                all_p5.append(p5)
                all_r10.append(r10)
                all_rr.append(rr)
                all_ap.append(ap)
                
                print(f"    ✓ {query[:45]:45s} | P@3={p3:.2f} MRR={rr:.2f} Results={len(result_ids):2d} Lat={latency_ms:5.1f}ms")
            else:
                print(f"    ⚠️  {query[:45]:45s} | No results or relevant docs")
    
    # Aggregate results
    if all_p3:
        results = {
            "profile": profile_name,
            "queries_evaluated": len(all_p3),
            "precision_at_3": statistics.mean(all_p3),
            "precision_at_5": statistics.mean(all_p5),
            "recall_at_10": statistics.mean(all_r10),
            "mrr": statistics.mean(all_rr),
            "map": statistics.mean(all_ap),
            "latency_mean": statistics.mean(latencies),
            "latency_median": statistics.median(latencies),
            "latency_p95": sorted(latencies)[int(0.95 * len(latencies))] if len(latencies) > 1 else latencies[0],
            "avg_results": statistics.mean(num_results_per_query),
        }
    else:
        print("    ❌ No queries were successfully evaluated!")
        results = None
    
    return results

def print_comparison(all_results: List[Dict]):
    """Print comparison table"""
    
    print("\n" + "="*120)
    print("RANKING PROFILE COMPARISON")
    print("="*120)
    
    # Header
    print(f"{'Profile':<30} {'Queries':>8} {'P@3':>6} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'MAP':>6} {'Lat(ms)':>8} {'Results':>8}")
    print("-" * 120)
    
    # Rows
    for result in all_results:
        print(f"{result['profile']:<30} "
              f"{result['queries_evaluated']:>8d} "
              f"{result['precision_at_3']:>6.3f} "
              f"{result['precision_at_5']:>6.3f} "
              f"{result['recall_at_10']:>6.3f} "
              f"{result['mrr']:>6.3f} "
              f"{result['map']:>6.3f} "
              f"{result['latency_median']:>8.1f} "
              f"{result['avg_results']:>8.1f}")
    
    print("="*120)
    
    # Best performers
    print("\n" + "="*120)
    print("BEST PERFORMERS")
    print("="*120)
    
    metrics = [
        ("Precision@3", "precision_at_3"),
        ("Precision@5", "precision_at_5"),
        ("Recall@10", "recall_at_10"),
        ("MRR", "mrr"),
        ("MAP", "map"),
    ]
    
    for label, key in metrics:
        best = max(all_results, key=lambda x: x[key])
        print(f"  {label:15s}: {best['profile']:30s} ({best[key]:.3f})")
    
    # Fastest
    fastest = min(all_results, key=lambda x: x['latency_median'])
    print(f"  {'Latency':15s}: {fastest['profile']:30s} ({fastest['latency_median']:.1f}ms)")
    
    # Quality-Speed Trade-off
    print("\n" + "="*120)
    print("QUALITY-SPEED TRADE-OFF ANALYSIS")
    print("="*120)
    for result in sorted(all_results, key=lambda x: x['latency_median']):
        quality_score = (result['precision_at_3'] + result['mrr']) / 2
        print(f"  {result['profile']:30s} | Quality: {quality_score:.3f} | Latency: {result['latency_median']:6.1f}ms")

def save_results(all_results: List[Dict], filename: str = "evaluation_results.json"):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {filename}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*120)
    print("RANKING PROFILE EVALUATION")
    print("="*120)
    
    # Connect to Vespa
    print("\n1. Connecting to Vespa...")
    try:
        vespa_app = Vespa(url="http://localhost", port=8080)
        print("   ✓ Connected to Vespa")
    except Exception as e:
        print(f"   ❌ Failed to connect to Vespa: {e}")
        print("   Make sure Vespa is running (python main.py)")
        return
    
    # Check test data
    print("\n2. Checking test data...")
    total_queries = sum(len(queries) for queries in TEST_QUERIES.values())
    labeled_queries = len(RELEVANCE_JUDGMENTS)
    print(f"   Total queries: {total_queries}")
    print(f"   Labeled queries: {labeled_queries}")
    
    if labeled_queries == 0:
        print("   ⚠️  WARNING: No relevance judgments found!")
        print("   Add relevance judgments to RELEVANCE_JUDGMENTS dict in this file")
        print("   Continuing anyway for latency measurement...")
    
    # Run evaluations
    print(f"\n3. Evaluating {len(PROFILES_TO_TEST)} profiles...")
    all_results = []
    
    for profile_name, config in PROFILES_TO_TEST.items():
        result = evaluate_profile(
            vespa_app,
            profile_name,
            config,
            TEST_QUERIES,
            RELEVANCE_JUDGMENTS
        )
        if result:
            all_results.append(result)
    
    # Compare results
    if all_results:
        print("\n4. Results:")
        print_comparison(all_results)
        
        # Save results
        save_results(all_results)
        
        print("\n" + "="*120)
        print("EVALUATION COMPLETE ✓")
        print("="*120)
        print("\nNext steps:")
        print("  1. Add more queries and relevance judgments")
        print("  2. Test more ranking profiles")
        print("  3. Run A/B tests on top performers")
        print("  4. Deploy winner to production")
    else:
        print("\n❌ No results to compare. Check errors above.")

if __name__ == "__main__":
    main()

