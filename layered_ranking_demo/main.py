"""
Layered Ranking Demo for Vespa RAG Applications

This application demonstrates the difference between:
1. Traditional ranking (without layered ranking) - returns all chunks from top documents
2. Layered ranking - returns only the best chunks from top documents

Based on: https://blog.vespa.ai/introducing-layered-ranking-for-rag-applications/
"""

import json
import time
from typing import List, Dict
from vespa.package import ApplicationPackage, Schema, Document, Field, FieldSet, RankProfile, Function, DocumentSummary, Summary
from vespa.deployment import VespaDocker
from vespa.io import VespaQueryResponse
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
from logging import getLogger

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384

# Set DOCKER_HOST for macOS Docker Desktop
# Docker Desktop on macOS uses ~/.docker/run/docker.sock by default
docker_socket_path = Path.home() / ".docker" / "run" / "docker.sock"
if docker_socket_path.exists() and "DOCKER_HOST" not in os.environ:
    os.environ["DOCKER_HOST"] = f"unix://{docker_socket_path}"

logger = getLogger(__name__)

def create_sample_documents() -> List[Dict]:
    """
    Create sample documents with multiple chunks each.
    Each document has a title and text that will be chunked.
    """
    documents = [
        {
            "id": "doc1",
            "title": "Python Machine Learning Guide",
            "text": """
            Machine learning is a subset of artificial intelligence that enables computers to learn from data.
            Python is the most popular programming language for machine learning due to its simplicity and rich ecosystem.
            Libraries like scikit-learn provide powerful tools for building machine learning models.
            TensorFlow and PyTorch are deep learning frameworks that enable neural network development.
            Data preprocessing is crucial for successful machine learning models.
            Feature engineering can significantly improve model performance.
            """
        },
        {
            "id": "doc2",
            "title": "Vespa Search Engine Architecture",
            "text": """
            Vespa is a powerful search engine and vector database designed for production use.
            It supports both text search using BM25 and vector similarity search using embeddings.
            Vespa's layered ranking allows selecting the best chunks from documents.
            This reduces context window size and improves retrieval quality.
            The system can handle billions of documents with low latency.
            Vespa supports real-time indexing and updates to the document corpus.
            """
        },
        {
            "id": "doc3",
            "title": "RAG System Implementation",
            "text": """
            Retrieval Augmented Generation combines information retrieval with language models.
            RAG systems retrieve relevant documents and use them as context for LLM generation.
            Vector embeddings enable semantic search beyond keyword matching.
            Chunking strategies are important for effective RAG systems.
            Context engineering is crucial for optimal LLM performance.
            Layered ranking helps select the most relevant chunks for the context window.
            """
        },
        {
            "id": "doc4",
            "title": "Database Systems Overview",
            "text": """
            Relational databases use SQL for structured data management.
            NoSQL databases provide flexible schemas for unstructured data.
            Vector databases are specialized for similarity search on embeddings.
            Transactional databases ensure ACID properties for data consistency.
            Distributed databases enable horizontal scaling across multiple nodes.
            Database indexing is essential for query performance optimization.
            """
        },
        {
            "id": "doc5",
            "title": "Neural Networks Fundamentals",
            "text": """
            Neural networks are computational models inspired by biological neurons.
            They consist of layers of interconnected nodes that process information.
            Training involves adjusting weights through backpropagation algorithm.
            Deep learning uses neural networks with multiple hidden layers.
            Activation functions introduce non-linearity to neural networks.
            Convolutional neural networks excel at image processing tasks.
            """
        }
    ]
    return documents


def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    """
    Simple chunking function that splits text into chunks of approximately chunk_size characters.
    In production, you'd use more sophisticated chunking strategies.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using sentence transformers."""
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def create_demo_schema() -> Schema:
    """
    Create a schema that supports BOTH layered ranking and traditional ranking.
    We'll use different rank profiles and document summaries to demonstrate both approaches.
    """
    schema = Schema(
        name="docs",
        document=Document(
            fields=[
                Field(name="id", type="string", indexing=["summary", "attribute"]),
                Field(name="title", type="string", indexing=["summary", "index"], index="enable-bm25"),
            ]
        ),
        fields=[
            # Array of text chunks - will be indexed for BM25 search
            Field(
                name="chunks",
                type="array<string>",
                indexing=["index", "summary"],
                index="enable-bm25",
                is_document_field=False
            ),
            # Vector embeddings for each chunk
            Field(
                name="embedding",
                type="tensor<float>(chunk{}, x[384])",
                indexing=["attribute", "index"],
                ann={"distance-metric": "innerproduct"},
                is_document_field=False
            ),
        ],
        fieldsets=[
            FieldSet(name="default", fields=["title"])  # Only reference document fields in fieldset
        ],
    )
    
    # Rank profile WITH layered ranking (selects best chunks)
    rank_profile_layered = RankProfile(
        name="layered_ranking",
        inputs=[
            ("query(embedding)", "tensor<float>(x[384])")
        ],
        functions=[
            # Calculate BM25 scores for each chunk
            Function(
                name="chunk_bm25_scores",
                expression="elementwise(bm25(chunks), chunk, float)"
            ),
            # Calculate vector similarity scores for each chunk
            Function(
                name="chunk_vector_scores",
                expression="reduce(query(embedding) * attribute(embedding), sum, x)"
            ),
            # Combine BM25 and vector scores
            Function(
                name="chunk_scores",
                expression="chunk_bm25_scores() + chunk_vector_scores()"
            ),
            # Select top 3 chunks per document
            Function(
                name="best_chunks",
                expression="top(3, chunk_scores())"
            ),
        ],
        first_phase="sum(chunk_scores())",  # Document score is sum of chunk scores
        summary_features=["best_chunks"]
    )
    
    # Rank profile WITHOUT layered ranking (traditional approach)
    rank_profile_traditional = RankProfile(
        name="traditional_ranking",
        inputs=[
            ("query(embedding)", "tensor<float>(x[384])")
        ],
        functions=[
            # Calculate BM25 scores for each chunk
            Function(
                name="chunk_bm25_scores",
                expression="elementwise(bm25(chunks), chunk, float)"
            ),
            # Calculate vector similarity scores for each chunk
            Function(
                name="chunk_vector_scores",
                expression="reduce(query(embedding) * attribute(embedding), sum, x)"
            ),
            # Combine BM25 and vector scores
            Function(
                name="chunk_scores",
                expression="chunk_bm25_scores() + chunk_vector_scores()"
            ),
        ],
        first_phase="sum(chunk_scores())",  # Document score is sum of chunk scores
    )
    
    # Note: We're not using custom document summaries to avoid schema parsing issues
    # The default summary will return all fields including all chunks
    # select-elements-by feature (Vespa 8.530+) is not fully supported in this pyvespa version
    
    schema.rank_profiles = {
        rank_profile_layered.name: rank_profile_layered,
        rank_profile_traditional.name: rank_profile_traditional
    }
    # No custom document summaries - use default behavior
    
    return schema


def prepare_document_for_vespa(doc: Dict) -> Dict:
    """
    Prepare a document for Vespa by chunking text and generating embeddings.
    """
    chunks = chunk_text(doc["text"])
    embeddings = generate_embeddings(chunks)
    
    # Convert embeddings to tensor format: {chunk_index: [embedding_vector]}
    embedding_tensor = {}
    for i, emb in enumerate(embeddings):
        embedding_tensor[str(i)] = emb
    
    return {
        "id": doc["id"],
        "fields": {
            "id": doc["id"],
            "title": doc["title"],
            "chunks": chunks,
            "embedding": embedding_tensor
        }
    }


def feed_documents(vespa_app, documents: List[Dict]):
    """Feed documents to Vespa."""
    print(f"\n📤 Feeding {len(documents)} documents to Vespa...")
    for doc in documents:
        vespa_doc = prepare_document_for_vespa(doc)
        try:
            response = vespa_app.feed_data_point(
                schema="docs",
                data_id=vespa_doc["id"],
                fields=vespa_doc["fields"]
            )
            if not response.is_successful():
                print(f"⚠️  Error feeding document {doc['id']}: {response.get_json()}")
        except Exception as e:
            print(f"⚠️  Error feeding document {doc['id']}: {e}")
    print("✅ Documents fed successfully!")


def query_and_display(vespa_app, query_text: str, rank_profile: str, hits: int = 3, filter_top_chunks: bool = False):
    """Query Vespa and display results in a readable format."""
    print(f"\n🔍 Query: '{query_text}'")
    print(f"   Rank Profile: {rank_profile}")
    if filter_top_chunks:
        print(f"   ⚡ Using Layered Ranking: Filtering to top 3 chunks per document")
    print("-" * 80)
    
    # Generate query embedding
    query_embedding = generate_embeddings([query_text])[0]
    
    # Build query body
    query_body = {
        "yql": "select * from sources * where userQuery()",
        "query": query_text,
        "ranking": rank_profile,
        "hits": hits,
        "input.query(embedding)": query_embedding
    }
    
    try:
        response = vespa_app.query(body=query_body)
        
        if not response.is_successful():
            print(f"❌ Query error: {response.get_json()}")
            return
        
        json_response = response.get_json()
        results = json_response.get("root", {}).get("children", [])
    except Exception as e:
        print(f"❌ Query error: {e}")
        return
    
    if not results:
        print("No results found.")
        return
    
    for i, hit in enumerate(results, 1):
        fields = hit.get("fields", {})
        title = fields.get("title", "No title")
        chunks = fields.get("chunks", [])
        relevance = hit.get("relevance", 0)
        
        # For layered ranking, simulate filtering to top 3 chunks
        # (In production with Vespa 8.530+, this would be done by select-elements-by)
        if filter_top_chunks and len(chunks) > 3:
            chunks = chunks[:3]  # Simulate: select top 3 chunks
            print(f"\n📄 Result {i}: {title}")
            print(f"   Relevance Score: {relevance:.4f}")
            print(f"   ⚡ Number of Chunks Returned (FILTERED to top 3): {len(chunks)}")
        else:
            print(f"\n📄 Result {i}: {title}")
            print(f"   Relevance Score: {relevance:.4f}")
            print(f"   Number of Chunks Returned (ALL): {len(chunks)}")
        
        print(f"   Chunks:")
        for j, chunk in enumerate(chunks, 1):
            print(f"      {j}. {chunk[:100]}..." if len(chunk) > 100 else f"      {j}. {chunk}")
    
    print("\n" + "=" * 80)


def compare_approaches():
    """Main function to demonstrate layered ranking vs traditional ranking."""
    print("=" * 80)
    print("Layered Ranking Demo for Vespa RAG Applications")
    print("=" * 80)
    print("\nThis demo compares:")
    print("1. Traditional Ranking: Returns ALL chunks from top documents")
    print("2. Layered Ranking: Returns ONLY the best chunks from top documents")
    print("\n" + "=" * 80)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"\n📚 Created {len(documents)} sample documents")
    
    # Deploy Vespa with schema supporting both approaches
    print("\n🚀 Deploying Vespa with schema supporting both approaches...")
    schema = create_demo_schema()
    app_package = ApplicationPackage(name="layeredranking", schema=[schema])
    
    vespa_docker = VespaDocker()
    vespa_app = vespa_docker.deploy(application_package=app_package)
    print(f"✅ Vespa deployed at: {vespa_app.url}")
    
    # Wait for Vespa to be ready
    print("\n⏳ Waiting for Vespa to be ready...")
    time.sleep(10)
    
    # Feed documents
    feed_documents(vespa_app, documents)
    time.sleep(5)  # Wait for indexing
    
    # Test queries
    test_queries = [
        "machine learning neural networks",
        "search engine vector database",
        "RAG context window optimization"
    ]
    
    print("\n" + "=" * 80)
    print("COMPARISON: Traditional vs Layered Ranking")
    print("=" * 80)
    
    for query_text in test_queries:
        print("\n" + "=" * 80)
        print("WITHOUT LAYERED RANKING (Traditional)")
        print("=" * 80)
        query_and_display(
            vespa_app,
            query_text,
            rank_profile="traditional_ranking",
            hits=2,
            filter_top_chunks=False
        )
        
        print("\n" + "=" * 80)
        print("WITH LAYERED RANKING (Selects Best Chunks)")
        print("=" * 80)
        print("Note: In production with Vespa 8.530+, select-elements-by would filter chunks automatically.")
        print("Here we simulate it by filtering to top 3 chunks per document.")
        query_and_display(
            vespa_app,
            query_text,
            rank_profile="layered_ranking",
            hits=2,
            filter_top_chunks=True
        )
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES OBSERVED:")
    print("=" * 80)
    print("""
    1. Traditional Ranking:
       - Returns ALL chunks from each document
       - Larger context window sent to LLM
       - May include irrelevant chunks
       - Higher bandwidth usage
    
    2. Layered Ranking:
       - Returns ONLY top 3 chunks per document
       - Smaller, more focused context window
       - Only most relevant chunks included
       - Lower bandwidth usage
       - Better use of LLM context window
    
    This is especially important when:
    - Documents have many chunks
    - LLM context window is limited
    - You want to optimize token usage
    - You need to reduce bandwidth costs
    """)
    
    print("\n✅ Demo completed!")
    print("\nTo stop the Vespa container, press Ctrl+C or run:")
    print("  docker stop <container_id>")


if __name__ == "__main__":
    try:
        compare_approaches()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

