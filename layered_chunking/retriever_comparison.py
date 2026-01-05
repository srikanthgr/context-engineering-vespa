"""
Comparison of Hybrid + Python Filtering vs Layered Ranking

This module implements both approaches as LangChain retrievers for direct comparison.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from vespa.application import Vespa
from vespa.io import VespaQueryResponse


class VespaStreamingHybridRetriever(BaseRetriever):
    """
    Hybrid ranking with Python-side chunk filtering.
    
    This approach:
    1. Uses Vespa's hybrid rank profile (semantic similarity + native rank)
    2. Returns ALL chunks from matched documents
    3. Filters chunks in Python based on similarity threshold
    4. Selects top K chunks per document
    
    Pros:
    - Easy to customize filtering logic
    - No schema changes needed
    - Quick to iterate
    
    Cons:
    - Only uses semantic similarity for chunk selection
    - More network transfer (all chunks)
    - Python processing overhead
    """
    
    app: Vespa
    user: str
    pages: int = 5
    chunks_per_page: int = 3
    chunk_similarity_threshold: float = 0.8

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        response: VespaQueryResponse = self.app.query(
            yql="select id, url, title, page, authors, chunks from pdf where userQuery() or ({targetHits:20}nearestNeighbor(embedding,q))",
            groupname=self.user,
            ranking="hybrid",  # Uses hybrid rank profile
            query=query,
            hits=self.pages,
            body={
                "presentation.format.tensors": "short-value",
                "input.query(q)": f'embed(e5, "query: {query} ")',
            },
            timeout="2s",
        )
        
        if not response.is_successful():
            raise ValueError(
                f"Query failed with status code {response.status_code}, "
                f"url={response.url} response={response.json}"
            )
        
        return self._parse_response(response)

    def _parse_response(self, response: VespaQueryResponse) -> List[Document]:
        documents: List[Document] = []
        
        for hit in response.hits:
            fields = hit["fields"]
            
            # Get chunks with their similarity scores
            chunks_with_scores = self._get_chunk_similarities(fields)
            
            # Filter by threshold and select top K
            # NOTE: Only uses semantic similarity, not lexical match
            best_chunks_on_page = " ### ".join(
                [
                    chunk
                    for chunk, score in chunks_with_scores[0 : self.chunks_per_page]
                    if score > self.chunk_similarity_threshold
                ]
            )
            
            documents.append(
                Document(
                    id=fields["id"],
                    page_content=best_chunks_on_page,
                    metadata={
                        "title": fields["title"],
                        "url": fields["url"],
                        "page": fields["page"],
                        "authors": fields["authors"],
                        "features": fields["matchfeatures"],
                        "retriever": "hybrid_python",
                    },
                )
            )
        
        return documents

    def _get_chunk_similarities(self, hit_fields: dict) -> List[tuple]:
        """Extract and sort chunks by similarity scores."""
        match_features = hit_fields["matchfeatures"]
        similarities = match_features["similarities"]
        
        # Convert similarity dict to list of scores
        chunk_scores = []
        for i in range(0, len(similarities)):
            chunk_scores.append(similarities.get(str(i), 0))
        
        chunks = hit_fields["chunks"]
        chunks_with_scores = list(zip(chunks, chunk_scores))
        
        # Sort by similarity score (descending)
        return sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)


class VespaStreamingLayeredRetriever(BaseRetriever):
    """
    Layered ranking with Vespa-side chunk filtering.
    
    This approach:
    1. Uses Vespa's layered ranking profile
    2. Computes chunk scores as: distance_score + bm25_score
    3. Filters chunks in Vespa (must have BOTH semantic + lexical match)
    4. Returns best K chunks per document (via best_chunks function)
    
    Pros:
    - Dual-criteria filtering (semantic + lexical)
    - More precise chunk selection
    - Better performance at scale
    - Filtering happens in Vespa (optimized C++)
    
    Cons:
    - Less flexible (requires schema changes)
    - More complex rank profile
    - Harder to debug
    """
    
    app: Vespa
    user: str
    pages: int = 5
    chunks_per_page: int = 3  # Controlled by top(N) in rank profile

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        response: VespaQueryResponse = self.app.query(
            yql="select id, url, title, page, authors, chunks from pdf where userQuery() or ({targetHits:20}nearestNeighbor(embedding,q))",
            groupname=self.user,
            ranking="layeredranking",  # Uses layered rank profile
            query=query,
            hits=self.pages,
            body={
                "presentation.format.tensors": "short-value",
                "input.query(q)": f'embed(e5, "query: {query} ")',
            },
            timeout="2s",
        )
        
        if not response.is_successful():
            raise ValueError(
                f"Query failed with status code {response.status_code}, "
                f"url={response.url} response={response.json}"
            )
        
        return self._parse_response(response)

    def _parse_response(self, response: VespaQueryResponse) -> List[Document]:
        documents: List[Document] = []
        
        for hit in response.hits:
            fields = hit["fields"]
            
            # Extract best chunks identified by Vespa
            best_chunks = self._get_best_chunks(fields)
            
            # NOTE: Due to pyvespa limitation, all chunks are still returned
            # In native Vespa with select-elements-by, only best chunks would be returned
            best_chunks_content = " ### ".join(best_chunks)
            
            documents.append(
                Document(
                    id=fields["id"],
                    page_content=best_chunks_content,
                    metadata={
                        "title": fields["title"],
                        "url": fields["url"],
                        "page": fields["page"],
                        "authors": fields["authors"],
                        "features": fields["matchfeatures"],
                        "retriever": "layered_ranking",
                    },
                )
            )
        
        return documents

    def _get_best_chunks(self, hit_fields: dict) -> List[str]:
        """Extract chunks identified as 'best' by Vespa's layered ranking."""
        match_features = hit_fields["matchfeatures"]
        best_chunks_indices = match_features.get("best_chunks", {})
        
        chunks = hit_fields["chunks"]
        
        # Sort by score (descending) to get top chunks in order
        sorted_indices = sorted(
            best_chunks_indices.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Extract the actual chunk content
        best_chunks = [chunks[int(idx)] for idx, score in sorted_indices]
        
        return best_chunks


class HybridLayeredRetriever(BaseRetriever):
    """
    Combination approach: Use layered ranking with additional Python logic.
    
    This approach:
    1. Uses Vespa's layered ranking for primary filtering
    2. Applies additional custom Python logic on selected chunks
    3. Best of both worlds: Vespa's dual-criteria + Python flexibility
    
    Pros:
    - Efficient primary filtering in Vespa
    - Flexibility for custom logic in Python
    - Can add domain-specific rules
    
    Use cases:
    - Need dual-criteria base filtering
    - Plus custom re-ranking or formatting
    - Domain-specific chunk selection logic
    """
    
    app: Vespa
    user: str
    pages: int = 5
    min_chunk_length: int = 100  # Custom Python rule
    boost_citation_chunks: bool = True  # Custom Python rule

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Step 1: Use layered ranking in Vespa
        response: VespaQueryResponse = self.app.query(
            yql="select id, url, title, page, authors, chunks from pdf where userQuery() or ({targetHits:20}nearestNeighbor(embedding,q))",
            groupname=self.user,
            ranking="layeredranking",
            query=query,
            hits=self.pages,
            body={
                "presentation.format.tensors": "short-value",
                "input.query(q)": f'embed(e5, "query: {query} ")',
            },
            timeout="2s",
        )
        
        if not response.is_successful():
            raise ValueError(
                f"Query failed with status code {response.status_code}"
            )
        
        return self._parse_response(response, query)

    def _parse_response(self, response: VespaQueryResponse, query: str) -> List[Document]:
        documents: List[Document] = []
        
        for hit in response.hits:
            fields = hit["fields"]
            
            # Step 2: Get Vespa-selected best chunks
            vespa_best_chunks = self._get_vespa_best_chunks(fields)
            
            # Step 3: Apply custom Python logic
            final_chunks = self._apply_custom_logic(
                vespa_best_chunks,
                query,
                fields
            )
            
            best_chunks_content = " ### ".join(final_chunks)
            
            documents.append(
                Document(
                    id=fields["id"],
                    page_content=best_chunks_content,
                    metadata={
                        "title": fields["title"],
                        "url": fields["url"],
                        "page": fields["page"],
                        "authors": fields["authors"],
                        "features": fields["matchfeatures"],
                        "retriever": "hybrid_layered",
                    },
                )
            )
        
        return documents

    def _get_vespa_best_chunks(self, hit_fields: dict) -> List[tuple]:
        """Get chunks selected by Vespa with their scores."""
        match_features = hit_fields["matchfeatures"]
        best_chunks_indices = match_features.get("best_chunks", {})
        chunks = hit_fields["chunks"]
        
        chunks_with_scores = [
            (chunks[int(idx)], score)
            for idx, score in best_chunks_indices.items()
        ]
        
        return sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)

    def _apply_custom_logic(
        self,
        chunks_with_scores: List[tuple],
        query: str,
        fields: dict
    ) -> List[str]:
        """Apply custom Python filtering/boosting logic."""
        processed_chunks = []
        
        for chunk, score in chunks_with_scores:
            # Custom rule 1: Filter by minimum length
            if len(chunk) < self.min_chunk_length:
                continue
            
            # Custom rule 2: Boost chunks with citations
            if self.boost_citation_chunks and self._has_citations(chunk):
                # Move citation chunks to the front
                processed_chunks.insert(0, chunk)
            else:
                processed_chunks.append(chunk)
        
        return processed_chunks

    def _has_citations(self, chunk: str) -> bool:
        """Check if chunk contains citations."""
        citation_markers = ["et al.", ")", "]", "al.,"]
        return any(marker in chunk for marker in citation_markers)


# Example usage
if __name__ == "__main__":
    from vespa.application import Vespa
    
    # Connect to Vespa
    vespa_app = Vespa(url="http://localhost", port=8080)
    
    # Initialize retrievers
    hybrid_retriever = VespaStreamingHybridRetriever(
        app=vespa_app,
        user="jo-bergum",
        chunks_per_page=3,
        chunk_similarity_threshold=0.8
    )
    
    layered_retriever = VespaStreamingLayeredRetriever(
        app=vespa_app,
        user="jo-bergum",
        pages=5
    )
    
    combined_retriever = HybridLayeredRetriever(
        app=vespa_app,
        user="jo-bergum",
        min_chunk_length=100,
        boost_citation_chunks=True
    )
    
    # Test query
    query = "why is colbert effective for retrieval?"
    
    print("=" * 80)
    print("HYBRID + PYTHON FILTERING")
    print("=" * 80)
    hybrid_docs = hybrid_retriever.get_relevant_documents(query)
    for i, doc in enumerate(hybrid_docs[:2]):
        print(f"\nDocument {i+1}:")
        print(f"Title: {doc.metadata['title']}")
        print(f"Chunks: {doc.page_content[:200]}...")
        print(f"Retriever: {doc.metadata['retriever']}")
    
    print("\n" + "=" * 80)
    print("LAYERED RANKING")
    print("=" * 80)
    layered_docs = layered_retriever.get_relevant_documents(query)
    for i, doc in enumerate(layered_docs[:2]):
        print(f"\nDocument {i+1}:")
        print(f"Title: {doc.metadata['title']}")
        print(f"Chunks: {doc.page_content[:200]}...")
        print(f"Retriever: {doc.metadata['retriever']}")
    
    print("\n" + "=" * 80)
    print("COMBINED APPROACH")
    print("=" * 80)
    combined_docs = combined_retriever.get_relevant_documents(query)
    for i, doc in enumerate(combined_docs[:2]):
        print(f"\nDocument {i+1}:")
        print(f"Title: {doc.metadata['title']}")
        print(f"Chunks: {doc.page_content[:200]}...")
        print(f"Retriever: {doc.metadata['retriever']}")

