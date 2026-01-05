"""
VespaStreamingLayeredRetriever - LangChain Retriever using Layered Ranking

This retriever uses Vespa's layered ranking feature to:
1. Score chunks using both semantic similarity (embeddings) and lexical relevance (BM25)
2. Filter chunks that must match BOTH criteria (via join operation)
3. Select top K best chunks per document
4. Return only the most relevant chunks for LLM context

Compared to VespaStreamingHybridRetriever:
- Hybrid: Uses only semantic similarity, filters in Python
- Layered: Uses semantic + lexical, filters in Vespa, better precision
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
import logging

logger = logging.getLogger(__name__)


class VespaStreamingLayeredRetriever(BaseRetriever):
    """
    LangChain retriever using Vespa's layered ranking.
    
    Features:
    - Dual-criteria chunk filtering (semantic + lexical)
    - Automatic best-chunk selection in Vespa
    - No manual threshold needed (join operation provides filtering)
    - Higher precision than similarity-only filtering
    
    Attributes:
        app: Vespa application instance
        user: Streaming mode groupname (required for streaming mode)
        pages: Maximum number of documents to retrieve
        chunks_per_page: Number of top chunks to extract per document
        include_metadata: Whether to include match features in metadata
        
    Example:
        >>> from vespa.application import Vespa
        >>> vespa_app = Vespa(url="http://localhost", port=8080)
        >>> retriever = VespaStreamingLayeredRetriever(
        ...     app=vespa_app,
        ...     user="jo-bergum",
        ...     pages=5,
        ...     chunks_per_page=3
        ... )
        >>> docs = retriever.get_relevant_documents("why is colbert effective?")
    """
    
    app: Vespa
    user: str
    pages: int = 5
    chunks_per_page: int = 3
    include_metadata: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using layered ranking.
        
        Args:
            query: Search query text
            
        Returns:
            List of Document objects with selected chunks
            
        Raises:
            ValueError: If query fails
        """
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
            timeout="5s",
        )
        
        if not response.is_successful():
            error_msg = (
                f"Query failed with status code {response.status_code}, "
                f"url={response.url}, response={response.json}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Retrieved {len(response.hits)} documents for query: {query}")
        
        return self._parse_response(response)

    def _parse_response(self, response: VespaQueryResponse) -> List[Document]:
        """
        Parse Vespa response and extract best chunks.
        
        Args:
            response: Vespa query response
            
        Returns:
            List of Document objects with filtered chunks
        """
        documents: List[Document] = []
        
        for hit in response.hits:
            fields = hit["fields"]
            
            # Extract best chunks identified by Vespa's layered ranking
            best_chunks = self._extract_best_chunks(fields)
            
            if not best_chunks:
                logger.warning(
                    f"No best chunks found for document {fields.get('id', 'unknown')}"
                )
                continue
            
            # Join chunks with separator for LLM context
            page_content = " ### ".join(best_chunks)
            
            # Build metadata
            metadata = self._build_metadata(fields, len(best_chunks))
            
            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )
        
        logger.info(f"Parsed {len(documents)} documents with best chunks")
        
        return documents

    def _extract_best_chunks(self, fields: Dict[str, Any]) -> List[str]:
        """
        Extract chunks identified as 'best' by Vespa's layered ranking.
        
        This implements the workaround for pyvespa's lack of select-elements-by support.
        In native Vespa with select-elements-by, only best chunks would be returned.
        Here we manually filter using the best_chunks indices from matchfeatures.
        
        Args:
            fields: Document fields from Vespa response
            
        Returns:
            List of best chunk texts, sorted by score (highest first)
        """
        match_features = fields.get("matchfeatures", {})
        best_chunks_dict = match_features.get("best_chunks", {})
        
        if not best_chunks_dict:
            # Fallback: If no best_chunks in matchfeatures, return all chunks
            # This shouldn't happen if rank profile is configured correctly
            logger.warning(
                "No 'best_chunks' found in matchfeatures, returning all chunks"
            )
            return fields.get("chunks", [])
        
        all_chunks = fields.get("chunks", [])
        
        if not all_chunks:
            logger.warning("No chunks field in document")
            return []
        
        # Build list of (chunk_content, score, original_index) tuples
        selected_chunks = []
        for idx_str, score in best_chunks_dict.items():
            idx = int(idx_str)
            if 0 <= idx < len(all_chunks):
                selected_chunks.append((all_chunks[idx], score, idx))
            else:
                logger.warning(
                    f"Invalid chunk index {idx} (total chunks: {len(all_chunks)})"
                )
        
        # Sort by score descending (highest scores first)
        selected_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to chunks_per_page and return just the text
        top_chunks = selected_chunks[:self.chunks_per_page]
        
        logger.debug(
            f"Selected {len(top_chunks)} chunks from {len(all_chunks)} total chunks"
        )
        
        return [chunk for chunk, score, idx in top_chunks]

    def _build_metadata(
        self, 
        fields: Dict[str, Any], 
        num_chunks_selected: int
    ) -> Dict[str, Any]:
        """
        Build metadata dictionary for Document.
        
        Args:
            fields: Document fields from Vespa response
            num_chunks_selected: Number of chunks selected as best
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "id": fields.get("id", "unknown"),
            "title": fields.get("title", ""),
            "url": fields.get("url", ""),
            "page": fields.get("page", 0),
            "authors": fields.get("authors", []),
            "retriever": "layered_ranking",
            "chunks_selected": num_chunks_selected,
            "total_chunks": len(fields.get("chunks", [])),
        }
        
        # Optionally include match features for analysis
        if self.include_metadata:
            match_features = fields.get("matchfeatures", {})
            metadata["features"] = {
                "best_chunks": match_features.get("best_chunks", {}),
                "chunk_scores": match_features.get("chunk_scores", {}),
                "distance_scores": match_features.get("my_distance_scores", {}),
                "text_scores": match_features.get("my_text_scores", {}),
            }
        
        return metadata

    def get_chunk_details(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Get detailed scoring information for chunks in a document.
        
        Useful for debugging and understanding why certain chunks were selected.
        
        Args:
            document: Document returned by this retriever
            
        Returns:
            Dictionary with chunk scoring details, or None if not available
            
        Example:
            >>> docs = retriever.get_relevant_documents("colbert effectiveness")
            >>> details = retriever.get_chunk_details(docs[0])
            >>> print(details['chunks'][0])
            {
                'text': 'ColBERT achieves...',
                'chunk_score': 0.920,
                'distance_score': 0.192,
                'text_score': 0.728,
                'index': 3
            }
        """
        if not self.include_metadata or "features" not in document.metadata:
            return None
        
        features = document.metadata["features"]
        best_chunks = features.get("best_chunks", {})
        chunk_scores = features.get("chunk_scores", {})
        distance_scores = features.get("distance_scores", {})
        text_scores = features.get("text_scores", {})
        
        # Extract chunk texts from page_content
        chunk_texts = document.page_content.split(" ### ")
        
        # Build detailed information
        chunk_details = []
        for i, (idx_str, score) in enumerate(best_chunks.items()):
            idx = int(idx_str)
            chunk_info = {
                "text": chunk_texts[i] if i < len(chunk_texts) else "",
                "chunk_score": score,
                "distance_score": distance_scores.get(idx_str, 0),
                "text_score": text_scores.get(idx_str, 0),
                "index": idx,
                "rank": i + 1,
            }
            chunk_details.append(chunk_info)
        
        return {
            "document_id": document.metadata["id"],
            "title": document.metadata["title"],
            "chunks": chunk_details,
            "total_chunks": document.metadata["total_chunks"],
            "selected_chunks": document.metadata["chunks_selected"],
        }


class VespaStreamingLayeredRetrieverWithFallback(VespaStreamingLayeredRetriever):
    """
    Enhanced version with fallback to hybrid ranking if layered ranking fails.
    
    Useful during development or when layered rank profile might not be available.
    """
    
    fallback_ranking: str = "hybrid"
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Try layered ranking first, fall back to hybrid if it fails."""
        try:
            return super()._get_relevant_documents(query)
        except Exception as e:
            logger.warning(
                f"Layered ranking failed: {e}. Falling back to {self.fallback_ranking}"
            )
            return self._get_with_fallback(query)
    
    def _get_with_fallback(self, query: str) -> List[Document]:
        """Fallback to hybrid ranking."""
        response: VespaQueryResponse = self.app.query(
            yql="select id, url, title, page, authors, chunks from pdf where userQuery() or ({targetHits:20}nearestNeighbor(embedding,q))",
            groupname=self.user,
            ranking=self.fallback_ranking,
            query=query,
            hits=self.pages,
            body={
                "presentation.format.tensors": "short-value",
                "input.query(q)": f'embed(e5, "query: {query} ")',
            },
            timeout="5s",
        )
        
        if not response.is_successful():
            raise ValueError(f"Fallback query also failed: {response.status_code}")
        
        return self._parse_hybrid_response(response)
    
    def _parse_hybrid_response(self, response: VespaQueryResponse) -> List[Document]:
        """Parse hybrid ranking response (fallback behavior)."""
        documents = []
        
        for hit in response.hits:
            fields = hit["fields"]
            
            # For hybrid, use similarity scores to select chunks
            chunks_with_scores = self._get_chunk_similarities(fields)
            best_chunks = [
                chunk for chunk, score in chunks_with_scores[:self.chunks_per_page]
            ]
            
            page_content = " ### ".join(best_chunks)
            metadata = self._build_metadata(fields, len(best_chunks))
            metadata["retriever"] = "hybrid_fallback"
            
            documents.append(Document(page_content=page_content, metadata=metadata))
        
        return documents
    
    def _get_chunk_similarities(self, fields: Dict[str, Any]) -> List[tuple]:
        """Get chunks sorted by similarity (for hybrid fallback)."""
        match_features = fields.get("matchfeatures", {})
        similarities = match_features.get("similarities", {})
        
        chunk_scores = [
            similarities.get(str(i), 0) 
            for i in range(len(fields.get("chunks", [])))
        ]
        
        chunks = fields.get("chunks", [])
        chunks_with_scores = list(zip(chunks, chunk_scores))
        
        return sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from vespa.application import Vespa
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Connect to Vespa
    try:
        vespa_app = Vespa(url="http://localhost", port=8080)
    except Exception as e:
        print(f"Failed to connect to Vespa: {e}")
        sys.exit(1)
    
    # Initialize retriever
    retriever = VespaStreamingLayeredRetriever(
        app=vespa_app,
        user="jo-bergum",
        pages=5,
        chunks_per_page=3,
        include_metadata=True
    )
    
    # Test queries
    test_queries = [
        "why is colbert effective for retrieval?",
        "what is late interaction in neural IR?",
        "how does token-level similarity work?",
    ]
    
    print("=" * 80)
    print("Testing VespaStreamingLayeredRetriever")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        try:
            docs = retriever.get_relevant_documents(query)
            
            print(f"Retrieved {len(docs)} documents\n")
            
            for i, doc in enumerate(docs[:2], 1):  # Show top 2
                print(f"Document {i}:")
                print(f"  Title: {doc.metadata['title']}")
                print(f"  Page: {doc.metadata['page']}")
                print(f"  Chunks: {doc.metadata['chunks_selected']}/{doc.metadata['total_chunks']}")
                print(f"  Content preview: {doc.page_content[:150]}...")
                
                # Show detailed scoring
                details = retriever.get_chunk_details(doc)
                if details:
                    print(f"  Chunk scores:")
                    for chunk_info in details['chunks']:
                        print(f"    [{chunk_info['rank']}] "
                              f"Total: {chunk_info['chunk_score']:.3f} = "
                              f"Distance: {chunk_info['distance_score']:.3f} + "
                              f"BM25: {chunk_info['text_score']:.3f}")
                print()
        
        except Exception as e:
            print(f"  Error: {e}\n")
    
    print("=" * 80)
    print("Testing with fallback retriever")
    print("=" * 80)
    
    fallback_retriever = VespaStreamingLayeredRetrieverWithFallback(
        app=vespa_app,
        user="jo-bergum",
        pages=5,
        chunks_per_page=3
    )
    
    try:
        docs = fallback_retriever.get_relevant_documents(test_queries[0])
        print(f"Successfully retrieved {len(docs)} documents")
        print(f"Using retriever: {docs[0].metadata['retriever']}")
    except Exception as e:
        print(f"Error: {e}")

