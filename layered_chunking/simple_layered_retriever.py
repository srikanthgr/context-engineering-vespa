"""
Simple conversion of VespaStreamingHybridRetriever to use layered ranking.
Direct drop-in replacement with minimal changes.
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from typing import List


class VespaStreamingLayeredRetriever(BaseRetriever):
    """
    LangChain retriever using Vespa's layered ranking.
    
    Key differences from VespaStreamingHybridRetriever:
    - Uses "layeredranking" rank profile (semantic + lexical filtering)
    - Extracts chunks from "best_chunks" instead of "similarities"
    - No chunk_similarity_threshold needed (join operation provides filtering)
    """
    
    app: Vespa
    user: str
    pages: int = 5
    chunks_per_page: int = 3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        response: VespaQueryResponse = self.app.query(
            yql="select id, url, title, page, authors, chunks from pdf where userQuery() or ({targetHits:20}nearestNeighbor(embedding,q))",
            groupname=self.user,
            ranking="layeredranking",  # Changed from "hybrid"
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
                f"Query failed with status code {response.status_code}, url={response.url} response={response.json}"
            )
        
        return self._parse_response(response)

    def _parse_response(self, response: VespaQueryResponse) -> List[Document]:
        documents: List[Document] = []
        
        for hit in response.hits:
            fields = hit["fields"]
            chunks_with_scores = self._get_best_chunks(fields)
            
            # Best k chunks already filtered by layered ranking (no threshold needed)
            best_chunks_on_page = " ### ".join(
                [
                    chunk
                    for chunk, score in chunks_with_scores[0 : self.chunks_per_page]
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
                    },
                )
            )
        
        return documents

    def _get_best_chunks(self, hit_fields: dict) -> List[tuple]:
        """
        Extract best chunks identified by Vespa's layered ranking.
        
        Replaces _get_chunk_similarities from hybrid retriever.
        Uses "best_chunks" from matchfeatures instead of "similarities".
        """
        match_features = hit_fields["matchfeatures"]
        best_chunks = match_features["best_chunks"]  # Changed from "similarities"
        
        # Get all chunks
        chunks = hit_fields["chunks"]
        
        # Build list of (chunk_text, score) for selected chunks
        chunks_with_scores = []
        for idx_str, score in best_chunks.items():
            idx = int(idx_str)
            if idx < len(chunks):
                chunks_with_scores.append((chunks[idx], score))
        
        # Sort by score descending (highest first)
        return sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)


# Example usage - compare both retrievers side by side
if __name__ == "__main__":
    from vespa.application import Vespa
    
    # Connect to Vespa
    vespa_app = Vespa(url="http://localhost", port=8080)
    
    # Layered retriever (new)
    layered_retriever = VespaStreamingLayeredRetriever(
        app=vespa_app,
        user="jo-bergum",
        pages=5,
        chunks_per_page=3
    )
    
    # Test query
    query = "why is colbert effective for retrieval?"
    
    print("=" * 80)
    print("LAYERED RANKING RETRIEVER")
    print("=" * 80)
    
    docs = layered_retriever.get_relevant_documents(query)
    
    for i, doc in enumerate(docs[:2], 1):
        print(f"\nDocument {i}:")
        print(f"  Title: {doc.metadata['title']}")
        print(f"  Page: {doc.metadata['page']}")
        print(f"  Content: {doc.page_content[:200]}...")
        
        # Show best chunks scores
        if "best_chunks" in doc.metadata["features"]:
            best_chunks = doc.metadata["features"]["best_chunks"]
            print(f"  Best chunks scores: {best_chunks}")

