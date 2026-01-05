from fastapi import FastAPI
from contextlib import asynccontextmanager
from logging import getLogger
from vespa.package import ApplicationPackage
from vespa.deployment import VespaDocker
from vespa.io import VespaQueryResponse
from vespa.application import Vespa
import json
from typing import List, Dict
from pathlib import Path
import os
import unicodedata
from vespa.package import Component, Parameter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LCDocument
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

logger = getLogger(__name__)
vespa_app = None
vespa_docker = None

# Set DOCKER_HOST for macOS Docker Desktop
# Docker Desktop on macOS uses ~/.docker/run/docker.sock by default
docker_socket_path = Path.home() / ".docker" / "run" / "docker.sock"
if docker_socket_path.exists() and "DOCKER_HOST" not in os.environ:
    os.environ["DOCKER_HOST"] = f"unix://{docker_socket_path}"

logger = getLogger(__name__)

def sample_pdfs():
    return [
        {
            "title": "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction",
            "url": "https://arxiv.org/pdf/2112.01488.pdf",
            "authors": "Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia",
        },
        {
            "title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
            "url": "https://arxiv.org/pdf/2004.12832.pdf",
            "authors": "Omar Khattab, Matei Zaharia",
        },
        {
            "title": "On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval",
            "url": "https://arxiv.org/pdf/2108.11480.pdf",
            "authors": "Craig Macdonald, Nicola Tonellotto",
        },
        {
            "title": "A Study on Token Pruning for ColBERT",
            "url": "https://arxiv.org/pdf/2112.06540.pdf",
            "authors": "Carlos Lassance, Maroua Maachou, Joohee Park, Stéphane Clinchant",
        },
        {
            "title": "Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval",
            "url": "https://arxiv.org/pdf/2106.11251.pdf",
            "authors": "Xiao Wang, Craig Macdonald, Nicola Tonellotto, Iadh Ounis",
        },
    ]

from vespa.package import Schema, Document, Field, FieldSet, HNSW


def create_schema():
    schema = Schema(
    name="pdf",
    mode="streaming",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary", "index"]),
            Field(name="title", type="string", indexing=["summary", "index"]),
            Field(name="url", type="string", indexing=["summary", "index"]),
            Field(name="authors", type="array<string>", indexing=["summary", "index"]),
            Field(name="page", type="int", indexing=["summary", "index"]),
            Field(
                name="metadata",
                type="map<string,string>",
                indexing=["summary", "index"],
            ),
            Field(
                name="chunks", 
                type="array<string>", 
                indexing=["summary", "index"],
                index=["enable-bm25"],
                select_elements_by="best_chunks"
            ),
            Field(
                name="embedding",
                type="tensor(chunk{}, x[384])",
                indexing=["input chunks", "embed e5", "attribute", "index"],
                ann=HNSW(distance_metric="angular"),
                is_document_field=False,
            ),
        ],
    ),
    fieldsets=[FieldSet(name="default", fields=["chunks", "title"])],
    )
    return schema

def create_component_config():
    components=[
        Component(
            id="e5",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    "transformer-model",
                    {
                        "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/e5-small-v2-int8.onnx"
                    },
                ),
                Parameter(
                    "tokenizer-model",
                    {
                        "url": "https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/tokenizer.json"
                    },
                ),
            ],
        )
    ]
    return components

from vespa.package import RankProfile, Function, FirstPhaseRanking

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,  # chars, not llm tokens
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

def create_documents():
    import hashlib
    my_docs_to_feed = []
    my_docs_to_feed = []
    for pdf in sample_pdfs():
        url = pdf["url"]
        loader = PyPDFLoader(url)
        pages = loader.load_and_split()
        for index, page in enumerate(pages):
            source = page.metadata["source"]
            chunks = text_splitter.transform_documents([page])
            text_chunks = [chunk.page_content for chunk in chunks]
            text_chunks = [remove_control_characters(chunk) for chunk in text_chunks]
            page_number = index + 1
            vespa_id = f"{url}#{page_number}"
            hash_value = hashlib.sha1(vespa_id.encode()).hexdigest()
            fields = {
                "title": pdf["title"],
                "url": url,
                "page": page_number,
                "id": hash_value,
                "authors": [a.strip() for a in pdf["authors"].split(",")],
                "chunks": text_chunks,
                "metadata": page.metadata,
            }
            my_docs_to_feed.append(fields)
        return my_docs_to_feed

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

def create_rank_profile():
    rank_profile = RankProfile(
    name="hybrid",
    inputs=[("query(q)", "tensor(x[384])")],
    functions=[
        Function(
            name="similarities",
            expression="cosine_similarity(query(q), attribute(embedding),x)",
        )
    ],
    first_phase=FirstPhaseRanking(
        expression="nativeRank(title) + nativeRank(chunks) + reduce(similarities, max, chunk)",
        rank_score_drop_limit=0.0,
    ),
    match_features=[
        "closest(embedding)",
        "similarities",
        "nativeRank(chunks)",
        "nativeRank(title)",
        "elementSimilarity(chunks)",
    ],
)
    return rank_profile

def create_layered_rank_profile():
    rank_profile = RankProfile(
        name="layeredranking",  
        inputs = [("query(q)", "tensor(x[384])")],
        functions=[
            Function(
                name="my_distance",
                expression="euclidean_distance(query(q), attribute(embedding), x)",
            ),
            Function(
                name="my_distance_scores",
                expression="1 / (1+my_distance)",
            ),
            Function(
                name="my_text_scores",
                expression="elementwise(bm25(chunks), chunk, float)",
            ),
            Function(
                name="chunk_scores",
                expression="join(my_distance_scores, my_text_scores, f(a,b)(a+b))",
            ),
            Function(
                name="best_chunks",
                expression="top(3, chunk_scores)",
            ),
        ],
        first_phase="sum(chunk_scores())",
        match_features=[
            "my_distance",
            "my_distance_scores", 
            "my_text_scores",
            "chunk_scores",
            "best_chunks",
        ],
        summary_features=["best_chunks"]
    )
    return rank_profile

def create_layered_with_second_phase():
    from vespa.package import SecondPhaseRanking
    
    rank_profile = RankProfile(
        name="layered_with_second_phase",  
        inputs=[("query(q)", "tensor(x[384])")],
        functions=[
            Function(
                name="my_distance",
                expression="euclidean_distance(query(q), attribute(embedding), x)",
            ),
            Function(
                name="my_distance_scores",
                expression="1 / (1+my_distance)",
            ),
            Function(
                name="my_text_scores",
                expression="elementwise(bm25(chunks), chunk, float)",
            ),
            Function(
                name="chunk_scores",
                expression="join(my_distance_scores, my_text_scores, f(a,b)(a+b))",
            ),
            Function(
                name="best_chunks",
                expression="top(3, chunk_scores)",
            ),
            # Additional second-phase features
            Function(
                name="title_score",
                expression="nativeRank(title)",
            ),
            Function(
                name="max_similarity",
                expression="reduce(my_distance_scores, max, chunk)",
            ),
            Function(
                name="avg_similarity", 
                expression="reduce(my_distance_scores, avg, chunk)",
            ),
        ],
        first_phase="sum(chunk_scores())",
        second_phase=SecondPhaseRanking(
            expression="sum(chunk_scores()) * 0.7 + title_score * 0.2 + max_similarity * 0.1",
            rerank_count=100  # Rerank top 100 from first phase
        ),
        match_features=[
            "my_distance",
            "my_distance_scores", 
            "my_text_scores",
            "chunk_scores",
            "best_chunks",
            "title_score",
            "max_similarity",
            "avg_similarity",
        ],
        summary_features=["best_chunks"]
    )
    return rank_profile

def create_layered_with_diversity():
    from vespa.package import SecondPhaseRanking
    
    rank_profile = RankProfile(
        name="layered_with_diversity",  
        inputs=[("query(q)", "tensor(x[384])")],
        functions=[
            Function(
                name="my_distance",
                expression="euclidean_distance(query(q), attribute(embedding), x)",
            ),
            Function(
                name="my_distance_scores",
                expression="1 / (1+my_distance)",
            ),
            Function(
                name="my_text_scores",
                expression="elementwise(bm25(chunks), chunk, float)",
            ),
            Function(
                name="chunk_scores",
                expression="join(my_distance_scores, my_text_scores, f(a,b)(a+b))",
            ),
            Function(
                name="best_chunks",
                expression="top(3, chunk_scores)",
            ),
            # Diversity metrics
            Function(
                name="chunk_spread",
                expression="reduce(my_distance_scores, max, chunk) - reduce(my_distance_scores, min, chunk)",
            ),
            Function(
                name="avg_chunk_score",
                expression="reduce(chunk_scores, avg, chunk)",
            ),
            Function(
                name="max_chunk_score",
                expression="reduce(chunk_scores, max, chunk)",
            ),
        ],
        first_phase="sum(chunk_scores())",
        second_phase=SecondPhaseRanking(
            # Reward documents with diverse chunk scores (more comprehensive coverage)
            # Higher spread = more diverse content
            expression="sum(chunk_scores()) * 0.7 + chunk_spread * 2.0 + avg_chunk_score * 0.3",
            rerank_count=50
        ),
        match_features=[
            "my_distance_scores", 
            "my_text_scores",
            "chunk_scores",
            "best_chunks",
            "chunk_spread",
            "avg_chunk_score",
            "max_chunk_score",
        ],
        summary_features=["best_chunks"]
    )
    return rank_profile

def create_layered_with_maxsim():
    from vespa.package import SecondPhaseRanking
    
    rank_profile = RankProfile(
        name="layered_with_maxsim",  
        inputs=[("query(q)", "tensor(x[384])")],
        functions=[
            Function(
                name="my_distance",
                expression="euclidean_distance(query(q), attribute(embedding), x)",
            ),
            Function(
                name="my_distance_scores",
                expression="1 / (1+my_distance)",
            ),
            Function(
                name="my_text_scores",
                expression="elementwise(bm25(chunks), chunk, float)",
            ),
            Function(
                name="chunk_scores",
                expression="join(my_distance_scores, my_text_scores, f(a,b)(a+b))",
            ),
            Function(
                name="best_chunks",
                expression="top(3, chunk_scores)",
            ),
            # MaxSim-style scoring (ColBERT-like)
            Function(
                name="maxsim_score",
                expression="reduce(my_distance_scores, max, chunk)",
            ),
            Function(
                name="sumsim_score",
                expression="reduce(my_distance_scores, sum, chunk)",
            ),
        ],
        first_phase="sum(chunk_scores())",
        second_phase=SecondPhaseRanking(
            # More expensive but precise: combine max and sum similarities
            expression="maxsim_score * 10 + sumsim_score * 0.1 + sum(chunk_scores()) * 0.5",
            rerank_count=100
        ),
        match_features=[
            "my_distance_scores", 
            "my_text_scores",
            "chunk_scores",
            "best_chunks",
            "maxsim_score",
            "sumsim_score",
        ],
        summary_features=["best_chunks"]
    )
    return rank_profile

def create_layered_with_normalized_fusion():
    from vespa.package import SecondPhaseRanking
    
    rank_profile = RankProfile(
        name="layered_normalized",  
        inputs=[
            ("query(q)", "tensor(x[384])"),
        ],
        functions=[
            Function(
                name="my_distance",
                expression="euclidean_distance(query(q), attribute(embedding), x)",
            ),
            Function(
                name="my_distance_scores",
                expression="1 / (1+my_distance)",
            ),
            Function(
                name="my_text_scores",
                expression="elementwise(bm25(chunks), chunk, float)",
            ),
            # Normalize scores before combining
            Function(
                name="normalized_semantic",
                expression="my_distance_scores / (reduce(my_distance_scores, sum, chunk) + 0.001)",
            ),
            Function(
                name="normalized_lexical",
                expression="my_text_scores / (reduce(my_text_scores, sum, chunk) + 0.001)",
            ),
            # Combine with fixed 50/50 weighting (balanced hybrid)
            Function(
                name="chunk_scores",
                expression="join(normalized_semantic, normalized_lexical, f(a,b)(a * 0.5 + b * 0.5))",
            ),
            Function(
                name="best_chunks",
                expression="top(3, chunk_scores)",
            ),
        ],
        first_phase="sum(chunk_scores())",
        second_phase=SecondPhaseRanking(
            expression="sum(chunk_scores()) * 100",  # Scale for better scoring
            rerank_count=100
        ),
        match_features=[
            "my_distance_scores", 
            "my_text_scores",
            "normalized_semantic",
            "normalized_lexical",
            "chunk_scores",
            "best_chunks",
        ],
        summary_features=["best_chunks"]
    )
    return rank_profile
class VespaStreamingLayeredRetriever(BaseRetriever):
    """
    LangChain retriever using Vespa's layered ranking.
    
    Key differences from VespaStreamingHybridRetriever:
    - Uses "layeredranking" rank profile (semantic + lexical filtering)
    - Extracts chunks from "best_chunks" instead of "similarities"
    - Join operation provides initial filtering, min_chunk_score provides additional threshold
    """
    
    app: Vespa
    user: str
    pages: int = 5
    chunks_per_page: int = 3
    min_chunk_score: float = 0.0  # Minimum score threshold for chunk inclusion

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[LCDocument]:
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

    def _parse_response(self, response: VespaQueryResponse) -> List[LCDocument]:
        documents: List[LCDocument] = []
        
        for hit in response.hits:
            fields = hit["fields"]
            chunks_with_scores = self._get_best_chunks(fields)
            
            # Filter chunks by score threshold and limit to chunks_per_page
            filtered_chunks = [
                chunk
                for chunk, score in chunks_with_scores[0 : self.chunks_per_page]
                if score >= self.min_chunk_score
            ]
            
            # Skip documents with no chunks passing the threshold
            if not filtered_chunks:
                continue
            
            best_chunks_on_page = " ### ".join(filtered_chunks)
            
            documents.append(
                LCDocument(
                    id=fields["id"],
                    page_content=best_chunks_on_page,
                    metadata={
                        "title": fields["title"],
                        "url": fields["url"],
                        "page": fields["page"],
                        "authors": fields["authors"],
                        "features": fields["matchfeatures"],
                        "chunks_filtered": len(chunks_with_scores) - len(filtered_chunks),
                        "chunks_returned": len(filtered_chunks),
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vespa_app, vespa_docker
    logger.info("Starting up: Building Vespa application package...")

    try:
        schema = create_schema()
        
        schema.add_rank_profile(create_rank_profile())
        schema.add_rank_profile(create_layered_rank_profile())

        schema.add_rank_profile(create_layered_with_second_phase())
        schema.add_rank_profile(create_layered_with_diversity())
        schema.add_rank_profile(create_layered_with_maxsim())
        schema.add_rank_profile(create_layered_with_normalized_fusion())

        package = ApplicationPackage(
            name="langchainstreaming",
            schema=[schema],
            components=create_component_config(),
        )
        logger.info("Deploying Vespa application to Docker...")
        vespa_docker = VespaDocker()
        vespa_app = vespa_docker.deploy(
            application_package=package,
            max_wait_configserver=300  # Wait up to 5 minutes for config server
        )
        logger.info(f"Successfully deployed Vespa application at {vespa_app.url}")
        logger.info("Vespa is ready! Use POST /batch-feed to load documents.")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    yield

    logger.info("Shutting down: Cleaning up Vespa Docker container...")
    try:
        if vespa_docker and vespa_docker.container:
            vespa_docker.container.stop()
            vespa_docker.container.remove()
            logger.info("Vespa Docker container stopped and removed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        vespa_app = None
        vespa_docker = None
    logger.info("Cleanup completed")

app = FastAPI(
    title="RAG Blueprint with Vespa",
    description="FastAPI application with Vespa RAG deployment for document search and retrieval",
    version="0.1.0",
    lifespan=lifespan   
)
from typing import Iterable

def vespa_feed(user: str) -> Iterable[dict]:
    for doc in create_documents():
        yield {"fields": doc, "id": doc["id"], "groupname": user}

@app.get("/batch-feed")    
async def batch_feed():
    from vespa.io import VespaResponse, VespaQueryResponse
    async with vespa_app.asyncio() as async_app:
        def callback(response: VespaResponse, id: str):
            if not response.is_successful():
                print(f"Document {id} failed to feed with status code {response.status_code}, url={response.url} response={response.get_json()}")
        vespa_app.feed_iterable(schema="pdf", iter=vespa_feed("jo-bergum"), namespace="personal", callback=callback)
        return {"message": "Documents fed successfully"}

from fastapi import Query
@app.get("/query")  
async def query_endpoint(q: str = Query(..., alias="query", description="Search query text")):
    response: VespaQueryResponse = vespa_app.query(
        yql="select id,title,page,chunks from pdf where userQuery() or ({targetHits:10}nearestNeighbor(embedding,q))",
        groupname="jo-bergum",
        ranking="hybrid",
        query="why is colbert effective?",
        body={
            "presentation.format.tensors": "short-value",
            "input.query(q)": 'embed(e5, "why is colbert effective?")',
        },
        timeout="2s",
    )   
    if not response.is_successful():
        return {
            "status": "error",
            "message": "Query failed",
            "status_code": response.status_code,
            "response": response.get_json()
        }
    return json.dumps(response.get_json(), indent=2)

@app.get("/query-layered")  
async def query_endpoint(q: str = Query(..., alias="query", description="Search query text")):
    response: VespaQueryResponse = vespa_app.query(
        yql="select id,title,page,chunks from pdf where userQuery() or ({targetHits:10}nearestNeighbor(embedding,q))",
        groupname="jo-bergum",
        ranking="layeredranking",
        query=q,
        body={
            "presentation.format.tensors": "short-value",
            "input.query(q)": f'embed(e5, "{q}")',
        },
        timeout="2s",
    )   
    if not response.is_successful():
        return {
            "status": "error",
            "message": "Query failed",
            "status_code": response.status_code,
            "response": response.get_json()
        }
    return json.dumps(response.get_json(), indent=2)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt_template = """You are a helpful AI assistant that provides comprehensive, detailed answers based on the given context.

Instructions:
- Provide a thorough answer using specific details, examples, and data points from the context
- Synthesize information from multiple sources when available
- Include relevant numbers, comparisons, and technical details
- Cite sources by mentioning the page number(s) and document title
- If the context contains tables, performance metrics, or experimental results, include them
- Structure your answer with clear paragraphs for better readability

Context:
{context}

Question: {question}

Detailed Answer:"""
prompt = ChatPromptTemplate.from_template(prompt_template)
model = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0.1,  # Lower temperature for more focused, deterministic answers
    model="gpt-4o-mini"  # Use GPT-4 variant for better comprehension
)


def format_prompt_context(docs) -> str:
    context = []
    for i, d in enumerate(docs, 1):
        context.append(f"--- Document {i} ---\n")
        context.append(f"Title: {d.metadata['title']}\n")
        context.append(f"Authors: {', '.join(d.metadata['authors'])}\n")
        context.append(f"URL: {d.metadata['url']}\n")
        context.append(f"Page: {d.metadata['page']}\n\n")
        context.append(f"{d.page_content}\n\n")
    return "".join(context)


def create_rag_chain(retriever):
    """Create a RAG chain with the given retriever."""
    return (
        {
            "context": retriever | format_prompt_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )


@app.get("/query-layered-retriever")  
async def query_endpoint(
    q: str = Query(..., alias="query", description="Search query text"),
    min_score: float = Query(0.0, alias="min_score", description="Minimum chunk score threshold (0.0-1.0)")
):
    retriever = VespaStreamingLayeredRetriever(
        app=vespa_app, 
        user="jo-bergum",
        min_chunk_score=min_score
    )
    documents = retriever._get_relevant_documents(q)
    return json.dumps([doc.model_dump() for doc in documents], indent=2)

@app.get("/query-layered-retriever-chain")
async def query_chain_endpoint(
    q: str = Query(..., alias="query", description="Search query text"),
    min_score: float = Query(0.0, alias="min_score", description="Minimum chunk score threshold (0.0-1.0)")
):
    retriever = VespaStreamingLayeredRetriever(
        app=vespa_app, 
        user="jo-bergum",
        min_chunk_score=min_score
    )
    chain = create_rag_chain(retriever)
    result = chain.invoke(q)
    return {"result": result}


@app.get("/query-layered-retriever-chain-second-phase")
async def query_chain_endpoint_second_phase(
    q: str = Query(..., alias="query", description="Search query text"),
    min_score: float = Query(0.0, alias="min_score", description="Minimum chunk score threshold (0.0-1.0)")
):
    retriever = VespaStreamingLayeredRetriever(
        app=vespa_app, 
        user="jo-bergum",
        min_chunk_score=min_score
    )
    chain = create_rag_chain(retriever)
    result = chain.invoke(q)
    return {"result": result}

@app.get("/query-layered-retriever-chain-diversity")
async def query_chain_endpoint_diversity(
    q: str = Query(..., alias="query", description="Search query text"),
    min_score: float = Query(0.0, alias="min_score", description="Minimum chunk score threshold (0.0-1.0)")
):
    retriever = VespaStreamingLayeredRetriever(
        app=vespa_app, 
        user="jo-bergum",
        min_chunk_score=min_score
    )
    chain = create_rag_chain(retriever)
    result = chain.invoke(q)
    return {"result": result}

@app.get("/query-layered-retriever-chain-maxsim")
async def query_chain_endpoint_maxsim(
    q: str = Query(..., alias="query", description="Search query text"),
    min_score: float = Query(0.0, alias="min_score", description="Minimum chunk score threshold (0.0-1.0)")
):
    retriever = VespaStreamingLayeredRetriever(
        app=vespa_app, 
        user="jo-bergum",
        min_chunk_score=min_score
    )
    chain = create_rag_chain(retriever)
    result = chain.invoke(q)
    return {"result": result}

@app.get("/query-layered-retriever-chain-normalized-fusion")
async def query_chain_endpoint_normalized_fusion(
    q: str = Query(..., alias="query", description="Search query text"),
    min_score: float = Query(0.0, alias="min_score", description="Minimum chunk score threshold (0.0-1.0)")
):
    retriever = VespaStreamingLayeredRetriever(
        app=vespa_app, 
        user="jo-bergum",
        min_chunk_score=min_score
    )
    chain = create_rag_chain(retriever)
    result = chain.invoke(q)
    return {"result": result}


@app.get("/query-layered-retriever-chain-global-phase")
async def query_chain_endpoint_global_phase(
    q: str = Query(..., alias="query", description="Search query text"),
    min_score: float = Query(0.0, alias="min_score", description="Minimum chunk score threshold (0.0-1.0)")
):
    retriever = VespaStreamingLayeredRetriever(
        app=vespa_app, 
        user="jo-bergum",
        min_chunk_score=min_score
    )
    chain = create_rag_chain(retriever)
    result = chain.invoke(q)
    return {"result": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)