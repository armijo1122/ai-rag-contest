from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from llama_index.core import (ServiceContext,
                         KnowledgeGraphIndex)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from llama_index.core.callbacks import (CallbackManager, LlamaDebugHandler, )



from custom.template import SUMMARY_EXTRACT_TEMPLATE
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor


def read_data(path: str = "data") -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        required_exts=[
            ".txt",
        ],
    )
    return reader.load_data()


def build_pipeline(
    llm: LLM,
    embed_model: BaseEmbedding,
    template: str = None,
    vector_store: BasePydanticVectorStore = None,
) -> IngestionPipeline:
    transformation = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=50),
        CustomTitleExtractor(metadata_mode=MetadataMode.EMBED),
        CustomFilePathExtractor(last_path_length=4, metadata_mode=MetadataMode.EMBED),
        # SummaryExtractor(
        #     llm=llm,
        #     metadata_mode=MetadataMode.EMBED,
        #     prompt_template=template or SUMMARY_EXTRACT_TEMPLATE,
        # ),
        embed_model,
    ]

    return IngestionPipeline(transformations=transformation, vector_store=vector_store)


async def build_vector_store(
    config: dict, reindex: bool = False
) -> tuple[AsyncQdrantClient, QdrantVectorStore]:
    client = AsyncQdrantClient(
        # url=config["QDRANT_URL"],
        #location=":memory:"
        path="/content/local_qdrant"
    )
    if reindex:
        try:
            print("delete collection...")
            await client.delete_collection(config["COLLECTION_NAME"] or "aiops24")
        except UnexpectedResponse as e:
            print(f"Collection not found: {e}")

        try:
            print("create collection...")
            await client.create_collection(
                collection_name=config["COLLECTION_NAME"] or "aiops24",
                vectors_config=models.VectorParams(
                    size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT
                ),
            )
        except UnexpectedResponse:
            print("Collection already exists")
    return client, QdrantVectorStore(
        aclient=client,
        collection_name=config["COLLECTION_NAME"] or "aiops24",
        parallel=4,
        batch_size=32,
    )

def build_kg_engine(
    llm: LLM,
    embed_model: BaseEmbedding,
    similarity_top_n: 5
):
    documents = read_data("data")
    #setup the service context
    
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=SentenceSplitter(chunk_size=1024, chunk_overlap=50),
        callback_manager=callback_manager,
    )
    
    #setup the storage context
    
    graph_store = Neo4jGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    #Construct the Knowlege Graph Undex
    index = KnowledgeGraphIndex.from_documents(documents=documents,
                                            max_triplets_per_chunk=3,
                                            service_context=service_context,
                                            storage_context=storage_context,
                                            include_embeddings=True)
    query_engine = index.as_query_engine(include_text=True,
                                     response_mode ="tree_summarize",
                                     embedding_mode="hybrid",
                                     similarity_top_k=similarity_top_n,)
    return query_engine
