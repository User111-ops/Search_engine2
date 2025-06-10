from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import DensePassageRetriever
from haystack.utils import Document

def create_pipeline(documents):
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    query_embedder = OpenAITextEmbedder()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    pipeline = Pipeline()
    pipeline.add_component("query_embedder", query_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")

    return pipeline
