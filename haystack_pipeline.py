from haystack import Pipeline
from haystack.nodes import SentenceTransformersTextEmbedder, DenseRetriever, OpenAITextEmbedder
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document

def create_pipeline(documents):
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    query_embedder = OpenAITextEmbedder()
    retriever = DenseRetriever(document_store=document_store, embedding_model=query_embedder)
    
    pipeline = Pipeline()
    pipeline.add_node(query_embedder, name="query_embedder", inputs=["query"])
    pipeline.add_node(retriever, name="retriever", inputs=["query_embedder"])
    
    return pipeline