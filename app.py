import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.schema import Document
from haystack.nodes import EmbeddingRetriever, OpenAITextEmbedder
from haystack.document_stores import InMemoryDocumentStore

warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(page_title="Recherche IA dans vos documents")
st.title("📄 Recherche intelligente avec Haystack")

uploaded_files = st.file_uploader("Upload files here", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    documents = []

    for file in uploaded_files:
        content = file.read().decode("utf-8", errors="ignore")
        documents.append(Document(content=content, meta={"name": file.name, "path": "upload"}))

    st.success("✅ Files recieved !")

    question = st.text_input("What are you looking for ?")

    if question:
        with st.spinner("Loading..."):
            document_store = InMemoryDocumentStore()
            document_store.write_documents(documents)

            embedder = OpenAITextEmbedder()
            retriever = EmbeddingRetriever(document_store=document_store, embedding_model=embedder)

            pipeline = Pipeline()
            pipeline.add_node(embedder, name="query_embedder", inputs=["query"])
            pipeline.add_node(retriever, name="retriever", inputs=["query_embedder"])

            results = pipeline.run(query=question)
            docs = results["documents"]

            if docs:
                top_doc = docs[0]
                st.markdown("### 📌 Result")
                st.write(top_doc.content)
                st.caption(f"Match in `{top_doc.meta['name']}` --> `{top_doc.meta['path']}`")
            else:
                st.warning("No match found.")
