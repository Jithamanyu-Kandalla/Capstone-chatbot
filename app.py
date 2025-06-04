import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, TextLoader, UnstructuredImageLoader,
    UnstructuredVideoLoader
)
import os
import tempfile

st.set_page_config(page_title="Multi-File Chatbot", layout="wide")
st.title("📄 Chat with Any File")

uploaded_file = st.file_uploader("Upload your file", type=None)

def load_and_split_docs(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif extension in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif extension in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    elif extension in [".txt"]:
        loader = TextLoader(file_path)
    elif extension in [".png", ".jpg", ".jpeg"]:
        loader = UnstructuredImageLoader(file_path)
    elif extension in [".mp4", ".avi", ".mov"]:
        loader = UnstructuredVideoLoader(file_path)
    else:
        st.error(f"Unsupported file type: {extension}")
        return []
    
    return loader.load()

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.info("🔍 Processing the file...")
    
    documents = load_and_split_docs(file_path)
    if not documents:
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=retriever)

    query = st.text_input("Ask a question about your file:")
    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
        st.success(response)