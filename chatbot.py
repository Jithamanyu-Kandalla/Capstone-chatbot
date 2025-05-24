import streamlit as st
import torch
import pdfplumber
import pandas as pd
import docx
import spacy
import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Authenticate Hugging Face API securely
hf_token = os.getenv("HUGGINGFACE_TOKEN")  # Load from environment secrets
login(token=hf_token)

# Load NLP Model (spaCy for Named Entity Recognition)
nlp = spacy.load("en_core_web_sm")

# Load Semantic Embedding Model for Contextual Retrieval
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load LLM Model
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

# Initialize FAISS for RAG
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts([], embedding_model)

# Extract text from multiple files
def extract_text_from_files(files):
    """Extracts and preprocesses text from multiple document types."""
    all_text = ""

    for file in files:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                all_text += "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(file)
            all_text += df.to_string()

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            all_text += "\n".join([para.text for para in doc.paragraphs])

        elif file.type == "text/plain":
            all_text += file.getvalue().decode("utf-8")

    return preprocess_text(all_text)

# Enhance text preprocessing using NLP techniques
def preprocess_text(text):
    """Enhances text preprocessing using NLP techniques."""
    doc = nlp(text)
    processed_text = " ".join([token.text for token in doc if not token.is_stop])  # Remove stopwords
    return processed_text

# Split extracted text into smaller chunks for optimized retrieval
def chunk_text(text):
    """Splits large text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_text(text)

# Store extracted file text into FAISS vector database
def store_chunks_in_vector_db(texts):
    """Stores document chunks in FAISS."""
    global vector_db
    chunks = [chunk_text(txt) for txt in texts]
    flat_chunks = [item for sublist in chunks for item in sublist]  # Flatten list
    vector_db.add_texts(flat_chunks)

# Generate semantic embeddings for better retrieval
def embed_text(text):
    """Generate semantic embeddings for better retrieval."""
    return semantic_model.encode(text)

# Expand queries semantically before retrieval
def expand_query(query):
    """Use NLP to find related terms for better retrieval."""
    doc = nlp(query)
    synonyms = [token.text for token in doc if not token.is_stop]
    expanded_query = " ".join(synonyms)  # Remove unnecessary words for relevance
    return expanded_query

# Retrieve relevant document chunks using enhanced filtering
def retrieve_relevant_chunks(query):
    """Retrieves document chunks using semantic embeddings."""
    expanded_query = expand_query(query)  # Enhance query understanding
    query_embedding = embed_text(expanded_query)  # Convert query to embeddings

    retrieved_docs = vector_db.similarity_search_by_vector(query_embedding, k=5)
    return "\n".join([doc.page_content for doc in retrieved_docs])

# Streamlit UI
st.title("Hybrid Chatbot (NLP + LLM + RAG) with Multi-Document Retrieval & Semantic Filtering")

# File Upload Section
uploaded_files = st.file_uploader("Upload Files (PDF, Excel, Word, TXT)", accept_multiple_files=True)

# User Query Input
user_query = st.text_input("Ask a question:")

if user_query:
    # Process Files if Uploaded
    extracted_text = extract_text_from_files(uploaded_files) if uploaded_files else ""
    store_chunks_in_vector_db([extracted_text])  # Store chunked data
    
    # Retrieve relevant data & generate response
    context = retrieve_relevant_chunks(user_query)
    input_text = f"Context:\n{context}\nQuery: {user_query}"

    # Generate response using LLM
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=250)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Display Chatbot Response
    st.write("🔹 Chatbot Response:")
    st.write(response)