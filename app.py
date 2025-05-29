import os
# Fix for PyTorch and Streamlit on Streamlit Cloud
os.environ["TORCH_HOME"] = "/tmp/torch"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🤖 Chatbot with Transformers", layout="wide")

# --------------------- MODEL LOADING ---------------------
@st.cache_resource
def load_embedder():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# --------------------- EMBEDDER ---------------------
class Embedder:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    def encode(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            return self.mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

# --------------------- VECTOR STORE ---------------------
class SimpleVectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.texts = []
        self.embeddings = []

    def add_texts(self, texts):
        for text in texts:
            if text.strip():
                emb = self.embedder.encode([text])[0]
                self.texts.append(text)
                self.embeddings.append(emb)

    def similarity_search(self, query, k=3):
        if not self.embeddings:
            return []
        query_emb = self.embedder.encode([query])[0]
        sims = cosine_similarity([query_emb], self.embeddings)[0]
        top_k = np.argsort(sims)[-k:][::-1]
        return [self.texts[i] for i in top_k if sims[i] > 0.2]

# --------------------- UTILS ---------------------
class SimpleTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        return chunks

def extract_text_from_file(file):
    try:
        if file.type == "text/plain":
            return file.getvalue().decode("utf-8")
        elif file.type == "application/pdf":
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                return "\n".join(p.extract_text() or '' for p in pdf.pages)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            from docx import Document
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(file)
            return df.to_string()
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
    return ""

# --------------------- APP ---------------------
def main():
    st.title("📄 Chat with Your Documents (Transformers + PyTorch)")
    st.markdown("Upload PDFs, Word docs, Excel or text files and ask questions about them.")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    tokenizer, embed_model = load_embedder()
    qa_model = load_qa_pipeline()

    with st.sidebar:
        st.header("📁 Upload Files")
        files = st.file_uploader("Upload files", type=["pdf", "txt", "docx", "xlsx", "xls"], accept_multiple_files=True)

        if st.button("🔄 Process Documents"):
            if files:
                st.info("⏳ Processing files...")
                embedder = Embedder(tokenizer, embed_model)
                splitter = SimpleTextSplitter()
                chunks = []
                for file in files:
                    text = extract_text_from_file(file)
                    if text:
                        chunks.extend(splitter.split_text(text))

                vector_store = SimpleVectorStore(embedder)
                vector_store.add_texts(chunks)
                st.session_state.vector_store = vector_store
                st.success(f"✅ Processed {len(chunks)} chunks.")

    if st.session_state.vector_store:
        user_question = st.text_input("💬 Ask a question:")
        if user_question:
            with st.spinner("🔎 Finding answer..."):
                context_chunks = st.session_state.vector_store.similarity_search(user_question, k=3)
                context = " ".join(context_chunks)

                if context:
                    result = qa_model(question=user_question, context=context)
                    st.markdown("### 🤖 Answer")
                    st.success(result["answer"])

                    if st.checkbox("📄 Show source context"):
                        st.text_area("Relevant Context", context, height=200)
                else:
                    st.warning("No relevant content found in documents.")

    if not st.session_state.vector_store:
        st.info("📌 Upload and process documents from the sidebar to get started.")

if __name__ == "__main__":
    main()
