import streamlit as st
import pandas as pd
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
st.set_page_config(page_title="🧠 Transformers QA Bot", layout="wide")

# ----------------- Embedding + QA -----------------
@st.cache_resource
def load_embedder():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

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

# ----------------- Utilities -----------------
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
        query_embedding = self.embedder.encode([query])[0]
        sims = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k = np.argsort(sims)[-k:][::-1]
        return [self.texts[i] for i in top_k if sims[i] > 0.2]

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
    text = ""
    try:
        if file.type == "text/plain":
            text = file.getvalue().decode("utf-8")
        elif file.type == "application/pdf":
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                text = "\n".join(p.extract_text() or '' for p in pdf.pages)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            from docx import Document
            doc = Document(file)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(file)
            text = df.to_string()
    except Exception as e:
        st.error(f"File error [{file.name}]: {e}")
    return text.strip()

# ----------------- Main App -----------------
def main():
    st.title("🤖 Chat with Documents (QA Model)")
    st.markdown("Upload documents and ask questions with extractive QA powered by Transformers.")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    tokenizer, embed_model = load_embedder()
    qa_pipeline = load_qa_model()

    with st.sidebar:
        st.header("📁 Upload Documents")
        files = st.file_uploader("Choose files", type=["txt", "pdf", "docx", "xlsx", "xls"], accept_multiple_files=True)

        if st.button("🔄 Process"):
            if files:
                embedder = Embedder(tokenizer, embed_model)
                splitter = SimpleTextSplitter()
                all_chunks = []
                for file in files:
                    text = extract_text_from_file(file)
                    all_chunks.extend(splitter.split_text(text))

                store = SimpleVectorStore(embedder)
                store.add_texts(all_chunks)
                st.session_state.vector_store = store
                st.success(f"{len(all_chunks)} chunks processed!")

    if st.session_state.vector_store:
        user_question = st.text_input("💬 Ask a question:")
        if user_question:
            with st.spinner("Searching context..."):
                chunks = st.session_state.vector_store.similarity_search(user_question, k=3)
                context = " ".join(chunks)

                if context:
                    result = qa_pipeline(question=user_question, context=context)
                    st.markdown(f"### 🤖 Answer:\n**{result['answer']}**")
                    if st.checkbox("Show context"):
                        st.text_area("Retrieved Context", context, height=200)
                else:
                    st.warning("No relevant context found.")

if __name__ == "__main__":
    main()
