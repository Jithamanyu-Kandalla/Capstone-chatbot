import streamlit as st
import pandas as pd
import numpy as np
import warnings
import tempfile
import os

# Suppress warnings first
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set page config
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Import ML libraries after Streamlit setup
try:
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import pipeline
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

class SimpleTextSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text):
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap
            if start >= text_length:
                break
        
        return chunks

class SimpleVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.texts = []
        self.embeddings = []
    
    def add_texts(self, texts):
        for text in texts:
            if text and text.strip():
                self.texts.append(text)
                try:
                    embedding = self.embedding_model.encode([text])
                    self.embeddings.append(embedding[0])
                except Exception as e:
                    st.error(f"Error encoding text: {e}")
                    continue
    
    def similarity_search(self, query, k=3):
        if not self.texts or not self.embeddings:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = np.argsort(similarities)[-k:][::-1]
            return [self.texts[i] for i in top_indices if similarities[i] > 0.1]
        except Exception as e:
            st.error(f"Error in similarity search: {e}")
            return []

@st.cache_resource
def load_models():
    """Load models with better error handling"""
    try:
        # Load embedding model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Use a simpler, more reliable QA pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad"
        )
        
        return embedding_model, qa_pipeline
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def extract_text_from_file(file):
    """Extract text from uploaded file"""
    text = ""
    try:
        if file.type == "text/plain":
            text = file.getvalue().decode("utf-8")
        elif file.type == "application/pdf":
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            from docx import Document
            doc = Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(file)
            text = df.to_string()
    except Exception as e:
        st.error(f"Error processing file {file.name}: {e}")
    
    return text.strip()

def process_documents(uploaded_files, embedding_model):
    """Process documents and create vector store"""
    if not uploaded_files or not embedding_model:
        return None
    
    all_texts = []
    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text:
            all_texts.append(text)
    
    if all_texts:
        # Create vector store
        vector_store = SimpleVectorStore(embedding_model)
        
        # Split into chunks
        splitter = SimpleTextSplitter(chunk_size=512, chunk_overlap=50)
        all_chunks = []
        for text in all_texts:
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
        
        # Add to vector store
        vector_store.add_texts(all_chunks)
        return vector_store
    
    return None

def generate_answer(question, context, qa_pipeline):
    """Generate answer using QA pipeline"""
    if not qa_pipeline or not context:
        return "I couldn't find relevant information to answer your question."
    
    try:
        # Limit context length to avoid token limits
        max_context_length = 2000
        if len(context) > max_context_length:
            context = context[:max_context_length]
        
        result = qa_pipeline(question=question, context=context)
        
        if result['score'] > 0.1:  # Confidence threshold
            return result['answer']
        else:
            return "I couldn't find a confident answer to your question in the provided documents."
    
    except Exception as e:
        return f"Error generating answer: {str(e)[:100]}"

def main():
    st.title("🤖 Document Question Answering Chatbot")
    st.markdown("Upload documents and ask questions about their content!")
    
    # Load models
    embedding_model, qa_pipeline = load_models()
    
    if not embedding_model or not qa_pipeline:
        st.error("Failed to load required models. Please refresh the page.")
        return
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'xlsx', 'xls'],
            help="Upload PDF, Word, Excel, or text files"
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🔄 Process Documents"):
                with st.spinner("Processing documents..."):
                    vector_store = process_documents(uploaded_files, embedding_model)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.success("✅ Documents processed successfully!")
                    else:
                        st.error("❌ Error processing documents")
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    user_question = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know about your documents?"
    )
    
    if user_question and st.session_state.vector_store:
        with st.spinner("Finding answer..."):
            # Get relevant context
            relevant_chunks = st.session_state.vector_store.similarity_search(user_question, k=3)
            context = " ".join(relevant_chunks)
            
            if context:
                # Generate answer
                answer = generate_answer(user_question, context, qa_pipeline)
                
                # Display response
                st.markdown("### 🤖 Response:")
                st.markdown(answer)
                
                # Show context if requested
                if st.checkbox("Show source context"):
                    st.markdown("### 📝 Source Context:")
                    st.text_area("Relevant text from documents:", context, height=150)
            else:
                st.warning("No relevant information found in the uploaded documents.")
    
    elif user_question and not st.session_state.vector_store:
        st.warning("Please upload and process documents first!")
    
    # Instructions
    if not uploaded_files:
        st.info("👆 Please upload some documents using the sidebar to get started!")

if __name__ == "__main__":
    main()