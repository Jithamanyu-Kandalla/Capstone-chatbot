#!/usr/bin/env python3
import os
import sys
import tempfile
import shutil

# CRITICAL: Fix all Streamlit permission issues before any imports
def fix_streamlit_permissions():
    """Comprehensive fix for Streamlit permission issues in containers"""
    
    # Create a writable temp directory
    temp_dir = tempfile.mkdtemp()
    
    # Set ALL possible Streamlit environment variables
    streamlit_env_vars = {
        'STREAMLIT_CONFIG_DIR': temp_dir,
        'STREAMLIT_STATIC_DIR': temp_dir,
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
        'STREAMLIT_SERVER_HEADLESS': 'true',
        'STREAMLIT_SERVER_PORT': '7860',
        'STREAMLIT_SERVER_ADDRESS': '0.0.0.0',
        'STREAMLIT_GLOBAL_SUPPRESS_WARNING': 'true',
        'STREAMLIT_GLOBAL_SHOW_WARNING_ON_DIRECT_EXECUTION': 'false',
        'STREAMLIT_LOGGER_LEVEL': 'ERROR',
        'STREAMLIT_CLIENT_TOOLBAR_MODE': 'minimal',
        'STREAMLIT_SERVER_ENABLE_CORS': 'false',
        'STREAMLIT_GLOBAL_DISABLE_WATCHDOG_WARNING': 'true'
    }
    
    for key, value in streamlit_env_vars.items():
        os.environ[key] = value
    
    # Create the .streamlit directory structure
    streamlit_config_dir = os.path.join(temp_dir, '.streamlit')
    os.makedirs(streamlit_config_dir, exist_ok=True)
    
    # Create comprehensive config.toml
    config_content = """
[server]
headless = true
port = 7860
address = "0.0.0.0"
baseUrlPath = ""
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200
maxMessageSize = 200
enableWebsocketCompression = false

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverPort = 7860

[global]
suppressDeprecationWarnings = true
showWarningOnDirectExecution = false
disableWatchdogWarning = true

[client]
toolbarMode = "minimal"
showErrorDetails = false

[logger]
level = "ERROR"
enableRich = false

[theme]
base = "light"
"""
    
    # Write config file
    try:
        with open(os.path.join(streamlit_config_dir, 'config.toml'), 'w') as f:
            f.write(config_content)
    except Exception as e:
        print(f"Warning: Could not write config file: {e}")
    
    # Create credentials file to prevent machine ID generation
    try:
        credentials_content = """
[general]
email = ""
"""
        with open(os.path.join(streamlit_config_dir, 'credentials.toml'), 'w') as f:
            f.write(credentials_content)
    except Exception as e:
        print(f"Warning: Could not write credentials file: {e}")
    
    # Set Python path to include temp directory
    sys.path.insert(0, temp_dir)
    
    return temp_dir

# Apply the fix BEFORE any other imports
temp_directory = fix_streamlit_permissions()

# Now import everything else
import asyncio
import nest_asyncio

# Fix asyncio event loop issues
nest_asyncio.apply()

# Safe asyncio setup
def setup_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

setup_event_loop()

# Import Streamlit with error handling
try:
    import streamlit as st
    # Set page config immediately to prevent further config issues
    st.set_page_config(
        page_title="Document Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    print(f"Error importing Streamlit: {e}")
    sys.exit(1)

# Import other required libraries
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Simple implementations to replace complex dependencies
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
                    print(f"Error encoding text: {e}")
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
            print(f"Error in similarity search: {e}")
            return []

# Load models with caching and error handling
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_llm_components():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "microsoft/DialoGPT-small"  # Even smaller model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading language model: {e}")
        return None, None

# Initialize models
embedding_model = load_embedding_model()
tokenizer, language_model = load_llm_components()

# Initialize vector store
if embedding_model:
    vector_store = SimpleVectorStore(embedding_model)
else:
    vector_store = None

# File processing functions
def extract_text_from_file(file):
    """Extract text from a single uploaded file"""
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
            import docx
            doc = docx.Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(file)
            text = df.to_string()
    except Exception as e:
        st.error(f"Error processing file {file.name}: {e}")
    
    return text.strip()

def process_documents(uploaded_files):
    """Process all uploaded documents"""
    if not uploaded_files or not vector_store:
        return False
    
    all_texts = []
    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text:
            all_texts.append(text)
    
    if all_texts:
        # Split into chunks
        splitter = SimpleTextSplitter(chunk_size=512, chunk_overlap=50)
        all_chunks = []
        for text in all_texts:
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
        
        # Add to vector store
        vector_store.add_texts(all_chunks)
        return True
    
    return False

def generate_response(query, context=""):
    """Generate response using the language model"""
    if not tokenizer or not language_model:
        return "I'm sorry, but the language model is not available right now."
    
    try:
        if context:
            prompt = f"Context: {context[:500]}\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer:"
        
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=300, truncation=True)
        
        with torch.no_grad():
            outputs = language_model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
            # Clean up the response
            sentences = answer.split('.')
            clean_answer = '. '.join(sentences[:2])  # Take first 2 sentences
            return clean_answer if clean_answer else "I couldn't generate a proper response."
        
        return "I couldn't generate a proper response."
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)[:100]}"

# Streamlit UI
def main():
    st.title("🤖 Document Question Answering Chatbot")
    st.markdown("Upload documents and ask questions about their content!")
    
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
                    if process_documents(uploaded_files):
                        st.success("✅ Documents processed successfully!")
                    else:
                        st.error("❌ Error processing documents")
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Chat input
    user_question = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know about your documents?"
    )
    
    if user_question:
        with st.spinner("Thinking..."):
            # Get relevant context
            context = ""
            if vector_store and vector_store.texts:
                relevant_chunks = vector_store.similarity_search(user_question, k=3)
                context = " ".join(relevant_chunks)
            
            # Generate response
            response = generate_response(user_question, context)
            
            # Display response
            st.markdown("### 🤖 Response:")
            st.markdown(response)
            
            # Show context if available
            if context and st.checkbox("Show source context"):
                st.markdown("### 📝 Source Context:")
                st.text_area("Relevant text from documents:", context, height=150)
    
    # Instructions
    if not uploaded_files:
        st.info("👆 Please upload some documents using the sidebar to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("*Upload your documents and start asking questions!*")

if __name__ == "__main__":
    main()