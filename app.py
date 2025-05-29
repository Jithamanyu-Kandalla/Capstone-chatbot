import streamlit as st
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Import ML libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
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
            start = max(start + self.chunk_size - self.chunk_overlap, end)
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
            return [self.texts[i] for i in top_indices if similarities[i] > 0.2]
        except Exception as e:
            st.error(f"Error in similarity search: {e}")
            return []

@st.cache_resource
def load_embedding_model():
    """Load embedding model"""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

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
        splitter = SimpleTextSplitter(chunk_size=500, chunk_overlap=50)
        all_chunks = []
        for text in all_texts:
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
        
        # Add to vector store
        vector_store.add_texts(all_chunks)
        return vector_store
    
    return None

def generate_simple_answer(question, context):
    """Generate a simple answer by finding the most relevant sentence"""
    if not context:
        return "I couldn't find relevant information to answer your question."
    
    try:
        # Split context into sentences
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
        
        if not sentences:
            return "I found some information but couldn't extract a clear answer."
        
        # Simple keyword matching approach
        question_words = set(question.lower().split())
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            # Calculate simple word overlap score
            overlap = len(question_words.intersection(sentence_words))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
        
        if best_sentence and best_score > 0:
            return best_sentence.strip() + "."
        else:
            # Return first relevant sentence if no good match
            return sentences[0].strip() + "."
    
    except Exception as e:
        return f"Error generating answer: {str(e)[:100]}"

def main():
    st.title("🤖 Document Question Answering Chatbot")
    st.markdown("Upload documents and ask questions about their content!")
    
    # Load embedding model
    embedding_model = load_embedding_model()
    
    if not embedding_model:
        st.error("Failed to load embedding model. Please refresh the page.")
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
                    if vector_store and vector_store.texts:
                        st.session_state.vector_store = vector_store
                        st.success(f"✅ Documents processed! {len(vector_store.texts)} chunks created.")
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
            
            if relevant_chunks:
                context = " ".join(relevant_chunks)
                
                # Generate answer
                answer = generate_simple_answer(user_question, context)
                
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
    if not st.session_state.vector_store:
        st.info("👆 Please upload some documents using the sidebar to get started!")
    
    # Show debug info
    if st.session_state.vector_store:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Processed chunks:** {len(st.session_state.vector_store.texts)}")

if __name__ == "__main__":
    main()