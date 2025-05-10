pip! install pdfplumber

import streamlit as st
import pdfplumber
import docx
import pandas as pd
import re
import os
import io
import tempfile
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.service_account import Credentials
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import time
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Document QA Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state if not already
if 'qa_pipeline' not in st.session_state:
    st.session_state.qa_pipeline = None
if 'file_text' not in st.session_state:
    st.session_state.file_text = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Download NLTK resources if not already downloaded
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_resources()

# Text Preprocessing
@st.cache_data
def preprocess_text(text):
    """Preprocess text by lowercasing, removing punctuation, numbers, and stopwords, and lemmatizing"""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        tokens = word_tokenize(text)  # Tokenize the text
        tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error during text preprocessing: {e}")
        return text  # Return original text if preprocessing fails

# File Reading Functions
def read_pdf(uploaded_file):
    """Extract text from a PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        with pdfplumber.open(tmp_path) as pdf:
            text = ""
            total_pages = len(pdf.pages)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
                
                # Update progress
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {i+1}/{total_pages}")
                
        os.unlink(tmp_path)  # Remove temporary file
        status_text.empty()
        progress_bar.empty()
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def read_word(uploaded_file):
    """Extract text from a Word document"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        doc = docx.Document(tmp_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
        os.unlink(tmp_path)  # Remove temporary file
        return text
    except Exception as e:
        logger.error(f"Error extracting text from Word document: {e}")
        st.error(f"Error extracting text from Word document: {e}")
        return ""

def read_txt(uploaded_file):
    """Extract text from a text file"""
    try:
        text = uploaded_file.getvalue().decode('utf-8')
        return text
    except Exception as e:
        logger.error(f"Error extracting text from text file: {e}")
        st.error(f"Error extracting text from text file: {e}")
        return ""

def read_excel(uploaded_file):
    """Extract text from an Excel file"""
    try:
        data = pd.read_excel(uploaded_file)
        # Convert data to markdown for better readability
        return data.to_markdown(index=False)
    except Exception as e:
        logger.error(f"Error extracting text from Excel file: {e}")
        st.error(f"Error extracting text from Excel file: {e}")
        return ""

# Google Drive API Integration
def get_google_drive_credentials():
    """Get Google Drive API credentials either from environment or uploaded JSON"""
    if os.environ.get('GOOGLE_DRIVE_CREDENTIALS'):
        # If credentials are stored in environment variable (for deployment)
        credentials_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
        credentials_info = json.loads(credentials_json)
        return Credentials.from_service_account_info(credentials_info)
    else:
        # For local development or when credentials are uploaded
        credentials_file = st.sidebar.file_uploader(
            "Upload Google Drive credentials (JSON)", type=["json"]
        )
        if credentials_file:
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            return flow.run_local_server(port=0)
        return None

def list_files_from_google_drive():
    """List files from Google Drive"""
    creds = get_google_drive_credentials()
    if not creds:
        st.warning("Google Drive credentials not provided.")
        return []
    
    try:
        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(
            pageSize=20, 
            fields="files(id, name, mimeType, createdTime)",
            orderBy="createdTime desc"
        ).execute()
        files = results.get('files', [])
        return files
    except Exception as e:
        logger.error(f"Error listing files from Google Drive: {e}")
        st.error(f"Error accessing Google Drive: {e}")
        return []

def download_file_from_google_drive(file_id, file_name):
    """Download a file from Google Drive"""
    creds = get_google_drive_credentials()
    if not creds:
        st.warning("Google Drive credentials not provided.")
        return None
    
    try:
        service = build('drive', 'v3', credentials=creds)
        request = service.files().get_media(fileId=file_id)
        
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)
        
        done = False
        with st.spinner(f"Downloading {file_name}..."):
            while not done:
                status, done = downloader.next_chunk()
                st.progress(int(status.progress() * 100))
        
        file_content.seek(0)
        return file_content
    except Exception as e:
        logger.error(f"Error downloading file from Google Drive: {e}")
        st.error(f"Error downloading file: {e}")
        return None

# Chunking Function for Large Contexts
@st.cache_data
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for processing"""
    if not text:
        return []
        
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def query_chunks(chunks, question, qa_pipeline, top_k=3):
    """Query all chunks and return the best answer"""
    if not chunks or not question or not qa_pipeline:
        return "No text to analyze or question provided."
    
    try:
        answers = []
        with st.spinner("Searching for answers..."):
            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                # Update progress
                progress = (i + 1) / len(chunks)
                progress_bar.progress(progress)
                
                # Query the model
                try:
                    result = qa_pipeline({"context": chunk, "question": question})
                    if isinstance(result, list):  # Handle case where pipeline returns a list
                        for r in result:
                            answers.append({"answer": r["answer"], "score": r["score"], "context": chunk})
                    else:
                        answers.append({"answer": result["answer"], "score": result["score"], "context": chunk})
                except Exception as chunk_error:
                    logger.warning(f"Error processing chunk {i}: {chunk_error}")
                    continue
            
            progress_bar.empty()
            
        if not answers:
            return "I couldn't find an answer to your question in the document."
        
        # Sort answers by score and get top k
        answers.sort(key=lambda x: x["score"], reverse=True)
        top_answers = answers[:top_k]
        
        # Format top answers
        if top_k == 1:
            return top_answers[0]["answer"]
        else:
            response = "Here are the most relevant answers I found:\n\n"
            for i, ans in enumerate(top_answers):
                response += f"**Answer {i+1}** (confidence: {ans['score']:.2f}):\n{ans['answer']}\n\n"
            return response
    except Exception as e:
        logger.error(f"Error querying chunks: {e}")
        return f"An error occurred while processing your question: {e}"

# Hugging Face Model Setup
@st.cache_resource
def load_model(model_name="distilbert-base-cased-distilled-squad"):
    """Load QA model from Hugging Face"""
    try:
        with st.spinner(f"Loading model {model_name}... This may take a moment."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
            return qa_pipe
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

# Fine-Tuning Functionality
def fine_tune_model(train_file, val_file, epochs=3):
    """Fine-tune a Question Answering model"""
    try:
        # Save uploaded files to disk temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as train_tmp:
            train_tmp.write(train_file.getvalue())
            train_path = train_tmp.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as val_tmp:
            val_tmp.write(val_file.getvalue())
            val_path = val_tmp.name
        
        # Load dataset
        dataset = load_dataset('json', data_files={'train': train_path, 'validation': val_path})
        
        # Load base model
        model_name = "distilbert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        def preprocess_function(examples):
            return tokenizer(
                examples['question'], 
                examples['context'], 
                truncation=True, 
                padding='max_length', 
                max_length=384, 
                stride=128, 
                return_overflowing_tokens=True, 
                return_offsets_mapping=True
            )
        
        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        
        # Set up training arguments
        output_dir = "./results"
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation']
        )
        
        # Training progress
        with st.spinner("Fine-tuning the model... This may take several minutes."):
            train_result = trainer.train()
            
        # Save model
        model_save_path = "./fine_tuned_model"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        # Clean up temporary files
        os.unlink(train_path)
        os.unlink(val_path)
        
        # Create pipeline with fine-tuned model
        qa_pipeline = pipeline("question-answering", model=model_save_path, tokenizer=model_save_path)
        return qa_pipeline, train_result
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        st.error(f"Error during fine-tuning: {e}")
        return None, None

# UI Components
def render_sidebar():
    """Render the sidebar with options"""
    st.sidebar.title("Options")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_options = {
        "distilbert-base-cased-distilled-squad": "DistilBERT (Fast)",
        "deepset/roberta-base-squad2": "RoBERTa (Balanced)",
        "deepset/bert-large-uncased-whole-word-masking-squad2": "BERT Large (Accurate)"
    }
    selected_model = st.sidebar.selectbox(
        "Choose a QA model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    load_model_button = st.sidebar.button("Load Selected Model")
    if load_model_button:
        st.session_state.qa_pipeline = load_model(selected_model)
        if st.session_state.qa_pipeline:
            st.session_state.model_loaded = True
            st.sidebar.success(f"Model loaded successfully!")
    
    # Fine-tuning section
    st.sidebar.subheader("Fine-Tune Model")
    with st.sidebar.expander("Fine-Tuning Options"):
        train_file = st.file_uploader("Upload Training JSON File", type=["json"])
        val_file = st.file_uploader("Upload Validation JSON File", type=["json"])
        epochs = st.slider("Training Epochs", min_value=1, max_value=10, value=3)
        
        if train_file and val_file:
            if st.button("Start Fine-Tuning"):
                st.session_state.qa_pipeline, train_result = fine_tune_model(train_file, val_file, epochs)
                if st.session_state.qa_pipeline:
                    st.session_state.model_loaded = True
                    st.success("Model fine-tuned successfully!")
    
    # Customization options
    st.sidebar.subheader("Customization")
    with st.sidebar.expander("Preprocessing Options"):
        chunk_size = st.slider("Chunk Size (words)", min_value=100, max_value=1000, value=500)
        chunk_overlap = st.slider("Chunk Overlap (words)", min_value=10, max_value=200, value=50)
    
    with st.sidebar.expander("Answer Options"):
        top_k = st.slider("Number of answers to show", min_value=1, max_value=5, value=1)
    
    # Return user preferences
    return {
        "selected_model": selected_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k
    }

def render_file_upload_section():
    """Render the file upload section"""
    st.subheader("📄 Upload Documents")
    
    # File upload tab and Google Drive tab
    tab1, tab2 = st.tabs(["Upload File", "Google Drive"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a file (PDF, Word, Text, or Excel):", 
            type=["pdf", "docx", "txt", "xlsx", "xls"]
        )
        
        if uploaded_file:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            st.write("**File Details:**")
            st.json(file_details)
            
            process_button = st.button("Process Document")
            if process_button:
                process_uploaded_file(uploaded_file)
    
    with tab2:
        st.write("Connect to Google Drive to access your documents")
        if st.button("List Google Drive Files"):
            files = list_files_from_google_drive()
            if files:
                st.write(f"Found {len(files)} files in your Google Drive")
                for file in files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{file['name']} ({file['mimeType']})")
                    with col2:
                        if st.button("Download", key=file['id']):
                            file_content = download_file_from_google_drive(file['id'], file['name'])
                            if file_content:
                                st.success(f"Downloaded {file['name']}")
                                # Convert to a format that can be processed by the same functions
                                if file['mimeType'] == 'application/pdf':
                                    process_uploaded_file(file_content, file['name'], "pdf")
                                elif 'word' in file['mimeType']:
                                    process_uploaded_file(file_content, file['name'], "docx")
                                elif 'text' in file['mimeType']:
                                    process_uploaded_file(file_content, file['name'], "txt")
                                elif 'spreadsheet' in file['mimeType'] or 'excel' in file['mimeType']:
                                    process_uploaded_file(file_content, file['name'], "xlsx")
            else:
                st.info("No files found or Google Drive connection not established.")

def process_uploaded_file(uploaded_file, filename=None, filetype=None):
    """Process the uploaded file and extract text"""
    if filename is None:
        filename = uploaded_file.name
    
    if filetype is None:
        # Determine file type from filename
        if filename.endswith(".pdf"):
            filetype = "pdf"
        elif filename.endswith(".docx"):
            filetype = "docx"
        elif filename.endswith(".txt"):
            filetype = "txt"
        elif filename.endswith((".xlsx", ".xls")):
            filetype = "xlsx"
    
    with st.spinner(f"Processing {filename}..."):
        # Extract text based on file type
        if filetype == "pdf":
            text = read_pdf(uploaded_file)
        elif filetype == "docx":
            text = read_word(uploaded_file)
        elif filetype == "txt":
            text = read_txt(uploaded_file)
        elif filetype == "xlsx":
            text = read_excel(uploaded_file)
        else:
            st.error("Unsupported file type!")
            return
        
        if not text:
            st.error("Could not extract text from the file.")
            return
        
        # Store the extracted text in session state
        st.session_state.file_text = text
        
        # Preprocess text and create chunks
        preprocessed_text = preprocess_text(text)
        st.session_state.chunks = chunk_text(
            preprocessed_text, 
            st.session_state.get("chunk_size", 500),
            st.session_state.get("chunk_overlap", 50)
        )
        
        # Display success and text preview
        st.success(f"Successfully processed {filename}")
        with st.expander("Document Preview"):
            st.write(text[:1000] + ("..." if len(text) > 1000 else ""))
            st.info(f"Document contains {len(text.split())} words and {len(st.session_state.chunks)} chunks.")

def render_chat_section():
    """Render the chat interface"""
    st.subheader("💬 Ask Questions About Your Document")
    
    # Check if we have everything we need
    if not st.session_state.model_loaded:
        st.warning("Please load a model from the sidebar first.")
        return
    
    if not st.session_state.file_text:
        st.info("Please upload and process a document first.")
        return
    
    # Chat interface
    user_question = st.text_input("Ask a question about your document:", key="user_question")
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send")
    with col2:
        clear_button = st.button("Clear History")
    
    if clear_button:
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    if send_button and user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Get answer from model
        answer = query_chunks(
            st.session_state.chunks, 
            user_question, 
            st.session_state.qa_pipeline,
            st.session_state.get("top_k", 1)
        )
        
        # Add answer to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Clear the input box
        st.session_state.user_question = ""
        
        # Rerun to update the UI
        st.experimental_rerun()
    
    # Display chat history
    st.write("---")
    st.subheader("Conversation History")
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
        st.write("---")

def render_about_section():
    """Render information about the app"""
    with st.expander("About This App"):
        st.write("""
        ## Document QA Chatbot

        This application allows you to ask questions about your documents using state-of-the-art 
        question-answering AI models from Hugging Face. Upload PDF, Word, text, or Excel files 
        and get instant answers to your questions.

        ### Features
        - Upload documents from your computer or Google Drive
        - Select from different QA models to balance speed and accuracy
        - Fine-tune models on your own data
        - Customizable text processing parameters
        
        ### How It Works
        1. Upload your document
        2. Load or fine-tune a QA model
        3. Ask questions about your document
        4. Get AI-powered answers instantly

        Developed by Your Name/Organization
        """)

# Main Application
def main():
    st.title("📚 Document QA Chatbot")
    
    # Get user preferences from sidebar
    preferences = render_sidebar()
    
    # Store preferences in session state
    for key, value in preferences.items():
        st.session_state[key] = value
    
    # Load default model if not already loaded
    if not st.session_state.model_loaded and not st.session_state.qa_pipeline:
        with st.spinner("Loading default model..."):
            st.session_state.qa_pipeline = load_model()
            if st.session_state.qa_pipeline:
                st.session_state.model_loaded = True
                st.sidebar.success("Default model loaded!")
    
    # Render main sections
    render_file_upload_section()
    render_chat_section()
    render_about_section()

if __name__ == "__main__":
    main()
