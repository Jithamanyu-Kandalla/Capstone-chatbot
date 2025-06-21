import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import docx
import pandas as pd
import os
from openai import OpenAI

# --- CONFIG ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- FILE HANDLING ---
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

def get_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return extract_text_from_excel(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    elif uploaded_file.type.startswith("image/"):
        return extract_text_from_image(uploaded_file)
    else:
        return "Unsupported file type."

# --- LLM Functions ---
def get_summary(text):
    prompt = f"Summarize the following text:\n{text[:3000]}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def answer_question(text, question):
    prompt = f"Answer the following question based on the text:\n\nText:\n{text[:3000]}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Streamlit App ---
st.set_page_config(page_title="Document Insight Chatbot", layout="wide")
st.title("?? Document Insight Chatbot")

# --- Session State for Chat History and Processing ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

uploaded_file = st.file_uploader(
    "Upload a document (max 5MB)",
    type=["pdf", "docx", "xlsx", "txt", "jpg", "png"]
)

# --- File Size Limit (5MB for free tier) ---
if uploaded_file and uploaded_file.size > 5 * 1024 * 1024:
    st.error("File too large. Please upload a file smaller than 5MB.")
    uploaded_file = None

if uploaded_file:
    try:
        text = get_text_from_file(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        text = ""

    if text.strip() == "":
        st.warning("No readable text found in the document.")
    else:
        st.subheader("?? Summary")
        if st.button("Generate Summary", disabled=st.session_state.processing):
            st.session_state.processing = True
            with st.spinner("Summarizing..."):
                try:
                    summary = get_summary(text)
                    st.success(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
            st.session_state.processing = False

        st.subheader("?? Ask a Question")
        question = st.text_input("What would you like to know?")
        if st.button("Get Answer", disabled=st.session_state.processing):
            if question.strip():
                st.session_state.processing = True
                with st.spinner("Finding answer..."):
                    try:
                        answer = answer_question(text, question)
                        st.session_state.chat_history.append(
                            {"question": question, "answer": answer}
                        )
                        st.success(answer)
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.session_state.processing = False
            else:
                st.warning("Please enter a question.")

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("?? Chat History")
            for chat in st.session_state.chat_history:
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
