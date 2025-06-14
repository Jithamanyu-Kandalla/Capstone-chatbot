import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import docx
import pandas as pd
import openai
import os

# --- CONFIG ---
openai.api_key = os.getenv("Secret")

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
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def answer_question(text, question):
    prompt = f"Answer the following question based on the text:\n\nText:\n{text[:3000]}\n\nQuestion: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Streamlit App ---
st.set_page_config(page_title="Document Insight Chatbot", layout="wide")
st.title("📄 Document Insight Chatbot")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "xlsx", "txt", "jpg", "png"])

if uploaded_file:
    text = get_text_from_file(uploaded_file)

    if text.strip() == "":
        st.warning("No readable text found in the document.")
    else:
        st.subheader("📌 Summary")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = get_summary(text)
                st.success(summary)

        st.subheader("💬 Ask a Question")
        question = st.text_input("What would you like to know?")
        if st.button("Get Answer"):
            with st.spinner("Finding answer..."):
                answer = answer_question(text, question)
                st.success(answer)