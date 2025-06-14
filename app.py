import streamlit as st
from utils.file_handler import get_text_from_file
from utils.text_processor import get_summary, answer_question

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