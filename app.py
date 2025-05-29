
import os
os.environ["TORCH_HOME"] = "/tmp/torch"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from transformers import pipeline
import platform

st.title("🤖 QA Chatbot")
st.sidebar.markdown(f"🧪 Python version: `{platform.python_version()}`")

@st.cache_resource
def load_qa():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa = load_qa()

question = st.text_input("💬 Ask a question:")
context = st.text_area("📄 Paste some context here:")

if question and context:
    with st.spinner("🤔 Thinking..."):
        result = qa(question=question, context=context)
        st.success(f"Answer: {result['answer']}")
