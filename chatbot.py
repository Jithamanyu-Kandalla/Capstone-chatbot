pip install -r requirements.txt

# Install required packages
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from nltk.translate.bleu_score import sentence_bleu
from langchain.memory import ConversationBufferMemory
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import CLIPProcessor, CLIPModel

# Initialize Models
LLM_MODEL = "mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

# Fine-Tuning Configuration (LoRA)
config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(base_model, config)

# Initialize Retrieval-Augmented Generation (RAG)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("document_store", embeddings=embedding_model)

# Initialize Memory for Multi-Turn Dialogues
memory = ConversationBufferMemory()

# Multi-Modal Setup (Text + Image Retrieval)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize Streamlit UI
st.title("Hybrid Chatbot (LLM + RAG + Multi-Turn Memory)")

# User Input
user_query = st.text_input("Enter your query:")
if user_query:
    # Retrieve Relevant Docs for RAG
    retrieved_docs = db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Prepare Multi-Turn Context
    past_conversations = memory.load_memory_variables({})
    input_text = f"Context:\n{context}\nConversation:\n{past_conversations}\nQuery: {user_query}"

    # Generate Response
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=250)
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Store Conversation History
    memory.save_context({"input": user_query}, {"output": bot_response})

    # Display Response
    st.write("🔹 Chatbot Response:")
    st.write(bot_response)

    # Evaluate BLEU Score
    reference = [[word for word in context.split()]]
    candidate = [word for word in bot_response.split()]
    bleu_score = sentence_bleu(reference, candidate)
    st.write(f"🔹 BLEU Score: {bleu_score}")

    # Evaluate Perplexity
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = gpt2_tokenizer.encode(user_query, return_tensors="pt")
    loss = gpt2_model(input_ids, labels=input_ids).loss
    perplexity = torch.exp(loss).item()
    st.write(f"🔹 Perplexity: {perplexity}")

    # Multi-Modal Retrieval (Example: Text + Image)
    multimodal_inputs = clip_processor(text=[user_query], images=["path/to/sample.jpg"], return_tensors="pt")
    multimodal_outputs = clip_model(**multimodal_inputs)
    st.write("🔹 Multi-Modal Feature Extraction Complete!")

# Run using: streamlit run chatbot.py
# streamlit run chatbot.py