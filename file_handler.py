import pdfplumber
import pytesseract
from PIL import Image
import docx
import pandas as pd

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