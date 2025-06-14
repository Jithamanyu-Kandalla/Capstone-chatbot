import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

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