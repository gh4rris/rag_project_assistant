from config import MODEL

import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

def generate_answer(question: str, documents: list[dict]) -> str:
    doc_list = [f"{i}. Name: {doc["project"]}\n{doc["content"]}" for i, doc in enumerate(documents, 1)]
    doc_list_str = "\n\n".join(doc_list)
    prompt = f"""You will be provided with a question about a programming project from a dataset of multiple projects, along with some documents to provide additional context. Each document will include a project name and some content. Use any content that you deem relevant to answer the question, and always include any project names that you are refering to.

Question: {question}

Documents:
{doc_list_str}

Answer:"""
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return (response.choices[0].message.content or "").strip()