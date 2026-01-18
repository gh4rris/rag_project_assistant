from config import LLM_MODEL, SYSTEM_PROMPT
from src.utils import load_prompt

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
    doc_list = [f"{i}. Name: {doc["project"]}\nLabel: {doc["label"]}\n{doc["content"]}" for i, doc in enumerate(documents, 1)]
    doc_list_str = "\n\n".join(doc_list)
    system_prompt = load_prompt(SYSTEM_PROMPT)
    user_prompt = f"""
Question: {question}

Documents:
{doc_list_str}

Answer:"""
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return (response.choices[0].message.content or "").strip()