from config import LLM_MODEL, OPENROUTER_URL, SYSTEM_PROMPT, FALLBACKS
from src.utils import load_prompt

from typing import Iterator
from openai import OpenAI


def generate_answer(question: str, documents: list[dict], api_key: str) -> Iterator[str]:
    client = OpenAI(
        base_url=OPENROUTER_URL,
        api_key=api_key
    )

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
        extra_body={
            "models": FALLBACKS
        },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content