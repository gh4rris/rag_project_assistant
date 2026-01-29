# Rag Q&A Project Assistant

A Retrieval-Augmented Generation assistant that answers questions about my portfolio projects by retrieving data from the README files.

Live Assistant: [https://ragprojectassistant-jwgekvzyney6auurumdl37.streamlit.app/](https://ragprojectassistant-jwgekvzyney6auurumdl37.streamlit.app/)

**Note:** You will need an OpenRouter API Key to use the assistant. The key given is used for your session only and not stored. The model used is free tier, an OpenRouter account can be set up here: [https://openrouter.ai/](https://openrouter.ai/) to generate a key.

The retrieval system uses a hybrid of keyword search (BM25) and semantic search (chunk embeddings). The results are then reranked using a cross-encoder and passed to an LLM to generate a response.

## Features

- Hybrid Retrival (Reciprocal Rank Fusion of keyword & semantic search)
- Cross-Encoder Reranking
- LLM Answer Generation (meta-llama/llama-3.3-70b-instruct)
- Embedding Cache
- Streamlit UI
- Evaluation Metrics (Precision@k, Recall@k & F1 Score)

## Data Format

Project documentation is stored as structured JSON, derived from markdown README files. Content is organised into indexed sections and any content is formatted before being passed to the LLM for generation.

## LLM Configuration

The OpenRouter API is used to make requests to the model and so an API key is required to run the project. The model currently used is: meta-llama/llama-3.3-70b-instruct:free. It is accessed using the OpenAI SDK.

## Evaluation

The app includes an evaluation page where the retrieval model can be evaluated against a golden dataset of queries and relevant responses. The k parameter can be adjusted for evaluation, and is set to a default of 4, as it is set for the main application. Recall@k is the primary metric, as retrieving all relevant data to answer a quetion is prioritised. Precision@k is a secondary metric for minimizing non-relevant data, and F1 score is a summary metric.

**Note:** No API Key is required for evaluation, as it is only testing data retrieval.

## Installation

- Clone the repo

```bash
git clone https://github.com/gh4rris/rag_project_assistant.git
cd rag_project_assistant
```

- Set up virtual environment

```bash
uv venv
uv sync
```

- Create a .env file:

```env
OPENROUTER_API_KEY="<YOUR_API_KEY_HERE>"
```

- Run Streamlit:

```bash
streamlit run main.py
```
