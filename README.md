# Rag Q&A Project Assistant

A command-line Retrieval-Augmented Generation system that answers questions about my portfolio projects by retrieving data from the README files.

The retrieval system uses a hybrid of keyword search (BM25) and semantic search (chunk embeddings). The results are then reranked using a cross-encoder and passed to an LLM to generate a response.

## Features

- Hybrid Retrival (Reciprocal Rank Fusion of keyword & semantic search)
- Cross-Encoder Reranking
- LLM Answer Generation (xiaomi/mimo-v2-flash)
- Embedding Cache
- CLI Interface
- Evaluation Metrics (Precision@k, Recall@k & F1 Score)

## Data Format

Project documentation is stored as structured JSON, derived from markdown README files. Content is organised into indexed sections and any content is formatted before being passed to the LLM for generation.

## LLM Configuration

The OpenRouter API is used to make requests to the model and so an API key is required to run the project. The model used is: xiaomi/mimo-v2-flash:free. It is accessed using the OpenAI SDK.

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

## CLI Usage

### Ask a question

```bash
uv run main.py ask "What's the API instruction to create a post on a users profile for the networking site?"
```

Use the **ask** command followed by your question in quotes to ask the assistant a question about a project.

### Evaluation

```bash
uv run main.py evaluate -k 4
```

Use the **evaluate** command to evaluate the golden dataset using precision@k, recall@k and f1 score. The k parameter is optional and defaults to 4.
