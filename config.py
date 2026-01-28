import os

# paths
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(ROOT_PATH, "data")
PROJECTS = os.path.join(DATA, "projects.json")
STOP_WORDS = os.path.join(DATA, "stopwords.txt")
GOLDEN_DATASET = os.path.join(DATA, "golden_dataset.json")
CACHE = os.path.join(ROOT_PATH, "cache")
TEST_CACHE = os.path.join(CACHE, "tests")
PROMPTS = os.path.join(ROOT_PATH, "prompts")
PAGES = os.path.join(ROOT_PATH, "pages")

# Streamlit
MAIN_PAGE = os.path.join(PAGES, "main_page.py")
EVALUATION_PAGE = os.path.join(PAGES, "evaluation.py")

# Prompts
SYSTEM_PROMPT = os.path.join(PROMPTS, "system_prompt.txt")

# LLM
LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# KeywordSearch parameters
BM25_K1 = 1.5
BM25_B = 0.75

# SemanticSearch parameters
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 4
OVERLAP = 1

# HybridSearch parameters
RRF_K = 60
LIMIT_MULTIPLYER = 4
THRESHOLD_RANGE = 4

# CrossEncoder
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"

# General
RESULT_LIMIT = 4

