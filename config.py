import os

# paths
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(ROOT_PATH, "data")
PROJECTS = os.path.join(DATA, "projects.json")
STOP_WORDS = os.path.join(DATA, "stopwords.txt")
CACHE = os.path.join(ROOT_PATH, "cache")

# KeywordSearch parameters
BM25_K1 = 1.5
BM25_B = 0.75
RESULT_LIMIT = 5

# Semanticsearch parameters
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 4
OVERLAP = 1