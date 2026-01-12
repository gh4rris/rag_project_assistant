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

# SemanticSearch parameters
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 4
OVERLAP = 1

# HybridSearch parameters
RRF_K = 60

# General
RESULT_LIMIT = 5

x = {7: {"id": 7, "score": 20}, 3: {"id": 3, "score": 13}, 19: {"id": 19, "score": 14}}
y = sorted(x.values(), key=lambda x: x["score"], reverse=True)
print(y)