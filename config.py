import os

# paths
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(ROOT_PATH, "data")
PROJECTS = os.path.join(DATA, "projects.json")
STOP_WORDS = os.path.join(DATA, "stopwords.txt")

# InvertedIndex parameters
BM25_K1 = 1.5
BM25_B = 0.75