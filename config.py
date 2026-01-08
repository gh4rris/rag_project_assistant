import os


ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(ROOT_PATH, "data")
PROJECTS = os.path.join(DATA, "projects.json")
STOP_WORDS = os.path.join(DATA, "stopwords.txt")