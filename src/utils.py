from config import PROJECTS, STOP_WORDS

import json
import string
from pydantic import BaseModel
from typing import Any
from nltk.stem import PorterStemmer


class Section(BaseModel):
    id: str
    content: str | list[str] | dict[str, Any]
    type: str

class Project(BaseModel):
    project_name: str
    repo_url: str
    sections: list[Section]

def load_projects() -> list[Project]:
    with open(PROJECTS, "r") as f:
        return json.load(f)

def load_stop_words() -> list[str]:
    with open(STOP_WORDS, "r") as f:
        return f.read().splitlines()

def tokenize_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    stop_words = load_stop_words()
    filtered_words = [word for word in text.split() if word not in stop_words]
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in filtered_words]
