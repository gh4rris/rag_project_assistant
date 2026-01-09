from config import PROJECTS, STOP_WORDS

import json
import string
from pydantic import BaseModel
from typing import Any
from nltk.stem import PorterStemmer


class Section(BaseModel):
    id: int
    label: str
    content: str | list[str] | list[list[str]] | dict[str, Any]
    type: str

class Project(BaseModel):
    name: str
    repo_url: str
    sections: list[Section]

def load_projects() -> list[Project]:
    with open(PROJECTS, "r") as f:
        data = json.load(f)
    return [Project(**project) for project in data]

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

def format_section_content(section: Section) -> str:
    if section.type == "text":
        return section.content
    if section.type == "list":
        return "\n".join(section.content)
    if section.type == "instructions":
        groups = ["\n".join(group) for group in section.content]
        return "\n\n".join(groups)
    return ""
