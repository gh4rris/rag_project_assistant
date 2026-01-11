from src.utils import Section
from config import SEMANTIC_MODEL, CHUNK_SIZE, OVERLAP

import re
from numpy import ndarray, float32
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    section_map: dict[int, Section]
    sections: list[Section]

    def __init__(self):
        self.model = SentenceTransformer(SEMANTIC_MODEL)
        self.section_map = {}

    def __generate_embedding(self, text: str) -> ndarray[float32]:
        if not text.strip():
            raise ValueError("Text empty or contains only whitespace")
        return self.model.encode([text])[0]
    
    def build(self, sections: list[Section]) -> ndarray[float32]:
        self.sections = sections


def split_sentences(text: str) -> list[str]:
    stripped_text = text.strip()
    if not stripped_text:
        return []
    sentences = re.split(r"(?<=[.!?]\s)", stripped_text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def semantic_chunk(text: str, chunk_size: int=CHUNK_SIZE, overlap: int=OVERLAP) -> list[str]:
    sentences = split_sentences(text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i + chunk_size]
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks