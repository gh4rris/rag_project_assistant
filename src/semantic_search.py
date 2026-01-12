from src.utils import Project, Section, format_section_content
from config import CACHE, SEMANTIC_MODEL, CHUNK_SIZE, OVERLAP, RESULT_LIMIT

import re
import os
import pickle
import numpy as np
from numpy import float32
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    project_map: dict[int, dict]
    section_map: dict[int, Section]
    sections: list[Section]
    chunk_embeddings: NDArray[float32] | None
    chunk_metadata: list[dict] | None

    def __init__(self):
        self.model = SentenceTransformer(SEMANTIC_MODEL)
        self.project_map = {}
        self.section_map = {}
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.__chunk_embeddings_path = os.path.join(CACHE, "chunk_embeddings.npy")
        self.__chunk_metadata_path = os.path.join(CACHE, "metadata.json")

    def __generate_embedding(self, text: str) -> NDArray[float32]:
        if not text.strip():
            raise ValueError("Text empty or contains only whitespace")
        return self.model.encode([text])[0]

    def search_chunks(self, query: str, limit: int=RESULT_LIMIT) -> list[dict]:
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings loaded.")
        query_embedding = self.__generate_embedding(query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "section_id": self.chunk_metadata[i]["section_id"],
                    "score": score
                }
            )
        section_scores = {}
        for chunk_score in chunk_scores:
            section_id, score = chunk_score["section_id"], chunk_score["score"]
            if section_id not in section_scores or score > section_scores.get(section_id, 0):
                section_scores[section_id] = score
        sorted_scores = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for id, score in sorted_scores[:limit]:
            project = self.project_map[id]
            section = self.section_map[id]
            content = format_section_content(section)
            results.append(
                {
                    "project": project["name"],
                    "url": project["url"],
                    "id": section.id,
                    "label": section.label,
                    "content": content,
                    "type": section.type,
                    "score": score
                }
            )
        return results
    
    def build(self, projects: list[Project]) -> NDArray[float32]:
        all_chunks = []
        metadata = []
        for project in projects:
            for section in project.sections:
                self.project_map[section.id] = {
                    "name": project.name,
                    "url": project.repo_url
                }
                self.section_map[section.id] = section
                if section.type == "code":
                    continue
                content = format_section_content(section)
                content_chunks = semantic_chunk(content)
                for j, chunk in enumerate(content_chunks):
                    all_chunks.append(chunk)
                    metadata.append(
                        {
                            "section_id": section.id,
                            "chunk_idx": j,
                            "total_chunks": len(content_chunks)
                        }
                    )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

    def save(self) -> None:
        os.makedirs(CACHE, exist_ok=True)
        with open(self.__chunk_embeddings_path, "wb") as f:
            pickle.dump(f, self.chunk_embeddings)
        with open(self.__chunk_metadata_path, "wb") as f:
            pickle.dump(f, self.chunk_metadata)

    def load(self) -> None:
        with open(self.__chunk_embeddings_path, "rb") as f:
            self.chunk_embeddings = pickle.load(f)
        with open(self.chunk_metadata, "rb") as f:
            self.chunk_metadata = pickle.load(f)    


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

def cosine_similarity(vec1: NDArray[float32], vec2: NDArray[float32]) -> float32:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)