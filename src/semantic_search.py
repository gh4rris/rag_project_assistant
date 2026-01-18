from src.utils import Project, Section, format_section_content
from config import CACHE, SEMANTIC_MODEL, CHUNK_SIZE, OVERLAP

import re
import os
import pickle
import numpy as np
from numpy import float32
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    chunk_embeddings: NDArray[float32] | None # array of embedded Section chunks
    chunk_metadata: list[dict] | None # metadata corresponding to chunk embeddings

    def __init__(self, cache=CACHE):
        self.model = SentenceTransformer(SEMANTIC_MODEL)
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self._cache = cache
        self._chunk_embeddings_path = os.path.join(cache, "chunk_embeddings.npy")
        self._chunk_metadata_path = os.path.join(cache, "metadata.json")

    def _generate_embedding(self, text: str) -> NDArray[float32]:
        if not text.strip():
            raise ValueError("Text empty or contains only whitespace")
        return self.model.encode([text])[0]
    
    def _split_sentences(self, text: str) -> list[str]:
        stripped_text = text.strip()
        if not stripped_text:
            return []
        sentences = re.split(r"(?<=[.!?]\s)", stripped_text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    def _semantic_chunk(self, section: Section, chunk_size: int=CHUNK_SIZE, overlap: int=OVERLAP) -> list[str]:
        match section.type:
            case "text":
                sentences = self._split_sentences(section.content)
                chunks = []
                i = 0
                while i < len(sentences):
                    chunk = sentences[i:i + chunk_size]
                    if chunks and len(chunk) <= overlap:
                        break
                    chunks.append(" ".join(chunk))
                    i += chunk_size - overlap
                return chunks
            case "list":
                chunks = section.content
            case "instructions":
                chunks = ["\n".join(item) for item in section.content]
        return chunks
    
    def _cosine_similarity(self, vec1: NDArray[float32], vec2: NDArray[float32]) -> float32:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)

    def search_chunks(self, query: str, project_map: dict[int, Project], section_map: dict[int, Section], limit: int) -> list[dict]:
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings loaded.")
        query_embedding = self._generate_embedding(query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = self._cosine_similarity(query_embedding, chunk_embedding)
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
            project = project_map[id]
            section = section_map[id]
            content = format_section_content(section)
            results.append(
                {
                    "project": project.name,
                    "url": project.repo_url,
                    "summary": project.summary,
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
                if section.type == "code":
                    continue
                content_chunks = self._semantic_chunk(section)
                for i, chunk in enumerate(content_chunks):
                    all_chunks.append(chunk)
                    metadata.append(
                        {
                            "section_id": section.id,
                            "chunk_idx": i,
                            "total_chunks": len(content_chunks)
                        }
                    )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

    def save(self) -> None:
        os.makedirs(self._cache, exist_ok=True)
        with open(self._chunk_embeddings_path, "wb") as f:
            pickle.dump(self.chunk_embeddings, f)
        with open(self._chunk_metadata_path, "wb") as f:
            pickle.dump(self.chunk_metadata, f)

    def load(self) -> None:
        with open(self._chunk_embeddings_path, "rb") as f:
            self.chunk_embeddings = pickle.load(f)
        with open(self._chunk_metadata_path, "rb") as f:
            self.chunk_metadata = pickle.load(f)    






