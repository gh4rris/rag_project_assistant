from src.utils import Project, Section, tokenize_text, format_section_content
from config import CACHE, BM25_K1, BM25_B

import os
import math
import pickle
from collections import defaultdict, Counter


class KeywordSearch:
    index: defaultdict[str, set[int]]
    token_frequencies: defaultdict[int, Counter]
    section_lengths: dict[int, int]

    def __init__(self) -> None:
        self.index = defaultdict(set) # mapping tokens to sets of Section IDs
        self.token_frequencies = defaultdict(Counter) # mapping Section IDs to token Counter
        self.section_lengths = {} # mapping Section IDs to their token length

        self.__index_path = os.path.join(CACHE, "index.pkl")
        self.__token_frequencies = os.path.join(CACHE, "token_frequencies.pkl")
        self.__section_lengths = os.path.join(CACHE, "section_lengths.pkl")

    def _add_section(self, id: int, text: str) -> None:
        tokenized_text = tokenize_text(text)
        for token in set(tokenized_text):
            self.index[token].add(id)
        self.token_frequencies[id].update(tokenized_text)
        self.section_lengths[id] = len(tokenized_text)

    def _avg_section_length(self) -> float:
        if not self.section_lengths:
            return 0.0
        return sum(self.section_lengths.values()) / len(self.section_lengths)
    
    def _get_bm25_tf(self, id: int, token: str, k1: float=BM25_K1, b: float=BM25_B) -> float:
        section_length = self.section_lengths.get(id, 0)
        avg_section_length = self._avg_section_length()
        length_norm = (1 - b) + (b * (section_length / avg_section_length)) if avg_section_length > 0 else 1.0
        tf = self.token_frequencies.get(id, Counter())[token]
        return (tf * (k1 + 1)) / (tf + (k1 * length_norm))
    
    def _get_bm25_idf(self, token: str, section_map: dict[int, Section]) -> float:
        matches = self.index[token]
        return math.log((len(section_map) - len(matches) + 0.5) / (len(matches) + 0.5) + 1)
    
    def _bm25(self, id: int, token: str, section_map: dict[int, Section]) -> float:
        tf = self._get_bm25_tf(id, token)
        idf = self._get_bm25_idf(token, section_map)
        return tf * idf
    
    def bm25_search(self, query: str, project_map: dict[int, Project], section_map: dict[int, Section], limit: int) -> list[dict]:
        tokenized_query = tokenize_text(query)
        bm25_scores = {}
        for id in section_map:
            score = 0
            for token in tokenized_query:
                score += self._bm25(id, token, section_map)
            bm25_scores[id] = score
        sorted_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for id, score in sorted_scores[:limit]:
            project = project_map[id]
            section = section_map[id]
            content = format_section_content(section)
            results.append(
                {
                    "project": project.name,
                    "url": project.repo_url,
                    "id": section.id,
                    "label": section.label,
                    "content": content,
                    "type": section.type,
                    "score": score
                }
            )
        return results
    
    def build(self, projects: list[Project]) -> None:
        for project in projects:
            for section in project.sections:
                if section.type == "code":
                    continue
                content = format_section_content(section)
                self._add_section(section.id, content)

    def save(self) -> None:
        os.makedirs(CACHE, exist_ok=True)
        with open(self.__index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.__token_frequencies, "wb") as f:
            pickle.dump(self.token_frequencies, f)
        with open(self.__section_lengths, "wb") as f:
            pickle.dump(self.section_lengths, f)

    def load(self) -> None:
        with open(self.__index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.__token_frequencies, "rb") as f:
            self.token_frequencies = pickle.load(f)
        with open(self.__section_lengths, "rb") as f:
            self.section_lengths = pickle.load(f)