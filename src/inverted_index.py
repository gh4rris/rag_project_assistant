from src.utils import Project, Section, load_projects, tokenize_text, format_section_content
from config import BM25_K1, BM25_B, RESULT_LIMIT

import math
from collections import defaultdict, Counter


class InvertedIndex:
    index: defaultdict[str, set[int]]
    project_map: dict[int, int]
    section_map: dict[int, Section]
    term_frequencies: defaultdict[int, Counter]
    section_lengths: dict[int, int]

    def __init__(self) -> None:
        self.index = defaultdict(set) # mapping tokens to sets of Section IDs
        self.project_map = {} # mapping Section IDs to their Project index
        self.section_map = {} # mapping Section IDs to Section object
        self.term_frequencies = defaultdict(Counter) # mapping Section IDs to token Counter
        self.section_lengths = {} # mapping Section IDs to their token length

    def __add_section(self, id: int, text: str) -> None:
        tokenized_text = tokenize_text(text)
        for token in set(tokenized_text):
            self.index[token].add(id)
        self.term_frequencies[id].update(tokenized_text)
        self.section_lengths[id] = len(tokenized_text)

    def __avg_section_length(self) -> float:
        if not self.section_lengths:
            return 0.0
        return sum(self.section_lengths.values()) / len(self.section_lengths)
    
    def __get_bm25_tf(self, id: int, token: str, k1: float=BM25_K1, b: float=BM25_B) -> float:
        section_length = self.section_lengths.get(id, 0)
        avg_section_length = self.__avg_section_length()
        length_norm = (1 - b) + (b * (section_length / avg_section_length)) if avg_section_length > 0 else 1
        tf = self.term_frequencies.get(id, Counter())[token]
        return (tf * (k1 + 1)) / (tf + (k1 * length_norm))
    
    def __get_bm25_idf(self, token: str) -> float:
        matches = self.index[token]
        return math.log((len(self.section_map) - len(matches) + 0.5) / (len(matches) + 0.5) + 1)
    
    def __bm25(self, id: int, token: str) -> float:
        tf = self.__get_bm25_tf(id, token)
        idf = self.__get_bm25_idf(token)
        return tf * idf
    
    def bm25_search(self, query: str, limit: int=RESULT_LIMIT) -> list[dict]:
        tokenized_query = tokenize_text(query)
        bm25_scores = {}
        for id in self.section_map:
            score = 0
            for token in tokenized_query:
                score += self.__bm25(id, token)
            bm25_scores[id] = score
        sorted_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        projects = load_projects()
        results = []
        for id, score in sorted_scores[:limit]:
            project = projects[self.project_map[id]]
            section = self.section_map[id]
            results.append(
                {
                    "project": project.name,
                    "url": project.repo_url,
                    "id": section.id,
                    "label": section.label,
                    "content": section.content,
                    "type": section.type,
                    "score": score
                }
            )
        return results
    
    def build(self) -> None:
        projects = load_projects()
        for i, project in enumerate(projects):
            for section in project.sections:
                self.project_map[section.id] = i
                self.section_map[section.id] = section
                if section.type == "code":
                    continue
                content = format_section_content(section)
                self.__add_section(section.id, content)
