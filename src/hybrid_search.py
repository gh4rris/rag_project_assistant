from config import RRF_K, RESULT_LIMIT, CACHE, LIMIT_MULTIPLYER
from src.utils import load_projects, Project, Section
from src.keyword_search import KeywordSearch
from src.semantic_search import SemanticSearch

import os
import pickle


class HybridSearch:
    project_map: dict[int, Project]
    section_map: dict[int, Section]

    def __init__(self) -> None:
        self.project_map = {} # mapping Section IDs to Project object
        self.section_map = {} # mapping Section IDs to Section object
        self.keyword_search = KeywordSearch()
        self.semantic_search = SemanticSearch()

        self.__project_map_path = os.path.join(CACHE, "project_map.pkl")

    def __rrf_score(self, rank: int, k: int=RRF_K) -> float:
        return 1 / (k + rank)

    def __combine_rrf(self, bm25_results: list[dict], semantic_results: list[dict]) -> dict[int, dict]:
        rrf_scores = {}

        for rank, result in enumerate(bm25_results, 1):
            id = result["id"]
            rrf_scores[id] = {
                **result,
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": self.__rrf_score(rank)
            }
        for rank, result in enumerate(semantic_results, 1):
            id = result["id"]
            if id not in rrf_scores:
                rrf_scores[id] = {
                    **result,
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": self.__rrf_score(rank)
                }
            else:
                rrf_scores[id]["semantic_rank"] = rank
                rrf_scores[id]["rrf_score"] += self.__rrf_score(rank)

        return rrf_scores

    def rrf_search(self, query: str, limit: int=RESULT_LIMIT) -> list[dict]:
        bm25_results = self.keyword_search.bm25_search(query, self.project_map, self.section_map, limit * LIMIT_MULTIPLYER)
        semantic_results = self.semantic_search.search_chunks(query, self.project_map, self.section_map, limit * LIMIT_MULTIPLYER)
        rrf_results = self.__combine_rrf(bm25_results, semantic_results)
        sorted_rrf = sorted(rrf_results.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_rrf[:limit]
    
    def build(self) -> None:
        projects = load_projects()
        for project in projects:
            for section in project.sections:
                self.project_map[section.id] = project
                self.section_map[section.id] = section
        self.keyword_search.build(projects)
        self.semantic_search.build(projects)

    def save(self) -> None:
        os.makedirs(CACHE, exist_ok=True)
        with open(self.__project_map_path, "wb") as f:
            pickle.dump(self.project_map, f)
        self.keyword_search.save()
        self.semantic_search.save()

    def load(self) -> None:
        with open(self.__project_map_path, "rb") as f:
            self.project_map = pickle.load(f)
        for project in self.project_map.values():
            for section in project.sections:
                self.section_map[section.id] = section
        self.keyword_search.load()
        self.semantic_search.load()
        