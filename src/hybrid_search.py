from config import RRF_K, RESULT_LIMIT, CACHE, LIMIT_MULTIPLYER, THRESHOLD_RANGE, CROSS_ENCODER_MODEL
from src.utils import load_projects, load_golden_dataset, Project, Section
from src.keyword_search import KeywordSearch
from src.semantic_search import SemanticSearch

import os
import pickle
import streamlit as st
from sentence_transformers import CrossEncoder


class HybridSearch:
    project_map: dict[int, Project]
    section_map: dict[int, Section]

    def __init__(self, cache=CACHE) -> None:
        self.project_map = {} # mapping Section IDs to Project object
        self.section_map = {} # mapping Section IDs to Section object
        self.keyword_search = KeywordSearch(cache)
        self.semantic_search = SemanticSearch(cache)

        self._cache = cache
        self._project_map_path = os.path.join(cache, "project_map.pkl")

    def _rrf_score(self, rank: int, k: int=RRF_K) -> float:
        return 1 / (k + rank)

    def _combine_rrf(self, bm25_results: list[dict], semantic_results: list[dict]) -> dict[int, dict]:
        rrf_scores = {}

        for rank, result in enumerate(bm25_results, 1):
            id = result["id"]
            rrf_scores[id] = {
                **result,
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": self._rrf_score(rank)
            }
        for rank, result in enumerate(semantic_results, 1):
            id = result["id"]
            if id not in rrf_scores:
                rrf_scores[id] = {
                    **result,
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": self._rrf_score(rank)
                }
            else:
                rrf_scores[id]["semantic_rank"] = rank
                rrf_scores[id]["rrf_score"] += self._rrf_score(rank)

        return rrf_scores
    
    def _rerank_results(self, query: str, results: list[dict], limit: int) -> list[dict]:
        pairs = [[query, f"{result["project"]} - {result["summary"]}\n{result["label"]}:\n{result["content"]}"] for result in results]
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        scores = cross_encoder.predict(pairs)
        scored_results = []
        for result, score in zip(results, scores):
            scored_results.append({**result, "cross_encoder_score": score})
        scored_results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        min_threshold = scored_results[0]["cross_encoder_score"] - THRESHOLD_RANGE
        filtered_results = [result for result in scored_results if result["cross_encoder_score"] > min_threshold]
        return filtered_results[:limit]

    def rrf_search(self, query: str, rerank: bool=True, limit: int=RESULT_LIMIT) -> list[dict]:
        search_limit = limit * LIMIT_MULTIPLYER if rerank else limit
        bm25_results = self.keyword_search.bm25_search(query, self.project_map, self.section_map, search_limit)
        semantic_results = self.semantic_search.search_chunks(query, self.project_map, self.section_map, search_limit)
        rrf_results = self._combine_rrf(bm25_results, semantic_results)
        sorted_rrf = sorted(rrf_results.values(), key=lambda x: x["rrf_score"], reverse=True)
        if rerank:
            return self._rerank_results(query, sorted_rrf, limit)
        return sorted_rrf[:limit]
    
    def build(self) -> None:
        projects = load_projects()
        for project in projects:
            for section in project.sections:
                if section.type == "code":
                    continue
                self.project_map[section.id] = project
                self.section_map[section.id] = section
        self.keyword_search.build(projects)
        self.semantic_search.build(projects)

    def evaluate(self, k: int) -> dict[str, dict]:
        golden_dataset = load_golden_dataset()

        evaluations = {}
        for test_case in golden_dataset:
            results = self.rrf_search(test_case["question"], limit=k)
            result_ids = [result["id"] for result in results]
            result_scores = [f"{result["cross_encoder_score"]:.4f}" for result in results]
            relevant_results = [id for id in result_ids if id in test_case["relevant_sections"]]
            precision = len(relevant_results) / k
            recall = len(relevant_results) / len(test_case["relevant_sections"])
            evaluations[test_case["question"]] = {
                "precision": precision,
                "recall": recall,
                "f1_score": (2 * (precision * recall) / (precision + recall)) if precision + recall > 0 else 0,
                "retrieved_ids": result_ids,
                "retrieved_scores": result_scores,
                "relevant": relevant_results
            }
        return evaluations

    def save(self) -> None:
        os.makedirs(self._cache, exist_ok=True)
        with open(self._project_map_path, "wb") as f:
            pickle.dump(self.project_map, f)
        self.keyword_search.save()
        self.semantic_search.save()

    def load(self) -> None:
        with open(self._project_map_path, "rb") as f:
            self.project_map = pickle.load(f)
        for project in self.project_map.values():
            for section in project.sections:
                if section.type == "code":
                    continue
                self.section_map[section.id] = section
        self.keyword_search.load()
        self.semantic_search.load()
        

@st.cache_resource
def load_or_build_hybrid_search() -> HybridSearch:
    hybrid_search = HybridSearch()
    try:
        hybrid_search.load()
    except FileNotFoundError:
        hybrid_search.build()
        hybrid_search.save()
    return hybrid_search