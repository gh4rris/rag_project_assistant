from config import RRF_K, RESULT_LIMIT
from src.utils import load_projects
from src.keyword_search import KeywordSearch
from src.semantic_search import SemanticSearch

class HybridSearch:
    def __init__(self) -> None:
        self.keyword_search = KeywordSearch()
        self.semantic_search = SemanticSearch()

    def __combine_rrf(self, bm25_results: list[dict], semantic_results: list[dict], k: int) -> dict[int, dict]:
        rrf_scores = {}

        for i, result in enumerate(bm25_results, 1):
            id = result["id"]
            rrf_scores[id] = {
                **result,
                "bm25_rank": i,
                "semantic_rank": None,
                "rrf_score": 0
            }
        for i, result in enumerate(semantic_results, 1):
            id = result["id"]
            if id not in rrf_scores:
                rrf_scores[id] = {
                    **result,
                    "bm25_rank": None,
                    "semantic_rank": i,
                    "rrf_score": 0
                }
            else:
                rrf_scores[id]["semantic_rank"] = i
                bm25_rrf = 1 / (k + rrf_scores[id]["bm25_rank"])
                semantic_rrf = 1 / (k + i)
                rrf_scores[id]["rrf_score"] = bm25_rrf + semantic_rrf

    def rrf_search(self, query: str, k: int=RRF_K, limit: int=RESULT_LIMIT) -> list[dict]:
        bm25_results = self.keyword_search.bm25_search(query, limit)
        semantic_results = self.semantic_search.search_chunks(query, limit)
        rrf_results = self.__combine_rrf(bm25_results, semantic_results, k)
        sorted_rrf = sorted(rrf_results.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_rrf[:limit]
    
    def build(self) -> None:
        projects = load_projects()
        self.keyword_search.build(projects)
        self.keyword_search.save()
        self.semantic_search.build(projects)
        self.semantic_search.save()