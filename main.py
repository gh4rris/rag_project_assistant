from src.hybrid_search import HybridSearch
from src.augmented_generation import generate_answer


def main():
    hybrid_search = HybridSearch()
    try:
        hybrid_search.load()
    except FileNotFoundError:
        hybrid_search.build()
        hybrid_search.save()

    # question1 = "What's the maximum size profile image that can be uploaded?"
    question2 = "What's the API instruction to create a post on a users profile?"
    results = hybrid_search.rrf_search(question2)
    answer = generate_answer(question2, results)
    print(answer)
    
    # for result in results:
    #     print(f"Project: {result["project"]}")
    #     print(f"Repo URL: {result["url"]}")
    #     print(f"Section ID: {result["id"]}")
    #     print(f"Label: {result["label"]}")
    #     print(f"Content: {result["content"][:120]}...")
    #     print(f"Type: {result["type"]}")
    #     print(f"BM25 Rank: {result["bm25_rank"]}")
    #     print(f"Semantic Rank: {result["semantic_rank"]}")
    #     print(f"RRF Score: {result["rrf_score"]:.2f}\n")


if __name__ == "__main__":
    main()
