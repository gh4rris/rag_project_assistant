from src.hybrid_search import HybridSearch
# from src.augmented_generation import generate_answer


def main():
    hybrid_search = HybridSearch()
    try:
        hybrid_search.load()
    except FileNotFoundError:
        hybrid_search.build()
        hybrid_search.save()

    evaluations = hybrid_search.evaluate()

    for question, result in evaluations.items():
        print(f"Question: {question}")
        print(f"Precision: {result["precision"]:.2f}")
        print(f"Recall: {result["recall"]:.2f}")
        print(f"F1 Score: {result["f1_score"]:.2f}")
        print(f"Retrieved: {result["retrieved_ids"]}")
        print(f"RRF Scores: {result["retrieved_rrfs"]}")
        print(f"Relevant: {result["relevant"]}\n")
    
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
