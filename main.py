from src.keyword_search import KeywordSearch
from src.semantic_search import SemanticSearch


def main():
    keyword_search = KeywordSearch()
    try:
        keyword_search.load()
    except FileNotFoundError:
        keyword_search.build()
        keyword_search.save()

    results = keyword_search.bm25_search("Supervised learning project")
    
    for result in results:
        print(f"Project: {result["project"]}")
        print(f"Repo URL: {result["url"]}")
        print(f"Section ID: {result["id"]}")
        print(f"Label: {result["label"]}")
        print(f"Content: {result["content"][:80]}...")
        print(f"Type: {result["type"]}")
        print(f"Score: {result["score"]:.2f}\n")


if __name__ == "__main__":
    main()
