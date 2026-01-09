from src.inverted_index import InvertedIndex


def main():
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        inverted_index.build()
        inverted_index.save()

    results = inverted_index.bm25_search("Supervised learning project")
    
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
