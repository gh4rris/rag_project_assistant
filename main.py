from src.hybrid_search import HybridSearch
# from src.augmented_generation import generate_answer

import argparse


def main():
    hybrid_search = HybridSearch()
    try:
        hybrid_search.load()
    except FileNotFoundError:
        hybrid_search.build()
        hybrid_search.save()
    
    parser = argparse.ArgumentParser(description="Project Assistant CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    question_parser = subparser.add_parser("ask", help="Ask assistant a question about one of the projects")
    question_parser.add_argument("question", type=str, help="Question to ask the project assistant")

    subparser.add_parser("evaluate", help="Evaluate retrival using precision, recall & f1 score metrics")

    args = parser.parse_args()

    match args.command:
        case "ask":
            results = hybrid_search.rrf_search(args.question)
            for result in results:
                print(f"Project: {result["project"]}")
                print(f"Repo URL: {result["url"]}")
                print(f"Section ID: {result["id"]}")
                print(f"Label: {result["label"]}")
                print(f"Content: {result["content"][:120]}...")
                print(f"Type: {result["type"]}")
                print(f"Cross Encoder Score: {result["cross_encoder_score"]:.2f}\n")
        case "evaluate":
            evaluations = hybrid_search.evaluate()

            total_f1 = 0
            for question, result in evaluations.items():
                print(f"Question: {question}")
                print(f"Precision: {result["precision"]:.2f}")
                print(f"Recall: {result["recall"]:.2f}")
                print(f"F1 Score: {result["f1_score"]:.2f}")
                print(f"Retrieved: {result["retrieved_ids"]}")
                print(f"RRF Scores: {result["retrieved_rrfs"]}")
                print(f"Relevant: {result["relevant"]}\n")
                total_f1 += result["f1_score"]
            print(f"Mean F1: {(total_f1 / len(evaluations.items())):.2f}")
        case _:
            parser.print_help()
    


if __name__ == "__main__":
    main()
