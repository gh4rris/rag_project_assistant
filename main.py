from config import RESULT_LIMIT
from src.hybrid_search import HybridSearch
from src.augmented_generation import generate_answer

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

    evaluate_parser = subparser.add_parser("evaluate", help="Evaluate retrival using precision, recall & f1 score metrics")
    evaluate_parser.add_argument("-k", type=int, default=RESULT_LIMIT, nargs="?", help="Number of results to evaluate precision@k and recall@k")

    args = parser.parse_args()

    match args.command:
        case "ask":
            results = hybrid_search.rrf_search(args.question)
            generate_answer(args.question, results)
        case "evaluate":
            evaluations = hybrid_search.evaluate(args.k)

            for question, result in evaluations.items():
                print(f"Question: {question}")
                print(f"Precision@{args.k}: {result["precision"]:.2f}")
                print(f"Recall@{args.k}: {result["recall"]:.2f}")
                print(f"F1 Score: {result["f1_score"]:.2f}")
                print(f"Retrieved: {result["retrieved_ids"]}")
                print(f"Cross Encoder Scores: {result["retrieved_scores"]}")
                print(f"Relevant: {result["relevant"]}\n")
        case _:
            parser.print_help()
    


if __name__ == "__main__":
    main()
