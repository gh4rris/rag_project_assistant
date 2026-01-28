# from config import RESULT_LIMIT
from src.hybrid_search import load_or_build_hybrid_search
from src.augmented_generation import generate_answer

# import argparse
import streamlit as st


def main():
    hybrid_search = load_or_build_hybrid_search()

    st.set_page_config(
        page_title="Rag Q&A Project Assistant",
        layout="centered"
    )
    st.title("Rag Q&A Project Assistant")
    st.caption("Ask questions about my portfolio projects.")

    st.subheader("API Key")
    api_key = st.text_input(
        label="Enter your OpenRouter API Key:",
        type="password",
        help="Your key is used for this session only, and not stored."
    )

    if api_key:
        st.session_state["api_key"] = api_key

    st.subheader("Ask a question")
    question = st.text_input(
        label="The assistant will answer any questions about any of the projects worked on.",
        placeholder="How do I run the loan approval app locally?"
    )

    ask_button = st.button("Ask")

    if ask_button:
        if not api_key:
            st.error("Pease enter an API Key.")
            st.stop()
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        with st.spinner("Retrieving relevant context..."):
            results = hybrid_search.rrf_search(question)
        
        with st.spinner("Generating answer..."):
            answer_placeholder = st.empty()
            complete_response = ""
            for token in generate_answer(question,results, st.session_state["api_key"]):
                complete_response += token
                answer_placeholder.markdown(complete_response)
    
    # parser = argparse.ArgumentParser(description="Project Assistant CLI")
    # subparser = parser.add_subparsers(dest="command", help="Available commands")

    # question_parser = subparser.add_parser("ask", help="Ask assistant a question about one of the projects")
    # question_parser.add_argument("question", type=str, help="Question to ask the project assistant")

    # evaluate_parser = subparser.add_parser("evaluate", help="Evaluate retrival using precision, recall & f1 score metrics")
    # evaluate_parser.add_argument("-k", type=int, default=RESULT_LIMIT, nargs="?", help="Number of results to evaluate precision@k and recall@k")

    # args = parser.parse_args()

    # match args.command:
    #     case "ask":
    #         results = hybrid_search.rrf_search(args.question)
    #         generate_answer(args.question, results)
    #     case "evaluate":
    #         evaluations = hybrid_search.evaluate(args.k)

    #         for question, result in evaluations.items():
    #             print(f"Question: {question}")
    #             print(f"Precision@{args.k}: {result["precision"]:.2f}")
    #             print(f"Recall@{args.k}: {result["recall"]:.2f}")
    #             print(f"F1 Score: {result["f1_score"]:.2f}")
    #             print(f"Retrieved: {result["retrieved_ids"]}")
    #             print(f"Cross Encoder Scores: {result["retrieved_scores"]}")
    #             print(f"Relevant: {result["relevant"]}\n")
    #     case _:
    #         parser.print_help()
    


if __name__ == "__main__":
    main()
