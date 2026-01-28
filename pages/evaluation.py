from src.hybrid_search import load_or_build_hybrid_search

from config import RESULT_LIMIT

import streamlit as st

st.title("Evaluate model")
st.caption("Calculate precision@k, recall@k and F1 score against golden dataset")

st.set_page_config(
        page_title="Rag Evaluation",
        layout="centered",
    )

with st.spinner("Loading Model..."):
    hybrid_search = load_or_build_hybrid_search()

k = st.slider(
    label="K Parmeter",
    min_value=1,
    max_value=10,
    value=RESULT_LIMIT
    )

evaluate_button = st.button("Evaluate")

if evaluate_button:
    with st.spinner("Evaluating..."):
        evaluations = hybrid_search.evaluate(k)
    for question, result in evaluations.items():
        st.write(f"Question: {question}")
        st.write(f"Precision@{k}: {result["precision"]:.2f}")
        st.write(f"Recall@{k}: {result["recall"]:.2f}")
        st.write(f"F1 Score: {result["f1_score"]:.2f}")
        st.write(f"Retrieved: {result["retrieved_ids"]}")
        st.write(f"Cross Encoder Scores: {result["retrieved_scores"]}")
        st.write(f"Relevant: {result["relevant"]}")
        st.write("")