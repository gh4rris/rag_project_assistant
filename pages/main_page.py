from src.hybrid_search import load_or_build_hybrid_search
from src.augmented_generation import generate_answer

import streamlit as st


st.set_page_config(
        page_title="Rag Q&A Project Assistant",
        layout="centered"
    )
st.title("Rag Q&A Project Assistant")
st.caption("Ask questions about my portfolio projects.")

st.sidebar.title("Projects")
st.sidebar.subheader("rag_project_assistant")
st.sidebar.write("A Retrieval-Augmented Generation assistant that answers questions about my portfolio projects")
st.sidebar.subheader("loan_approval_ml")
st.sidebar.write("A supervised machine learning application that predicts loan approval likelihood")
st.sidebar.subheader("ar-united")
st.sidebar.write("A full-stack web application hub for animal rights groups and activists to network")
st.sidebar.subheader("vegan_youtube")
st.sidebar.write("Video data scraping & engagement analysis supervised learning project")

st.sidebar.title("Sample Questions:")
st.sidebar.write("How is the RAG project evaluated?")
st.sidebar.write("What's the maximum size profile image that can be uploaded to ar-united?")
st.sidebar.write("How do I run the loan approval app locally?")
st.sidebar.write("What are the tech stacks for the youtube analysis project and loan approval project?")
st.sidebar.write("What's the API instruction to create a post on a users profile on the networking site?")

with st.spinner("Loading Model..."):
    hybrid_search = load_or_build_hybrid_search()

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
        for token in generate_answer(question, results, st.session_state["api_key"]):
            complete_response += token
            answer_placeholder.markdown(complete_response)