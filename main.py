from config import MAIN_PAGE, EVALUATION_PAGE

import streamlit as st


def main():
    main_page = st.Page(MAIN_PAGE, title="Main Page")
    evaluation_page = st.Page(EVALUATION_PAGE, title="Evaluation")

    pg = st.navigation([main_page, evaluation_page])

    pg.run()
    
    
if __name__ == "__main__":
    main()
