from semantic_search import split_sentences, semantic_chunk, cosine_similarity, SemanticSearch
from utils import Project, Section

import pytest
import numpy as np
from numpy import float32

@pytest.mark.parametrize("text, expected", [
    ("First sentence. Second sentence! Third?", ["First sentence.", "Second sentence!", "Third?"]),
    ("Single sentence, no period", ["Single sentence, no period"]),
    ("   Lots of whitespace.  To be removed? ", ["Lots of whitespace.", "To be removed?"]),
    ("  ! ... ? ..", ["!", "...", "?", ".."]),
    ("", [])
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected

def test_semantic_chunk():
    sentences = "First. Second. Third! Fourth? Fith! Sixth! Seventh?"
    assert semantic_chunk(sentences, 3, 1) == ["First. Second. Third!", "Third! Fourth? Fith!", "Fith! Sixth! Seventh?"]
    assert semantic_chunk(sentences, 4, 1) == ["First. Second. Third! Fourth?", "Fourth? Fith! Sixth! Seventh?"]
    assert semantic_chunk(sentences, 4, 2) == ["First. Second. Third! Fourth?", "Third! Fourth? Fith! Sixth!", "Fith! Sixth! Seventh?"]

def test_cosine_similarity():
    vec1 = np.array([1, 2, 3], dtype=float32)
    vec2 = np.array([4, 5, 6], dtype=float32)
    assert cosine_similarity(vec1, vec2) == 0.9746318
    vec1 = np.array([0, 0, 0], dtype=float32)
    assert cosine_similarity(vec1, vec2) == 0

@pytest.fixture
def semantic_search():
    ss = SemanticSearch()

    projects = [
        Project(
            name="First project",
            repo_url="http://repo1.com",
            sections=[
                Section(
                    id=1,
                    label="First section",
                    content="Mock content 1",
                    type="text"
                ),
                Section(
                    id=2,
                    label="Second section",
                    content=["text 1", "text 2"],
                    type="list"
                )
            ]
        ),
        Project(
            name="Second project",
            repo_url="http://repo2.com",
            sections=[
                Section(
                    id=3,
                    label="First section",
                    content={
                        "my_code": ["git clone repo", "cd project"]
                    },
                    type="code"
                ),
                Section(
                    id=4,
                    label="Second section",
                    content=[["one", "two"], ["three", "four"]],
                    type="instructions"
                )
            ]
        )
    ]

    ss.build(projects)
    return ss

def test_build(semantic_search: SemanticSearch):
    assert semantic_search.project_map[1]["name"] == "First project", "Project name data"
    assert semantic_search.project_map[3]["url"] == "http://repo2.com", "Project repo data"
    assert isinstance(semantic_search.chunk_embeddings[0][0], float32), "Embeddings are 32 bit"
    assert semantic_search.chunk_metadata[0]["total_chunks"] == 1, "Chunk metadata"

def test_search_chunks(semantic_search: SemanticSearch):
    results = semantic_search.search_chunks("fake", 2)

    assert len(results) == 2, "Length of results set by limit"
    assert results[0]["project"] == "First project", "Semantic match between fake and mock"
    assert results[0]["score"] > 0, "Match returns positive score"
    