from semantic_search import SemanticSearch
from test_hybrid_search import test_projects

import pytest
import numpy as np
from numpy import float32


@pytest.fixture
def semantic_search():
    ss = SemanticSearch()
    ss.build(test_projects)
    return ss

def test_build(semantic_search: SemanticSearch):
    assert isinstance(semantic_search.chunk_embeddings[0][0], float32), "Embeddings are 32 bit"
    assert semantic_search.chunk_metadata[0]["total_chunks"] == 1, "Chunk metadata"

def test_search_chunks(semantic_search: SemanticSearch):
    project_map = {}
    section_map = {}
    for project in test_projects:
            for section in project.sections:
                project_map[section.id] = project
                section_map[section.id] = section

    results = semantic_search.search_chunks("fake", project_map, section_map, 2)

    assert len(results) == 2, "Length of results set by limit"
    assert results[0]["project"] == "First project", "Semantic match between fake and mock"
    assert results[0]["score"] > 0, "Match returns positive score"

@pytest.mark.parametrize("text, expected", [
    ("First sentence. Second sentence! Third?", ["First sentence.", "Second sentence!", "Third?"]),
    ("Single sentence, no period", ["Single sentence, no period"]),
    ("   Lots of whitespace.  To be removed? ", ["Lots of whitespace.", "To be removed?"]),
    ("  ! ... ? ..", ["!", "...", "?", ".."]),
    ("", [])
])
def test_split_sentences(text, expected, semantic_search: SemanticSearch):
    assert semantic_search._split_sentences(text) == expected

def test_semantic_chunk(semantic_search: SemanticSearch):
    sentences = "First. Second. Third! Fourth? Fith! Sixth! Seventh?"
    word = "word"
    assert semantic_search._semantic_chunk(sentences, 3, 1) == ["First. Second. Third!", "Third! Fourth? Fith!", "Fith! Sixth! Seventh?"]
    assert semantic_search._semantic_chunk(sentences, 4, 1) == ["First. Second. Third! Fourth?", "Fourth? Fith! Sixth! Seventh?"]
    assert semantic_search._semantic_chunk(sentences, 4, 2) == ["First. Second. Third! Fourth?", "Third! Fourth? Fith! Sixth!", "Fith! Sixth! Seventh?"]
    assert semantic_search._semantic_chunk(word, 4, 1) == ["word"]
    
def test_cosine_similarity(semantic_search: SemanticSearch):
    vec1 = np.array([1, 2, 3], dtype=float32)
    vec2 = np.array([4, 5, 6], dtype=float32)
    assert semantic_search._cosine_similarity(vec1, vec2) == 0.9746318
    vec1 = np.array([0, 0, 0], dtype=float32)
    assert semantic_search._cosine_similarity(vec1, vec2) == 0