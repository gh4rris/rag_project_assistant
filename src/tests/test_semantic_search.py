from config import TEST_CACHE
from utils import Section
from semantic_search import SemanticSearch
from mock_projects import mock_projects

import pytest
import numpy as np
from typing import cast
from numpy import float32
from numpy.typing import NDArray



semantic_search = SemanticSearch(TEST_CACHE)
try:
    semantic_search.load()
except FileNotFoundError:
    semantic_search.build(mock_projects)
    semantic_search.save()
project_map = {}
section_map = {}
for project in mock_projects:
        for section in project.sections:
            project_map[section.id] = project
            section_map[section.id] = section


def test_build():
    assert isinstance(cast(NDArray[float32], semantic_search.chunk_embeddings)[0][0], float32), "Embeddings are 32 bit"
    assert cast(list[dict], semantic_search.chunk_metadata)[0]["total_chunks"] == 1, "Chunk metadata"

def test_search_chunks():
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
def test_split_sentences(text, expected):
    assert semantic_search._split_sentences(text) == expected


@pytest.mark.parametrize("content, type, chunk_size, overlap, expected", [
    (
        "First. Second. Third! Fourth? Fith! Sixth! Seventh?", "text",
        3, 1,
        ["First. Second. Third!", "Third! Fourth? Fith!", "Fith! Sixth! Seventh?"]
    ),
    (
       "First. Second. Third! Fourth? Fith! Sixth! Seventh?", "text",
       4, 1,
       ["First. Second. Third! Fourth?", "Fourth? Fith! Sixth! Seventh?"]
    ),
    (
        "First. Second. Third! Fourth? Fith! Sixth! Seventh?", "text",
        4, 2,
        ["First. Second. Third! Fourth?", "Third! Fourth? Fith! Sixth!", "Fith! Sixth! Seventh?"]
    ),
    (
        ["text 1", "text 2", "text 3", "text 4"], "list",
        4, 1,
        ["text 1", "text 2", "text 3", "text 4"]
    ),
    (
         [["text 1", "text 1"], ["text 2", "text 2", "text 2"], ["text 3", "text 3"]], "instructions",
         4, 1,
         ["text 1\ntext 1", "text 2\ntext 2\ntext 2", "text 3\ntext 3"]
    )
])
def test_semantic_chunk(content, type, chunk_size, overlap, expected):
    section = Section(id=1, label="Test section", content=content, type=type)
    assert semantic_search._semantic_chunk(section, chunk_size, overlap) == expected
    
def test_cosine_similarity():
    vec1 = np.array([1, 2, 3], dtype=float32)
    vec2 = np.array([4, 5, 6], dtype=float32)
    assert semantic_search._cosine_similarity(vec1, vec2) == 0.9746318
    vec1 = np.array([0, 0, 0], dtype=float32)
    assert semantic_search._cosine_similarity(vec1, vec2) == 0