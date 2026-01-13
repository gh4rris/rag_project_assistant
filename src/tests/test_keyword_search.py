from keyword_search import KeywordSearch
from test_hybrid_search import test_projects

import pytest


@pytest.fixture
def keyword_search() -> KeywordSearch:
    ks = KeywordSearch()
    ks.build(test_projects)
    return ks


def test_build(keyword_search: KeywordSearch):
    assert keyword_search.index["mock"] == {1}, "Mock keyword in section 1 only"
    assert len(keyword_search.token_frequencies) == 3, "Token frequencies for 3 sections"
    assert len(keyword_search.section_lengths) == 3, "Token lengths for 3 sections"

def test_bm25_search(keyword_search: KeywordSearch):
    project_map = {}
    section_map = {}
    for project in test_projects:
            for section in project.sections:
                project_map[section.id] = project
                section_map[section.id] = section

    results = keyword_search.bm25_search("mock", project_map, section_map, 2)
    
    assert len(results) == 2, "Length of results set by limit"
    assert results[0]["project"] == "First project", "keyword returns best result"
    assert results[0]["score"] > 0, "Match returns positive score"
    assert results[1]["score"] == 0, "No match returns zero"
    