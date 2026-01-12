from keyword_search import KeywordSearch
from utils import Project, Section

import pytest


@pytest.fixture
def keyword_search() -> KeywordSearch:
    ks = KeywordSearch()

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

    ks.build(projects)
    return ks


def test_build(keyword_search: KeywordSearch):
    assert keyword_search.project_map[1]["name"] == "First project", "Project name data"
    assert keyword_search.project_map[3]["url"] == "http://repo2.com", "Project repo data"
    assert keyword_search.index["mock"] == {1}, "Mock keyword in section 1 only"
    assert len(keyword_search.token_frequencies) == 3, "Token frequencies for 3 sections"
    assert len(keyword_search.section_lengths) == 3, "Token lengths for 3 sections"

def test_bm25_search(keyword_search: KeywordSearch):
    results = keyword_search.bm25_search("mock", limit=2)
    
    assert len(results) == 2, "Length of results set by limit"
    assert results[0]["project"] == "First project", "keyword returns best result"
    assert results[0]["score"] > 0, "Match returns positive score"
    assert results[1]["score"] == 0, "No match returns zero"
    