from keyword_search import KeywordSearch
from utils import Project, Section

import pytest
from pytest_mock import MockerFixture


@pytest.fixture
def keyword_search_mock(mocker: MockerFixture) -> KeywordSearch:
    keyword_search = KeywordSearch()
    mock_projects = mocker.patch("keyword_search.load_projects")

    mock_projects.return_value = [
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

    keyword_search.build()
    return keyword_search


def test_build(keyword_search_mock: KeywordSearch):
    assert keyword_search_mock.project_map[1] == 0, "Section ID 1 is Project index 0"
    assert keyword_search_mock.project_map[2] == 0, "Section ID 2 is Project index 0"
    assert keyword_search_mock.project_map[3] == 1, "Section ID 3 is Project index 1"
    assert keyword_search_mock.project_map[4] == 1, "Section ID 4 is Project index 1"
    assert keyword_search_mock.section_map[2].label == "Second section", "Section mapping"
    assert keyword_search_mock.index["mock"] == {1}, "Mock keyword in section 1 only"
    assert len(keyword_search_mock.token_frequencies) == 3, "Token frequencies for 3 sections"
    assert len(keyword_search_mock.section_lengths) == 3, "Token lengths for 3 sections"

def test_bm25_search(keyword_search_mock: KeywordSearch):
    results = keyword_search_mock.bm25_search("mock", limit=2)
    
    assert len(results) == 2, "Length of results set by limit"
    assert results[0]["project"] == "First project", "keyword returns best result"
    assert results[0]["score"] > 0, "Match returns positive score"
    assert results[1]["score"] == 0, "No match returns zero"
    