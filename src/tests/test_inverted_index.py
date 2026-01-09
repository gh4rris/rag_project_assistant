from inverted_index import InvertedIndex
from utils import Project, Section

import pytest
from pytest_mock import MockerFixture


@pytest.fixture
def inverted_index_mock(mocker: MockerFixture) -> InvertedIndex:
    inverted_index = InvertedIndex()
    mock_projects = mocker.patch("inverted_index.load_projects")

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
                )
            ]
        )
    ]

    inverted_index.build()
    return inverted_index


def test_build(inverted_index_mock: InvertedIndex):
    assert inverted_index_mock.project_map[1] == 0, "Section ID 1 is Project index 0"
    assert inverted_index_mock.project_map[2] == 0, "Section ID 2 is Project index 0"
    assert inverted_index_mock.project_map[3] == 1, "Section ID 3 is Project index 1"
    assert inverted_index_mock.section_map[2].label == "Second section", "Section mapping"
    assert inverted_index_mock.index["mock"] == {1}, "Mock keyword in section 1 only"
    assert len(inverted_index_mock.term_frequencies) == 2, "Term frequencies for 2 sections"
    assert len(inverted_index_mock.section_lengths) == 2, "Token lengths for 2 sections"

def test_bm25_search(inverted_index_mock: InvertedIndex):
    results = inverted_index_mock.bm25_search("mock", limit=2)
    
    assert len(results) == 2, "Length of results set by limit"
    assert results[0]["project"] == "First project", "keyword returns best result"
    assert results[0]["score"] > 0, "Match returns positive score"
    assert results[1]["score"] == 0, "No match returns zero"
    