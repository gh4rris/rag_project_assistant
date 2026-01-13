from utils import Project, Section
from hybrid_search import HybridSearch

import pytest
from pytest_mock import MockerFixture

test_projects = [
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

@pytest.fixture
def hybrid_search(mocker: MockerFixture):
    hs = HybridSearch()
    mock_load_projects = mocker.patch("hybrid_search.load_projects")
    mock_load_projects.return_value = test_projects
    hs.build()
    return hs

def test_build(hybrid_search: HybridSearch):
    assert len(hybrid_search.project_map) == len(hybrid_search.section_map), "Project map and Section map same length"
    assert hybrid_search.project_map[1].name == "First project", "Section ID maps to project name"
    assert hybrid_search.project_map[3].repo_url == "http://repo2.com", "Section ID maps to project url"
    assert hybrid_search.section_map[4].type == "instructions", "Section ID maps to Section object"