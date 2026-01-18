from config import TEST_CACHE
from hybrid_search import HybridSearch
from mock_projects import mock_projects

import pytest
from pytest_mock import MockerFixture


@pytest.fixture
def hybrid_search(mocker: MockerFixture):
    hs = HybridSearch(TEST_CACHE)
    try:
        hs.load()
    except FileNotFoundError:
        mock_load_projects = mocker.patch("hybrid_search.load_projects")
        mock_load_projects.return_value = mock_projects
        hs.build()
        hs.save()
    return hs

def test_build(hybrid_search: HybridSearch):
    assert len(hybrid_search.project_map) == len(hybrid_search.section_map), "Project map and Section map same length"
    assert hybrid_search.project_map[1].name == "First project", "Section ID maps to project name"
    assert hybrid_search.project_map[4].repo_url == "http://repo2.com", "Section ID maps to project url"
    assert hybrid_search.section_map[2].type == "list", "Section ID maps to Section object"