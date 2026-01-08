from inverted_index import InvertedIndex
from utils import Project, Section

from pytest_mock import MockerFixture


def test_build(mocker: MockerFixture):
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
    
    assert inverted_index.project_map[1] == 0, "Section ID 1 is Project index 0"
    assert inverted_index.project_map[2] == 0, "Section ID 2 is Project index 0"
    assert inverted_index.project_map[3] == 1, "Section ID 3 is Project index 1"
    assert inverted_index.section_map[2].label == "Second section", "Section mapping"
    assert inverted_index.index["mock"] == {1}, "Mock keyword in section 1 only"
    assert len(inverted_index.term_frequencies) == 2, "Term frequencies for 3 sections"
    assert len(inverted_index.section_lengths) == 2, "Token lengths for 3 sections"

    