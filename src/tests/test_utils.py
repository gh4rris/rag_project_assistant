from utils import tokenize_text, format_section_content, Section

import pytest


def test_tokenize_text():
    assert tokenize_text("Tokenize this text!") == ["token", "text"]
    assert tokenize_text("IS THIS  A __ GOOD TEST OR   WHAT?;") == ["good", "test"]

@pytest.mark.parametrize("section, expected", [
    (
        Section(
        id=1,
        label="Test section",
        content="Mock content 1",
        type="text"
        ),
        "Mock content 1"
    ),
    (
        Section(
        id=2,
        label="Test section",
        content=["text 1", "text 2"],
        type="list"
        ),
        "- text 1\n- text 2"
    ),
    (
        Section(
        id=3,
        label="Test section",
        content=[["one", "two"], ["three", "four"]],
        type="instructions"
        ),
        "one\ntwo\n\nthree\nfour"
    )
])
def test_format_section_content(section, expected):
    assert format_section_content(section) == expected
