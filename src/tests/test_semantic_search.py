from semantic_search import split_sentences, semantic_chunk

import pytest

@pytest.mark.parametrize("text, expected", [
    ("First sentence. Second sentence! Third?", ["First sentence.", "Second sentence!", "Third?"]),
    ("Single sentence, no period", ["Single sentence, no period"]),
    ("   Lots of whitespace.  To be removed? ", ["Lots of whitespace.", "To be removed?"]),
    ("  ! ... ? ..", ["!", "...", "?", ".."]),
    ("", [])
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected

def test_semantic_chunk():
    sentences = "First. Second. Third! Fourth? Fith! Sixth! Seventh?"
    assert semantic_chunk(sentences, 3, 1) == ["First. Second. Third!", "Third! Fourth? Fith!", "Fith! Sixth! Seventh?"]
    assert semantic_chunk(sentences, 4, 1) == ["First. Second. Third! Fourth?", "Fourth? Fith! Sixth! Seventh?"]
    assert semantic_chunk(sentences, 4, 2) == ["First. Second. Third! Fourth?", "Third! Fourth? Fith! Sixth!", "Fith! Sixth! Seventh?"]