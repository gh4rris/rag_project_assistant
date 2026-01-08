from utils import tokenize_text


def test_tokenize_text():
    assert tokenize_text("Tokenize this text!") == ["token", "text"]
    assert tokenize_text("IS THIS  A __ GOOD TEST OR   WHAT?;") == ["good", "test"]
