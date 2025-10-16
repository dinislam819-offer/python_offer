from src.basics.find_anomalous_words.find_anomalous_words import find_anomalous_words

def test_normal_case():
    assert find_anomalous_words("Python is great for data science") == ["is", "science"]

def test_no_anomalous_words():
    assert find_anomalous_words("Hello world") == []

def test_edge_case_single_short_word():
    assert find_anomalous_words("A BB CCC DDDD EEEEE") == ["A", "EEEEE"]

def test_empty_string():
    assert find_anomalous_words("") == []

def test_all_words_same_length():
    assert find_anomalous_words("cat dog pig") == []

def test_with_punctuation():
    assert find_anomalous_words("Hi! My name is Alex. I like antidisestablishmentarianism") == ["Hi", "My", "is", "I", "antidisestablishmentarianism"]

def test_long_words():
    assert find_anomalous_words("short medium loooooong ") == ["loooooong"]

def test_single_word_not_anomalous():
    assert find_anomalous_words("abc") == []

def test_multiple_spaces():
    assert find_anomalous_words("a  bb   ccc    dddd   eeeee") == ["a", "eeeee"]