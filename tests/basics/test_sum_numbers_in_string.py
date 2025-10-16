from src.basics.sum_numbers_in_string.sum_numbers_in_string import sum_numbers_in_string

def test_regular_case():
    assert sum_numbers_in_string("I have 10 apples and 5 oranges") == 15

def test_no_numbers():
    assert sum_numbers_in_string("No numbers here") == 0

def test_only_numbers():
    assert sum_numbers_in_string("123 and 456 make 579") == 1158

def test_numbers_with_letters():
    assert sum_numbers_in_string("1a2b3c4d") == 10

def test_decimal_numbers():
    assert sum_numbers_in_string("Price is $19 and $29") == 48

def test_empty_string():
    assert sum_numbers_in_string("") == 0

def test_large_numbers():
    assert sum_numbers_in_string("1000000 2000000 3000000") == 6000000

def test_numbers_at_start_and_end():
    assert sum_numbers_in_string("123 start and end 456") == 579

def test_multiple_spaces():
    assert sum_numbers_in_string("1  2   3    4") == 10

def test_consecutive_digits():
    assert sum_numbers_in_string("abc123def456ghi") == 579

def test_single_digit():
    assert sum_numbers_in_string("a1b2c3") == 6

def test_numbers_with_special_chars():
    assert sum_numbers_in_string("Number#1! Number@2? Number%3;") == 6

def test_unicode_characters():
    assert sum_numbers_in_string("Привет 123 мир 456") == 579

def test_only_spaces():
    assert sum_numbers_in_string("     ") == 0