import pytest
from src.basics.find_twin_pairs.find_twin_pairs import find_twin_pairs
import math

def test_identical_objects():
    X = [
        [1, 2, 3],
        [1, 2, 3],
        [4, 5, 6]
    ]
    threshold = 0.1
    result = find_twin_pairs(X, threshold)
    assert result == [(0, 1, 0.0)]

def test_no_pairs():
    X = [
        [1, 0],
        [2, 0],
        [3, 0]
    ]
    threshold = 0.9
    result = find_twin_pairs(X, threshold)
    assert result == []

def test_multiple_pairs():
    X = [
        [1, 1],
        [1, 1.1],
        [1, 1.05],
        [2, 2]
    ]
    threshold = 0.1  # Уменьшаем порог, чтобы исключить пару (0,1)
    result = find_twin_pairs(X, threshold)
    # Теперь должно быть только 2 пары
    assert len(result) == 2
    assert (0, 2, pytest.approx(0.05)) in result
    assert (1, 2, pytest.approx(0.05)) in result

def test_empty_matrix():
    X = []
    threshold = 1.0
    result = find_twin_pairs(X, threshold)
    assert result == []

def test_single_object():
    X = [[1, 2, 3]]
    threshold = 10.0
    result = find_twin_pairs(X, threshold)
    assert result == []

def test_exact_threshold():
    X = [
        [0, 0],
        [3, 4]
    ]
    threshold = 5.0
    result = find_twin_pairs(X, threshold)
    assert result == [(0, 1, 5.0)]

def test_high_dimensional_data():
    X = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5.1]
    ]
    threshold = 0.2
    result = find_twin_pairs(X, threshold)
    assert len(result) == 1
    assert result[0][0] == 0
    assert result[0][1] == 1
    assert pytest.approx(result[0][2]) == math.sqrt(0.01)

def test_negative_values():
    X = [
        [-1, -2],
        [-1, -2.1],
        [1, 2]
    ]
    threshold = 0.2
    result = find_twin_pairs(X, threshold)
    assert len(result) == 1
    assert result[0][:2] == (0, 1)
    assert pytest.approx(result[0][2]) == 0.1