from src.data_structures.segment_sort.segment_sort import sort_segments

def test_normal_case():
    segments = [(1, 5), (2, 3), (10, 20)]
    expected = [(10, 20), (1, 5), (2, 3)]
    assert sort_segments(segments) == expected

def test_equal_length():
    segments = [(1, 3), (5, 7), (10, 12)]
    expected = [(1, 3), (5, 7), (10, 12)]  # порядок может быть любой, так как длины равны
    result = sort_segments(segments)
    assert len(result) == 3
    assert all((b - a) == 2 for a, b in result)

def test_single_segment():
    segments = [(0, 100)]
    expected = [(0, 100)]
    assert sort_segments(segments) == expected

def test_empty_list():
    segments = []
    expected = []
    assert sort_segments(segments) == expected

def test_negative_coordinates():
    segments = [(-5, -1), (-10, 0), (1, 2)]
    expected = [(-10, 0), (-5, -1), (1, 2)]
    assert sort_segments(segments) == expected

def test_float_coordinates():
    segments = [(1.5, 3.5), (0.1, 0.2), (10.0, 20.0)]
    expected = [(10.0, 20.0), (1.5, 3.5), (0.1, 0.2)]
    assert sort_segments(segments) == expected

def test_duplicate_segments():
    segments = [(1, 2), (1, 2), (5, 10), (5, 10)]
    expected = [(5, 10), (5, 10), (1, 2), (1, 2)]
    assert sort_segments(segments) == expected