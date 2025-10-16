from src.data_structures.pack_boxes.pack_boxes import pack_boxes

def test_basic_case():
    items = [2, 3, 5, 1, 4]
    limit = 6
    expected = [[2, 3], [5, 1], [4]]
    assert pack_boxes(items, limit) == expected

def test_empty_items():
    items = []
    limit = 5
    expected = []
    assert pack_boxes(items, limit) == expected

def test_single_item():
    items = [7]
    limit = 10
    expected = [[7]]
    assert pack_boxes(items, limit) == expected

def test_item_exceeds_limit():
    items = [3, 7, 2]
    limit = 5
    expected = [[3, 2]]
    assert pack_boxes(items, limit) == expected

def test_all_items_fit_in_one_box():
    items = [1, 2, 3]
    limit = 10
    expected = [[1, 2, 3]]
    assert pack_boxes(items, limit) == expected

def test_items_equal_to_limit():
    items = [5, 5, 5]
    limit = 5
    expected = [[5], [5], [5]]
    assert pack_boxes(items, limit) == expected

def test_mixed_weights():
    items = [4, 2, 1, 3, 5, 2, 1]
    limit = 5
    expected = [[4], [2, 1], [3], [5], [2, 1]]
    assert pack_boxes(items, limit) == expected

def test_zero_limit():
    items = [1, 2, 3]
    limit = 0
    expected = []
    assert pack_boxes(items, limit) == expected

def test_zero_weight_items():
    items = [0, 0, 0, 0]
    limit = 5
    expected = [[0, 0, 0, 0]]
    assert pack_boxes(items, limit) == expected

def test_large_numbers():
    items = [100, 200, 300, 400]
    limit = 500
    expected = [[100, 200], [300], [400]]
    assert pack_boxes(items, limit) == expected

def test_all_items_exceed_limit():
    items = [10, 20, 30]
    limit = 5
    expected = []
    assert pack_boxes(items, limit) == expected

def test_first_item_exceeds_limit():
    items = [8, 2, 3]
    limit = 5
    expected = [[2, 3]]
    assert pack_boxes(items, limit) == expected

def test_last_item_exceeds_limit():
    items = [2, 3, 8]
    limit = 5
    expected = [[2, 3]]
    assert pack_boxes(items, limit) == expected