from src.data_structures.schedule_conflicts.schedule_conflicts import find_schedule_conflicts

def test_no_conflicts():
    intervals = [("09:00", "10:00"), ("10:00", "11:00")]
    assert find_schedule_conflicts(intervals) == []

def test_simple_conflict():
    intervals = [("09:00", "10:00"), ("09:30", "11:00")]
    assert find_schedule_conflicts(intervals) == [(("09:00", "10:00"), ("09:30", "11:00"))]

def test_multiple_conflicts():
    intervals = [("09:00", "10:00"), ("09:30", "11:00"), ("10:30", "12:00")]
    result = find_schedule_conflicts(intervals)
    assert len(result) == 2
    assert (("09:00", "10:00"), ("09:30", "11:00")) in result
    assert (("09:30", "11:00"), ("10:30", "12:00")) in result

def test_edge_case_no_conflict():
    intervals = [("09:00", "10:00"), ("10:00", "11:00"), ("11:00", "12:00")]
    assert find_schedule_conflicts(intervals) == []

def test_full_overlap():
    intervals = [("09:00", "12:00"), ("10:00", "11:00")]
    assert find_schedule_conflicts(intervals) == [(("09:00", "12:00"), ("10:00", "11:00"))]

def test_identical_intervals():
    intervals = [("09:00", "10:00"), ("09:00", "10:00")]
    assert find_schedule_conflicts(intervals) == [(("09:00", "10:00"), ("09:00", "10:00"))]

def test_single_interval():
    intervals = [("09:00", "10:00")]
    assert find_schedule_conflicts(intervals) == []

def test_empty_input():
    intervals = []
    assert find_schedule_conflicts(intervals) == []