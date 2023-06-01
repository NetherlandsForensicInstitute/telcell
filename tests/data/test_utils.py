import datetime

from telcell.data.utils import extract_intervals, split_track_by_interval


def test_extract_intervals_by_minute(test_data):
    # Test data ranges from 2023-05-17 14:16:00 to 2023-05-17 15:05:00.
    timestamps = (m.timestamp for track in test_data for m in track)
    start = datetime.datetime.fromisoformat("2023-05-17 14:15:00+00:00")
    duration = datetime.timedelta(minutes=1)
    intervals = extract_intervals(timestamps, start, duration)
    assert len(intervals) == 50
    assert all(end - start == duration for start, end in intervals)


def test_extract_intervals_start_date_after_last_date_in_data(test_data):
    timestamps = (m.timestamp for track in test_data for m in track)
    start = datetime.datetime.fromisoformat("2023-05-20 14:15:00+00:00")
    duration = datetime.timedelta(minutes=1)
    intervals = extract_intervals(timestamps, start, duration)
    assert len(intervals) == 50
    assert all(end - start == duration for start, end in intervals)


def test_extract_intervals_duration_encompasses_all_measurements(test_data):
    timestamps = [m.timestamp for track in test_data for m in track]
    start = datetime.datetime.fromisoformat("2023-05-17 00:00:00+00:00")
    duration = datetime.timedelta(days=1)
    intervals = extract_intervals(timestamps, start, duration)
    assert len(intervals) == 1
    a, b = next(iter(intervals))
    assert a == start
    assert b == start + duration


def test_split_track_by_interval(test_data):
    # Only use the first track from the test data.
    track = test_data[0]
    start = datetime.datetime.fromisoformat("2023-05-17 14:30:00+00:00")
    end = datetime.datetime.fromisoformat("2023-05-17 14:40:00+00:00")
    a, b = split_track_by_interval(track, start, end)
    assert len(a) == 10
    assert len(b) == 40
    assert min(m.timestamp for m in a) == start
    # The `end` in the interval is exclusive.
    assert max(m.timestamp for m in a) == end - datetime.timedelta(minutes=1)
    # There must not be any overlap between `a` and `b`.
    assert not {m.timestamp for m in a}.intersection({m.timestamp for m in b})


def test_split_track_by_interval_all_within(test_data):
    # Only use the first track from the test data.
    track = test_data[0]
    start = datetime.datetime.fromisoformat("2023-05-17 14:00:00+00:00")
    end = datetime.datetime.fromisoformat("2023-05-17 16:00:00+00:00")
    a, b = split_track_by_interval(track, start, end)
    assert a == track
    assert len(b) == 0


def test_split_track_by_interval_all_outside(test_data):
    # Only use the first track from the test data.
    track = test_data[0]
    start = datetime.datetime.fromisoformat("2023-05-18 14:00:00+00:00")
    end = datetime.datetime.fromisoformat("2023-05-18 16:00:00+00:00")
    a, b = split_track_by_interval(track, start, end)
    assert len(a) == 0
    assert b == track
