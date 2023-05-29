import datetime
from typing import Iterable, Iterator, List, Tuple

from telcell.data.models import Track


def filter_tracks_by_owner(
        tracks: Iterable[Track],
        owner: str
) -> Iterator[Track]:
    for track in tracks:
        if track.owner == owner:
            yield track


def split_track_by_interval(
        track: Track,
        start: datetime.datetime,
        end: datetime.datetime
) -> Tuple[Track, Track]:
    selected_measurements = []
    remaining_measurements = []

    for measurement in track:
        if start <= measurement.timestamp < end:
            selected_measurements.append(measurement)
        else:
            remaining_measurements.append(measurement)

    selected = Track(track.owner, track.name, selected_measurements)
    remaining = Track(track.owner, track.name, remaining_measurements)
    return selected, remaining


def extract_intervals(
        timestamps: Iterable[datetime.datetime],
        start: datetime.datetime,
        duration: datetime.timedelta,
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Calculates, for a set of intervals and timestamps, which intervals have at
    least one timestamp.

    The first interval is defined by `start` and `duration`. The next interval
    starts adjacent to (before or after) the first interval, and so on. An
    interval is returned iff there is at least one timestamp in `timestamps`
    that is within the interval.

    :param timestamps: timestamps which determine which intervals are returned
    :param start: the start of the first interval
    :param duration: the duration of the intervals
    :return: a sorted list of intervals for which there is at least 1 timestamp
    """
    intervals = set()
    for ts in timestamps:
        sequence_no = (ts - start) // duration
        interval_start = start + sequence_no * duration
        intervals.add((interval_start, interval_start + duration))

    return sorted(intervals)
