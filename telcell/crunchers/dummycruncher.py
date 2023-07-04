import datetime
from itertools import pairwise
from typing import Any, Iterable, Iterator, Mapping, Tuple

from telcell.data.models import Track
from telcell.data.utils import extract_intervals, split_track_by_interval


def dummy_cruncher(tracks: Iterable[Track]) \
        -> Iterator[Tuple[Track, Track, Mapping[str, Any]]]:
    """
    Dummy data crunchers that takes each consecutive pair of tracks `(track_a,
    track_b)` and splits `track_a` into slices for each day. Yields a 3-tuple
    for each such slice that contains the following:
        - A `Track` consisting of the `track_a` measurements for that day;
        - A reference to `track_b`;
        - A mapping with a `Track` containing all other `track_a` measurements.
    """

    for track_a, track_b in pairwise(tracks):
        # For our `start` we use 5:00 AM on the day before the start of our
        # measurements.
        earliest = next(iter(track_a)).timestamp
        start = datetime.datetime.combine(
            earliest.date() - datetime.timedelta(days=1),
            datetime.time(5, tzinfo=earliest.tzinfo),
        )

        # Find all intervals of an hour represented in the data.
        intervals = extract_intervals(
            timestamps=(m.timestamp for m in track_a),
            start=start,
            duration=datetime.timedelta(hours=1)
        )

        for start, end in intervals:
            single_day, other = split_track_by_interval(track_a, start, end)
            yield single_day, track_b, {"background": other,
                                        "interval": (start, end)}
