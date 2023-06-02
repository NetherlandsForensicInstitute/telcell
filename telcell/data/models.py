from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Mapping, Sequence


@dataclass
class Measurement:
    """
    A single measurement of a device at a certain place and time.

    :param lat: The WGS 84 latitude coordinate
    :param lon: The WGS 84 longitude coordinate
    :param timestamp: The time of registration
    :param extra: Additional metadata related to the source that registered
            this measurement. These could for example inform the accuracy or
            uncertainty of the measured `(lat, lon)` coordinates.
    """
    lat: float
    lon: float
    timestamp: datetime
    extra: Mapping[str, Any]

    def __str__(self):
        return f"<{self.timestamp}: ({self.lat}, {self.lon})>"


@dataclass
class Track:
    """
    A history of measurements for a single device.

    :param owner: The owner of the device. Can be anything with a simcard.
    :param name: The name of the device.
    :param measurements: A series of measurements ordered by timestamp.
    """
    owner: str
    name: str
    measurements: Sequence[Measurement]

    def __len__(self) -> int:
        return len(self.measurements)

    def __iter__(self) -> Iterator[Measurement]:
        return iter(self.measurements)


class MeasurementPair:
    """
    A pair of two measurements. The pair can be made with different criteria,
    for example the time difference between the two measurements. It always
    contains the information from the two measurements it was created from.
    """
    def __init__(self, track_a, track_b):
        self.track_a = track_a
        self.track_b = track_b

        self.time_difference = abs(track_a.timestamp - track_b.timestamp)
        # TODO track/sensor kolomnaam in extra omzetten naar owner/name
        self.is_colocated = track_a.extra['track'] is not None and \
            track_a.extra['track'] == track_b.extra['track']

    def __str__(self):
        return f"<{self.track_a}: ({self.track_b})>"


def is_colocated(track_a: Track, track_b: Track) -> bool:
    """Checks if two tracks are colocated to each other."""
    if track_a is track_b:
        return True

    return track_a.owner is not None and track_a.owner == track_b.owner
