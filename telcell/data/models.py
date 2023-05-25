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


def is_colocated(track_a: Track, track_b: Track) -> bool:
    """Checks if two tracks are colocated to each other."""
    if track_a is track_b:
        return True

    return track_a.owner is not None and track_a.owner == track_b.owner
