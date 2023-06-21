from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Mapping, Sequence, Tuple
from functools import cached_property


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

    @property
    def latlon(self) -> Tuple[float, float]:
        return (self.lat, self.lon)

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


@dataclass(order=False)
class MeasurementPair:
    """
    A pair of two measurements. The pair can be made with different criteria,
    for example the time difference between the two measurements. It always
    contains the information from the two measurements it was created from.
    """

    measurement_a: Measurement
    measurement_b: Measurement

    @cached_property
    def time_difference(self):
        return abs(self.measurement_a.timestamp - self.measurement_b.timestamp)

    @cached_property
    def is_colocated(self):
        """Track means something else here. Since all info from the row in
        the csv is put in 'extra', this is the raw information that is
        present in the csv file. In that file the column 'track' has to
        exist. In the code this is renamed to 'owner'. So it is not a
        reference to the whole track, but only to the owner of the track."""
        return self.measurement_a.extra['track'] is not None and \
            self.measurement_a.extra['track'] == \
            self.measurement_b.extra['track']

    def __str__(self):
        return f"<{self.measurement_a}, ({self.measurement_b})>"

def is_colocated(track_a: Track, track_b: Track) -> bool:
    """Checks if two tracks are colocated to each other."""
    if track_a is track_b:
        return True

    return track_a.owner is not None and track_a.owner == track_b.owner
