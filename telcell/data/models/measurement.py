from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


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
