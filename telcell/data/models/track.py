from dataclasses import dataclass
from typing import Sequence

from telcell.data.models.measurement import Measurement


@dataclass
class Track:
    """
    A history of measurements for a single device.

    :param person: The owner of the device
    :param device: The name of the device
    :param measurements: A series of measurements ordered by timestamp
    """
    person: str
    device: str
    measurements: Sequence[Measurement]

    def __len__(self) -> int:
        return len(self.measurements)
