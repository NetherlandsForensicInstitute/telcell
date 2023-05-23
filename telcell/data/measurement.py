from __future__ import annotations
import datetime
from abc import abstractmethod
from typing import Iterable, List, Sized, Set, Optional

from .cell_identity import CellIdentity
from .item import Item


class CellMeasurement(Item):
    """
    An instance of a cell measurement.

    A cell measurement means that a particular cell is seen. The measurement has a `timestamp`, a `cell`, and optionally
    other values specified during construction.
    """

    def create_item(self, **values) -> Item:
        return CellMeasurement(**values)

    def __init__(
        self, timestamp: datetime.datetime, cell: Optional[CellIdentity], **extra
    ):
        super().__init__(**extra)
        assert isinstance(
            timestamp, datetime.datetime
        ), "timestamp must be `datetime.datetime`"
        assert cell is None or isinstance(
            cell, CellIdentity
        ), "cell_identity must be of class `CellIdentity`"
        self.timestamp = timestamp
        self.cell = cell

    def as_dict(self) -> dict:
        fields = super().as_dict()
        fields.update(
            {
                "timestamp": self.timestamp,
                "cell": self.cell,
            }
        )
        return fields

    def __hash__(self):
        return hash((self.timestamp, self.cell, repr(sorted(self.extra.items()))))

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.timestamp == other.timestamp
            and self.cell == other.cell
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.timestamp}, {self.cell}, {self.extra})"


class CellMeasurementSet(Iterable[CellMeasurement], Sized):
    """
    Set of measurements.

    Can be used as an interator of cell measurements. Uses a sqlite backend to store measurements.

    The class contains various methods to reduce the set by certain criteria, or to order the results. These methods
    return a modified instance of the set.
    """

    @staticmethod
    def create() -> CellMeasurementSet:
        from .measurement_sqlite import SqliteCellMeasurementSet

        return SqliteCellMeasurementSet()

    @staticmethod
    def from_measurements(
        measurements: Iterable[CellMeasurement],
    ) -> CellMeasurementSet:
        from .measurement_sqlite import SqliteCellMeasurementSet

        return SqliteCellMeasurementSet(measurements)

    @abstractmethod
    def add(self, measurement: CellMeasurement):
        raise NotImplementedError

    @property
    @abstractmethod
    def track_names(self) -> List[str]:
        """
        Tracks referenced in this set.

        @return: a list of track names
        """
        raise NotImplementedError

    @abstractmethod
    def get_cells(self) -> Set[CellIdentity]:
        raise NotImplementedError

    @property
    @abstractmethod
    def sensor_names(self) -> List[str]:
        """
        Sensors referenced in this set.

        @return: a list of sensor names
        """
        raise NotImplementedError

    @abstractmethod
    def select_by_timestamp(
        self, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
    ) -> CellMeasurementSet:
        raise NotImplementedError

    @abstractmethod
    def select_by_track(self, *track_names: str) -> CellMeasurementSet:
        """
        Select measurements by their track.

        @param track_names: the names of the selected tracks
        @return: an augmented version of this set with only measurements that refer to one of the selected tracks
        """
        raise NotImplementedError

    @abstractmethod
    def select_by_sensor(self, *sensor_names: str) -> CellMeasurementSet:
        """
        Select measurements by their sensor.

        @param sensor_names: the names of the selected sensors
        @return: an augmented version of this set with only measurements that refer to one of the selected sensors
        """
        raise NotImplementedError

    @abstractmethod
    def select_by_cell(self, *cells: CellIdentity) -> CellMeasurementSet:
        raise NotImplementedError

    @abstractmethod
    def limit(self, count: int) -> CellMeasurementSet:
        """
        Limit the number of results.

        @param count: the maximum number of measurements
        @return: an augmented version of this set with a limited number of measurements
        """
        raise NotImplementedError

    @abstractmethod
    def sort_by(self, key: str) -> CellMeasurementSet:
        """
        Sort pairs by the specified key.

        @param key: an SQL expression to use for sorting
        @return: an augmented version of this set with sorted pairs
        """
        raise NotImplementedError
