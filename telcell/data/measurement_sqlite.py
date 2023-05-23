import datetime
from typing import Optional, Iterable, Set

from . import serialization
from .cell_identity import CellIdentity
from .collection_sqlite import SqliteCollection
from .measurement import CellMeasurementSet, CellMeasurement


class SqliteCellMeasurementSet(SqliteCollection, CellMeasurementSet):
    """
    Set of measurements.

    Can be used as an interator of cell measurements. Uses a sqlite backend to store measurements.

    The class contains various methods to reduce the set by certain criteria, or to order the results. These methods
    return a modified instance of the set.
    """

    def __init__(
        self,
        items: Optional[Iterable[CellMeasurement]] = None,
        blacklist_types=None,
        sqlite_args: dict = None,
    ):
        if blacklist_types is None:
            self.blacklist_types = {}
        else:
            self.blacklist_types = blacklist_types
        default_sqlite_args = {
            "table_name": "measurement",
        }
        sqlite_args = (
            default_sqlite_args
            if sqlite_args is None
            else default_sqlite_args | sqlite_args
        )
        super().__init__(
            items=items,
            **sqlite_args,
        )
        CellMeasurementSet.__init__(self)

    @property
    def track_names(self) -> Set[str]:
        return self.get_unique_values("track")

    @property
    def sensor_names(self) -> Set[str]:
        return self.get_unique_values("sensor")

    def get_cells(self) -> Set[CellIdentity]:
        return set(
            CellIdentity.parse(radio=v[0], global_identity=v[1])
            for v in self.get_unique_values("cell.radio", "cell.identifier")
        )

    def create_collection(
        self,
        items: Optional[Iterable[CellMeasurement]] = None,
        sqlite_args: dict = None,
    ) -> CellMeasurementSet:
        return SqliteCellMeasurementSet(items=items, sqlite_args=sqlite_args)

    def serialize_item(self, item: CellMeasurement) -> dict:
        return serialization.serialize_cell_measurement(
            item, blacklist_types=self.blacklist_types
        )

    def deserialize_item(self, item: dict[str, str]) -> CellMeasurement:
        return serialization.deserialize_cell_measurement(item)

    def select_by_timestamp(
        self, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
    ) -> CellMeasurementSet:
        return self.select_by_range(timestamp=(timestamp_from, timestamp_to))

    def select_by_track(self, *track_names: str) -> CellMeasurementSet:
        return self.select_by_values(track=track_names)

    def select_by_sensor(self, *sensor_names: str) -> CellMeasurementSet:
        return self.select_by_values(sensor=sensor_names)

    def select_by_cell(self, *cells: CellIdentity) -> CellMeasurementSet:
        # NOTE: this breaks if the same cell identifier is used in different radio networks (which should not happen anyway)
        values = [cell.unique_identifier for cell in cells]
        return self.select_by_values(**{"cell.identifier": values})
