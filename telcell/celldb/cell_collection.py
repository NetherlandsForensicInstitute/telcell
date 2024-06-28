from __future__ import annotations
from abc import abstractmethod
import datetime
from typing import Optional, Iterable, Sized

import geopy

from telcell.cell_identity import CellIdentity


class Properties(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            return Properties()


class CellCollection(Iterable[Properties], Sized):
    """
    Collection of cell towers in a cellular network, that may be queried by cell id (e.g. for geocoding) or by
    coordinates (e.g. reverse geocoding).

    The known fields for cells may vary by database and is generally a subset of:
    - cell: the cell identity
    - wgs84: the geolocation of the cell tower or a connected device
    - accuracy: the accuracy of the location of the connected device in meters
    - azimuth: the orientation of the cell tower in meters
    """

    @abstractmethod
    def get(self, date: datetime.datetime, ci: CellIdentity) -> Optional[Properties]:
        """
        Retrieve a specific antenna from database.

        :param date: Used to select active antennas
        :param ci: The cell identity
        :return: The retrieved antenna or None
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        coords: geopy.Point = None,
        distance_limit_m: float = None,
        distance_lower_limit_m: float = None,
        date: datetime.datetime = None,
        radio: Optional[str | Iterable[str]] = None,
        mcc: int = None,
        mnc: int = None,
        count_limit: Optional[int] = 10000,
        exclude: Optional[CellIdentity] = None,
    ) -> CellCollection:
        """
        Given a Point, find antennas that are in reach from this point sorted by the distance from the grid point.

        :param coords: antenna selection criteria are relative to this point
        :param distance_limit_m: select antennas within this range of `coords`
        :param distance_lower_limit_m: select antennas beyond this range of `coords`
        :param date: select antennas which are valid at this date
        :param radio: select antennas using this radio technology, e.g.: LTE, UMTS, GSM
        :param mcc: select antennas with this MCC
        :param mnc: select antennas with this MNC
        :param count_limit: return at most this number of antennas
        :param exclude: excluded antennas with this `CellIdentity`
        :return: retrieved selected antennas
        """
        raise NotImplementedError

    def limit(self, count_limit: int) -> CellCollection:
        return self.search(None, count_limit=count_limit)
