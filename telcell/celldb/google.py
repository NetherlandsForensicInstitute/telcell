import datetime
from typing import Optional, Iterable

import geopy
import requests
import requests_cache

from telcell.cell_identity import (
    CellIdentity,
    CellGlobalIdentity,
    NRCell,
    EutranCellGlobalIdentity,
    Radio,
)
from telcell.celldb import CellCollection
from telcell.celldb.cell_collection import Properties


def _ci_to_dict(cell: CellIdentity) -> dict[str, str | int]:
    tower = {
        "mobileCountryCode": cell.mcc,
        "mobileNetworkCode": cell.mnc,
    }
    if isinstance(cell, CellGlobalIdentity):
        tower["locationAreaCode"] = cell.lac
    if isinstance(cell, NRCell):
        tower["newRadioCellId"] = cell.eci
    elif isinstance(cell, CellGlobalIdentity):
        tower["cellId"] = cell.ci
    elif isinstance(cell, EutranCellGlobalIdentity):
        tower["cellId"] = cell.eci

    return tower


class GoogleGeolocationService(CellCollection):
    """
    Google geolocation service.

    See: https://developers.google.com/maps/documentation/geolocation/overview
    """

    def __init__(self, key: str, user_agent: str = "TestApp", cache_name: str = None):
        self._url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={key}"
        self.headers = {"User-Agent": user_agent, "Content-Type": "application/json"}
        if cache_name is None:
            self._session = requests.Session()
        else:
            self._session = requests_cache.CachedSession(cache_name)

    def get(self, date: datetime.datetime, cell: CellIdentity) -> Properties:
        if cell.radio is None and isinstance(cell, EutranCellGlobalIdentity):
            info = self.get(date, CellIdentity.parse(f"{Radio.LTE.value}/{cell}"))
            if info is None:
                info = self.get(date, CellIdentity.parse(f"{Radio.NR.value}/{cell}"))
            return info

        data = {}
        if cell.radio:
            data["radioType"] = cell.radio

        data["cellTowers"] = [_ci_to_dict(cell)]

        res = self._session.post(self._url, json=data).json()
        if "error" in res:
            raise ValueError(
                f'google location service: {res["error"]["message"]}; input={data}'
            )

        point = geopy.Point(
            latitude=res["location"]["lat"], longitude=res["location"]["lng"]
        )
        accuracy = res["accuracy"]

        return Properties(cell=cell, wgs84=point, accuracy=accuracy)

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
        raise NotImplementedError(
            "google geolocation service does not support this feature"
        )

    def __iter__(self):
        raise NotImplementedError(
            "google geolocation service does not support this feature"
        )

    def __len__(self):
        raise NotImplementedError(
            "google geolocation service does not support this feature"
        )
