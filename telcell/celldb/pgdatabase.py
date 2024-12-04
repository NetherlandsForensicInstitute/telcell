import csv
import datetime
import math
import warnings
from typing import Optional, List, Iterable, Callable, Tuple

import geopy

from telcell.cell_identity import (
    Radio,
    CellIdentity,
    CellGlobalIdentity,
    EutranCellGlobalIdentity,
)
from . import duplicate_policy
from .cell_collection import CellCollection, Properties
from ..data.models import rd_to_point, point_to_rd
from ..geography import Angle


def _build_antenna(row: Tuple) -> Properties:
    date_start, date_end, radio, mcc, mnc, lac, ci, eci, rdx, rdy, azimuth_degrees = row
    if radio == Radio.GSM.value or radio == Radio.UMTS.value:
        retrieved_ci = CellIdentity.create(
            radio=radio, mcc=mcc, mnc=mnc, lac=lac, ci=ci
        )
    elif radio == Radio.LTE.value or radio == Radio.NR.value:
        retrieved_ci = CellIdentity.create(radio=radio, mcc=mcc, mnc=mnc, eci=eci)
    elif radio is not None:
        raise ValueError(f"unrecognized radio type: {radio}")
    elif lac is not None and ci <= 0xFFFF:
        retrieved_ci = CellIdentity.create(
            radio=radio, mcc=mcc, mnc=mnc, lac=lac, ci=ci
        )
    else:
        retrieved_ci = CellIdentity.create(radio=radio, mcc=mcc, mnc=mnc, eci=eci)

    coords = rd_to_point(rdx, rdy)
    azimuth = Angle(degrees=azimuth_degrees) if azimuth_degrees is not None else None
    return Properties(wgs84=coords, azimuth=azimuth, cell=retrieved_ci)


def _build_cell_identity_query(ci):
    if not isinstance(ci, CellIdentity):
        raise ValueError(f"ci expected to be CellIdentity; found: {type(ci)}")

    qwhere = []
    qargs = []
    if ci.radio is not None:
        qwhere.append("radio = %s")
        qargs.append(ci.radio)
    if ci.mcc is not None:
        qwhere.append(f"mcc = {ci.mcc}")
    if ci.mnc is not None:
        qwhere.append(f"mnc = {ci.mnc}")

    if isinstance(ci, CellGlobalIdentity):
        if ci.lac is not None:
            qwhere.append(f"lac = {ci.lac}")
        if ci.ci is not None:
            qwhere.append(f"ci = {ci.ci}")

    elif isinstance(ci, EutranCellGlobalIdentity):
        if ci.eci is not None:
            qwhere.append(f"eci = {ci.eci}")

    else:
        raise ValueError(f"unsupported cell type: {type(ci)}")

    if len(qwhere) == 0:
        qwhere.append("TRUE")

    return " AND ".join(qwhere), qargs


class PgCollection(CellCollection):
    def __init__(
        self,
        con,
        on_duplicate: Callable = duplicate_policy.warn,
        _qwhere=None,
        _qargs=None,
        _qorder=None,
        _count_limit: int = None,
    ):
        """
        Initializes a `PgDatabase` object.

        The parameter `on_duplicate` specifies the policy for `get()` to apply when the cell database has two or more
        hits. This may mean that the cell database is inconsistent. The policy is a function that takes a `CellIdentity`
        and a sequence of cell `Properties` as their two arguments and returns the `Properties` of a single cell or
        `None`. Currently, the following policies are available:
        - `celldb.duplicate_policy.drop` (default): returns None
        - `celldb.duplicate_policy.take_first`: returns the properties of the first cell found
        - `celldb.duplicate_policy.warn`: same as `take_first`, and emits a warning
        - `celldb.duplicate_policy.exception`: throws an exception

        :param con: an active postgres connection
        :param on_duplicate: policy when the cell database has two or more hits for the same cell in a call to `get()`.
        :param _qwhere: for private use: add criteria in WHERE clause
        :param _qargs: for private use: add query arguments
        :param _qorder: for private use: add criteria in ORDER clause
        :param _count_limit: for private use: add item limit
        """
        self._con = con
        self._on_duplicate = on_duplicate
        self._qwhere = _qwhere or ["TRUE"]
        self._qargs = _qargs or []
        self._qorder = _qorder or ""
        self._count_limit = _count_limit
        self._cur = None

    def get(self, date: datetime.datetime, ci: CellIdentity) -> Optional[Properties]:
        """
        Retrieve a specific antenna from database.

        :param date: Used to select active antennas
        :param ci: The cell identity
        :return: The retrieved antenna or None
        """
        if isinstance(date, datetime.date):
            date = datetime.datetime.combine(date, datetime.datetime.min.time())

        results = list(self.search(date=date, ci=ci))
        if len(results) == 0:
            return None
        elif len(results) > 1:
            return self._on_duplicate(ci, results)
        else:
            return results[0]

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
        random_order: bool = False,
        exclude: Optional[List[CellIdentity]] = None,
        ci: Optional[CellIdentity] = None,
    ) -> CellCollection:
        qwhere = list(self._qwhere)
        qargs = list(self._qargs)

        if coords is not None and distance_limit_m is None and count_limit is None:
            raise ValueError(
                "coords argument requires either distance_limit or count_limit"
            )

        if distance_limit_m is not None:
            if coords is None:
                raise ValueError("distance_limit argument requires coords")

            x, y = point_to_rd(coords)
            qwhere.append(
                f"ST_DWithin(rd, 'SRID=4326;POINT({x} {y})', {distance_limit_m})"
            )
            if distance_lower_limit_m is not None:
                qwhere.append(
                    f"NOT ST_DWithin(rd, 'SRID=4326;POINT({x} {y})', {distance_lower_limit_m})"
                )

        if date is not None:
            qwhere.append("(date_start is NULL OR %s >= date_start)")
            qwhere.append("(date_end is NULL OR %s < date_end)")
            qargs.extend([date, date])

        if ci is not None:
            add_qwhere, add_qargs = _build_cell_identity_query(ci)
            qwhere.append(add_qwhere)
            qargs.extend(add_qargs)

            assert radio is None, "radio argument makes no sense in combination with ci"
            assert mcc is None, "mcc argument makes no sense in combination with ci"
            assert mnc is None, "mnc argument makes no sense in combination with ci"
        else:
            if radio is not None:
                if isinstance(radio, str):
                    radio = [radio]
                qwhere.append(f"({' OR '.join(['radio = %s'])})")
                qargs.extend(radio)

            if mcc is not None:
                qwhere.append("mcc = %s")
                qargs.append(mcc)
            if mnc is not None:
                qwhere.append("mnc = %s")
                qargs.append(mnc)

        if exclude is not None:
            if isinstance(exclude, CellIdentity):
                exclude = [exclude]
            for addr in exclude:
                add_qwhere, add_qargs = _build_cell_identity_query(addr)
                qwhere.append(f"NOT ({add_qwhere})")
                qargs.extend(add_qargs)

        qorder = self._qorder
        if random_order is not None and random_order:
            qorder = "ORDER BY RANDOM()"
        elif coords is not None:
            x, y = point_to_rd(coords)
            qorder = f"ORDER BY ST_Distance(rd, 'SRID=4326;POINT({x} {y})')"

        count_limit = count_limit if count_limit is not None else self._count_limit

        return PgCollection(
            self._con, self._on_duplicate, qwhere, qargs, qorder, count_limit
        )

    def close(self):
        if self._cur is not None:
            self._cur.close()

    def __enter__(self):
        self._cur = self._con.cursor()
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __iter__(self):
        if self._cur is None:
            self._cur = self._con.cursor()

        q = f"""
            SELECT date_start, date_end, radio, mcc, mnc, lac, ci, eci, ST_X(rd), ST_Y(rd), azimuth
            FROM antenna_light
            WHERE {' AND '.join(qw for qw in self._qwhere)}
            {self._qorder}
        """
        if self._count_limit is not None:
            q += f" LIMIT {self._count_limit}"

        self._cur.execute(q, self._qargs)
        return self

    def __next__(self):
        row = self._cur.fetchone()
        if row is None:
            raise StopIteration

        return _build_antenna(row)

    def __len__(self):
        with self._con.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM antenna_light
                WHERE {' AND '.join(qw for qw in self._qwhere)}
            """,
                self._qargs,
            )

            return cur.fetchone()[0]


def csv_import(con, flo, progress: Callable = lambda x: x):
    """
    Import antenna data into a Postgres database from a CSV file.

    :param con: an open database connection
    :param flo: a file like object pointing to CSV data
    :param progress: an optional progress bar (like tqdm), or `None`
    """
    create_table(con)

    fieldnames = [
        "date_start",
        "date_end",
        "radio",
        "mcc",
        "mnc",
        "lac",
        "ci",
        "eci",
        "lon",
        "lat",
        "azimuth",
    ]
    reader = csv.DictReader(flo, fieldnames=fieldnames)

    with con.cursor() as cur:
        for i, row in enumerate(progress(list(reader))):
            try:
                (
                    date_start,
                    date_end,
                    radio,
                    mcc,
                    mnc,
                    lac,
                    ci,
                    eci,
                    lon,
                    lat,
                    azimuth,
                ) = [row[f] if row[f] != "" else None for f in fieldnames]
                lon, lat = float(lon), float(lat)
                assert math.isfinite(lon), f"invalid number for longitude: {lon}"
                assert math.isfinite(lat), f"invalid number for latitude: {lat}"
                assert ci is not None or eci is not None

                x, y = point_to_rd(geopy.Point(longitude=lon, latitude=lat))
                cur.execute(
                    """
                    INSERT INTO antenna_light(date_start, date_end, radio, mcc, mnc, lac, ci, eci, rd, azimuth)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, 'SRID=4326;POINT('||%s||' '||%s||')', %s)
                """,
                    (
                        date_start,
                        date_end,
                        radio,
                        mcc,
                        mnc,
                        lac,
                        ci,
                        eci,
                        x,
                        y,
                        azimuth,
                    ),
                )
                con.commit()
            except Exception as e:
                warnings.warn(f"import error at line {i+2}: {e}")


def csv_export(con, flo):
    """
    Export antenna data in a Postgres database to a CSV file.

    :param con: an open database connection
    :param flo: a file like object where CSV data will be written
    """

    sql_x = "ST_X(rd)"
    sql_y = "ST_Y(rd)"
    sql_lon = f"""
         5.38720621 + ((
         (5260.52916 * (({sql_x} - 155000) * 10 ^ -5)) +
         (105.94684 * (({sql_x} - 155000) * 10 ^ -5) * (({sql_y} - 463000) * 10 ^ -5)) +
         (2.45656 * (({sql_x} - 155000) * 10 ^ -5) * (({sql_y} - 463000) * 10 ^ -5) ^ 2) +
         (-0.81885 * (({sql_x} - 155000) * 10 ^ -5) ^ 3) +
         (0.05594 * (({sql_x} - 155000) * 10 ^ -5) * (({sql_y} - 463000) * 10 ^ -5) ^ 3) +
         (-0.05607 * (({sql_x} - 155000) * 10 ^ -5) ^ 3 * (({sql_y} - 463000) * 10 ^ -5)) +
         (0.01199 * (({sql_y} - 463000) * 10 ^ -5)) +
         (-0.00256 * (({sql_x} - 155000) * 10 ^ -5) ^ 3 * (({sql_y} - 463000) * 10 ^ -5) ^ 2) +
         (0.00128 * (({sql_x} - 155000) * 10 ^ -5) * (({sql_y} - 463000) * 10 ^ -5) ^ 4) +
         (0.00022 * (({sql_y} - 463000) * 10 ^ -5) ^ 2) +
         (-0.00022 * (({sql_x} - 155000) * 10 ^ -5) ^ 2) +
         (0.00026 * (({sql_x} - 155000) * 10 ^ -5) ^ 5)
         ) / 3600)
    """
    sql_lat = f"""
         52.15517440 + ((
         (3235.65389 * (({sql_y} - 463000) * 10 ^ -5)) +
         (-32.58297 * (({sql_x} - 155000) * 10 ^ -5) ^ 2) +
         (-0.2475 * (({sql_y} - 463000) * 10 ^ -5) ^ 2) +
         (-0.84978 * (({sql_x} - 155000) * 10 ^ -5) ^ 2 * (({sql_y} - 463000) * 10 ^ -5)) +
         (-0.0655 * (({sql_y} - 463000) * 10 ^ -5) ^ 3) +
         (-0.01709 * (({sql_x} - 155000) * 10 ^ -5) ^ 2 * (({sql_y} - 463000) * 10 ^ -5) ^ 2) +
         (-0.00738 * (({sql_x} - 155000) * 10 ^ -5)) +
         (0.0053 * (({sql_x} - 155000) * 10 ^ -5) ^ 4) +
         (-0.00039 * (({sql_x} - 155000) * 10 ^ -5) ^ 2 * (({sql_y} - 463000) * 10 ^ -5) ^ 3) +
         (0.00033 * (({sql_x} - 155000) * 10 ^ -5) ^ 4 * (({sql_y} - 463000) * 10 ^ -5)) +
         (-0.00012 * (({sql_x} - 155000) * 10 ^ -5) * (({sql_y} - 463000) * 10 ^ -5))
         ) / 3600)
    """
    q = f"""
        SELECT date_start, date_end, radio, mcc, mnc, lac, ci, eci, {sql_lon} lon, {sql_lat} lat, azimuth
        FROM antenna_light
    """
    with con.cursor() as cur:
        cur.copy_expert(f"copy ({q}) to stdout with csv header", flo)


def create_table(con):
    tablename = "antenna_light"
    with con.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {tablename}")
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        cur.execute(
            f"""
            CREATE TABLE {tablename} (
                id SERIAL PRIMARY KEY,
                date_start TIMESTAMP WITH TIME ZONE,
                date_end TIMESTAMP WITH TIME ZONE,
                radio VARCHAR(5) NULL,
                mcc INT NOT NULL,
                mnc INT NOT NULL,
                lac INT NULL,
                ci INT NULL,
                eci INT NULL,
                rd GEOMETRY(point,4326) NOT NULL,
                azimuth INT NULL
            )
        """
        )
        cur.execute(f"CREATE INDEX {tablename}_start ON {tablename}(date_start)")
        cur.execute(f"CREATE INDEX {tablename}_end ON {tablename}(date_end)")
        cur.execute(f"CREATE INDEX {tablename}_cgi ON {tablename}(mcc, mnc, lac, ci)")
        cur.execute(f"CREATE INDEX {tablename}_ecgi ON {tablename}(mcc, mnc, eci)")
        cur.execute(f"CREATE INDEX {tablename}_rd ON {tablename} USING GIST(rd)")
