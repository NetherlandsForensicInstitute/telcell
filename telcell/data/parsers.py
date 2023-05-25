"""
This module contains parser functions that read raw data and parse them into
`Measurement` or `Track` objects.
"""

import csv
import datetime
from itertools import groupby
from pathlib import Path
from typing import List, Union

from telcell.data.models import Measurement, Track


def parse_measurements_csv(path: Union[str, Path]) -> List[Track]:
    """
    Parse a `measurements.csv` file into `Track`s. The following columns are
    expected to be present:

        - track
        - sensor
        - celldb.wgs84.lat
        - celldb.wgs84.lon
        - timestamp

    Any additional columns are stored under the `extra` attribute of each
    resulting `Measurement` object.

    :param path: The path to the `measurements.csv` file that should be parsed
    :return: All `Track`s that were constructed from the data
    """
    tracks = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        # We assume the rows in the csv file are already sorted by:
        #   1. track (person, car etc. Called owner in the further code)
        #   2. sensor (of the device. Called name)
        #   3. timestamp
        # If this is not the case, we first have to call `sorted()` here.
        for (owner, name), group in groupby(
                reader,
                key=lambda row: (row["track"], row["sensor"])
        ):
            # In practice, we might need to consult an external database to
            # retrieve the (lat, lon) coordinates. In this case, they have
            # already been included in the `measurements.csv` input file.
            fmt = "%Y/%m/%d, %H:%M:%S"
            measurements = [
                Measurement(
                    lat=float(row['celldb.wgs84.lat']),
                    lon=float(row['celldb.wgs84.lon']),
                    timestamp=datetime.datetime.strptime(row['timestamp'], fmt),
                    # For now, we just store the entire `row` under `extra`,
                    # even though this leads to some duplicate data.
                    extra=row
                ) for row in group
            ]
            track = Track(owner, name, measurements)
            tracks.append(track)
    return tracks
