"""
Script to generate fake measurements-data for the telcell project.
This script creates two collocated 'paths' for each device (name).
"""

import random
import datetime
from datetime import timedelta, timezone, datetime
import itertools
import numpy as np
import pandas as pd

random.seed(1234)
startdate = datetime(2023, 5, 17, 12, 00, 00, tzinfo=timezone.utc)

# Setting that controls the length and width of the grid
grid_length = 36
# Create equally spaced WGS-84 coordinates on the grid within the range of the original data
lon_coords = np.linspace(start=3.60848669870189, stop=6.97261243364462, num=grid_length)
lat_coords = np.linspace(start=50.7885407786764, stop=53.2155513728185, num=grid_length)

dfs = []
for name in ["Bas", "Tim", "Stijn", "Daan", "Koen", "Martijn", "Pieter", "Maarten", "Henk", "Hans"]:
    # Random starting position on the grid (not exceeding the grid)
    lon_idx = random.randint(0, grid_length-1)
    lat_idx = random.randint(0, grid_length-1)

    owners = []
    devices = []
    timestamp = []
    cell_identifier = []
    lon = []
    lat = []
    degrees = []
    lon_idxs = []
    lat_idxs = []
    date = startdate

    for day in range(5):
        for measurement in range(20):
            owners.append(name)
            devices.append(name + "1")

            date += timedelta(minutes=random.randint(1, 30))
            timestamp.append(date.isoformat(" "))

            cell_identifier.append("123-4-5678-9012")

            # Randomly choose between going Up/Down, Left/Right on the grid. Staying neutral has a higher chance.
            movements = [[(0,0)], list(itertools.product([-2, -1, 1, 2], [-2, -1, 1, 2]))]
            neutral = 0.5 # chance of staying on the same position
            weights = [[neutral], 16*[(1-neutral)/16]]
            dx, dy = random.choices(list(itertools.chain(*movements)), weights=list(itertools.chain(*weights)))[0]

            # Movements cannot exceed the grid boundaries
            lon_idx = np.clip(lon_idx + dx, 0, grid_length-1)
            lat_idx = np.clip(lat_idx + dy, 0, grid_length-1)

            lon.append(lon_coords[lon_idx])
            lat.append(lat_coords[lat_idx])
            # TODO: Add degrees functionality
            degrees.append(0)

            lon_idxs.append(lon_idx)
            lat_idxs.append(lat_idx)

        date = startdate + timedelta(days=day+1, minutes=random.randint(0, 10))

    df1 = pd.DataFrame({
        'owner': owners,
        'device': devices,
        'timestamp': timestamp,
        'celldb.wgs84.lon': lon,
        'celldb.wgs84.lat': lat,
        'celldb.azimuth_degrees': degrees,
        'lon_index': lon_idxs,
        'lat_index': lat_idxs})

    # For the next device of this owner, we randomly add variations to the timestamp and location
    df2 = df1.copy()
    df2["device"] = name + "2"

    df2["timestamp"] = df2["timestamp"].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S%z") +
                                                         timedelta(minutes=random.randint(0, 10))).isoformat(" "))

    df2["lon_index"] = df2["lon_index"].apply(
        lambda x: np.clip(x + random.choices([0, 1, -1], weights=[0.7, 0.15, 0.15])[0], 0, grid_length-1))
    df2["lat_index"] = df2["lat_index"].apply(
        lambda x: np.clip(x + random.choices([0, 1, -1], weights=[0.7, 0.15, 0.15])[0], 0, grid_length-1))

    df2["celldb.wgs84.lon"] = df2["lon_index"].apply(lambda x: lon_coords[x])
    df2["celldb.wgs84.lat"] = df2["lat_index"].apply(lambda x: lat_coords[x])

    dfs.extend([df1, df2])

output_df = pd.concat(dfs).reset_index(drop=True).drop(["lon_index", "lat_index"], axis=1)
output_df.to_csv("test_measurements.csv", index_label="id")
