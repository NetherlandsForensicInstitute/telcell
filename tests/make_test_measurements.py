"""Script to generate fake measurements-data for the telcell project."""

import random
import datetime
from datetime import timedelta, timezone, datetime

import numpy as np
import pandas as pd

df_list = []

startdate = datetime(2023, 5, 17, 12, 00, 00, tzinfo=timezone.utc)

grid_length = 36
# Create equally spaced WGS-84 coordinates within the range of the original data
lon_range = np.linspace(start=3.60848669870189, stop=6.97261243364462, num=grid_length)
lat_range = np.linspace(start=50.7885407786764, stop=53.2155513728185, num=grid_length)

for name in ["Bas", "Tim", "Stijn", "Daan", "Koen", "Martijn", "Pieter", "Maarten", "Henk", "Hans"]:
    track = []
    sensor = []
    timestamp = []
    cell_identifier = []
    lon = []
    lat = []
    degrees = []
    lon_idxs = []
    lat_idxs = []
    date = startdate
    lon_idx, lat_idx = round(grid_length / 2), round(grid_length / 2)  # Starting position (center)

    for day in range(5):
        for measurement in range(20):
            track.append(name)
            sensor.append(name + "1")

            date += timedelta(minutes=random.randint(1, 30))
            timestamp.append(date.isoformat(" "))

            cell_identifier.append("123-4-5678-9012")

            # Randomly choose between going Up/Down, Left/Right on the grid
            dx = random.choice([0, 1, 2, -1, -2])
            dy = random.choice([0, 1, 2, -1, -2])
            # Movements cannot exceed the grid boundaries
            lon_idx = np.clip(lon_idx + dx, 0, grid_length-1)
            lat_idx = np.clip(lat_idx + dy, 0, grid_length-1)

            lon.append(lon_range[lon_idx])
            lat.append(lat_range[lat_idx])

            degrees.append(0)

            lon_idxs.append(lon_idx)
            lat_idxs.append(lat_idx)

        date = startdate + timedelta(days=day+1, minutes=random.randint(0, 10))

    df1 = pd.DataFrame({
        'track': track,
        'sensor': sensor,
        'timestamp': timestamp,
        'celldb.wgs84.lon': lon,
        'celldb.wgs84.lat': lat,
        'celldb.azimuth_degrees': degrees,
        'lon_index': lon_idxs,
        'lat_index': lat_idxs})

    # For the next sensor on the same track, we randomly add variations to the timestamp and location
    df2 = df1.copy()
    df2["sensor"] = name + "2"

    df2["timestamp"] = df2["timestamp"].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S%z") +
                                                         timedelta(minutes=random.randint(0, 10))).isoformat(" "))

    df2["lon_index"] = df2["lon_index"].apply(
        lambda x: np.clip(x + random.choices([0, 1, -1], weights=[0.7, 0.15, 0.15])[0], 0, grid_length-1))
    df2["lat_index"] = df2["lat_index"].apply(
        lambda x: np.clip(x + random.choices([0, 1, -1], weights=[0.7, 0.15, 0.15])[0], 0, grid_length-1))

    df2["celldb.wgs84.lon"] = df2["lon_index"].apply(lambda x: lon_range[x])
    df2["celldb.wgs84.lat"] = df2["lat_index"].apply(lambda x: lat_range[x])

    df_list.extend([df1, df2])

final_df = pd.concat(df_list).reset_index(drop=True).drop(["lon_index", "lat_index"], axis=1)
final_df.to_csv("test_measurements.csv", index_label="id")
