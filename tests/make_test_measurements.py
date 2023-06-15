"""Script to generate fake measurements for the telcell project."""

import random
import datetime
from datetime import timedelta, timezone, datetime

import numpy as np
import pandas as pd

startloc = (52.0907, 5.1214)
startdate = datetime(2023, 5, 17, 12, 00, 00, tzinfo=timezone.utc)

id = 0
ids = []
track = []
sensor = []
timestamp = []
cell_identifier = []
lon = []
lat = []
degrees = []
idx = []

date = startdate
grid_length = 36
start_idx = (round(grid_length/2), round(grid_length/2))

lon_range = np.linspace(start=3.60848669870189, stop=6.97261243364462, num=grid_length)
lat_range = np.linspace(start=50.7885407786764, stop=53.2155513728185, num=grid_length)


for d in range(5):
    for i in range(20):
        ids.append(id)
        id += 1

        track.append("Bas")
        sensor.append("Bas1")

        date += timedelta(minutes=random.randint(1,30))
        timestamp.append((date).isoformat(" "))

        cell_identifier.append("00-458-54-415")

        dx = random.choice([0, 1, 2, -1, -2])
        dy = random.choice([0, 1, 2, -1, -2])
        start_idx = (np.clip(start_idx[0] + dx, 0, grid_length-1), np.clip(start_idx[1] + dy, 0, grid_length-1))

        lon.append(lon_range[start_idx[0]])
        lat.append(lat_range[start_idx[1]])

        degrees.append(0)

        idx.append(start_idx)

    date = startdate + timedelta(days=d+1, minutes=random.randint(0,10))


df1 = pd.DataFrame({
    'id': ids,
    'track': track,
    'sensor': sensor,
    'timestamp': timestamp,
    'celldb.wgs84.lon': lon,
    'celldb.wgs84.lat': lat,
    'celldb.azimuth_degrees': degrees,
    'index':idx})




#df.to_csv('testdata.csv', index=False)
print(df1)

df2 = df1.copy()

df2["timestamp"] = df2["timestamp"].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S%z") + timedelta(minutes=random.randint(0,10))).isoformat(" "))
df2["index"]

print(df2)

"2023-05-17 12:17:00+00:00"