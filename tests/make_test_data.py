"""Script to generate fake testdata for the telcell project."""

import random
import datetime
from datetime import timedelta, timezone
import pandas as pd

startloc = (52.0907, 5.1214)
startdate = datetime.datetime(2023, 5, 17, 14, 16, 00, tzinfo=timezone.utc)

id = 0
ids = []
track = []
sensor = []
timestamp = []
lon = []
lat = []
degrees = []

# track 1
for i in range(50):
    ids.append(id)
    id += 1
    track.append("TelA")
    sensor.append("A1")
    timestamp.append((startdate + timedelta(minutes=i)).isoformat(" "))
    lat.append(startloc[0] + (i*0.001))
    lon.append(startloc[1] + (i*0.001))
    degrees.append(0)

# track 2
for i in range(50):
    ids.append(id)
    id += 1
    track.append("TelA")
    sensor.append("A2")
    timestamp.append((startdate + timedelta(minutes=i)).isoformat(" "))
    lat.append(startloc[0] + (i*0.001) + 0.001*random.random())
    lon.append(startloc[1] + (i*0.001) + 0.001*random.random())
    degrees.append(0)

# track 3
for i in range(50):
    ids.append(id)
    id += 1
    track.append("TelB")
    sensor.append("B1")
    timestamp.append((startdate + timedelta(minutes=i)).isoformat(" "))
    lat.append(startloc[0] - (i * 0.001))
    lon.append(startloc[1] - (i * 0.001))
    degrees.append(0)

df = pd.DataFrame({
    'id': ids,
    'track': track,
    'sensor': sensor,
    'timestamp': timestamp,
    'celldb.wgs84.lon': lon,
    'celldb.wgs84.lat': lat,
    'celldb.azimuth_degrees': degrees})

df.to_csv('testdata.csv', index=False)
