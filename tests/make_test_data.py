"""Script to generate fake testdata for the telcell project."""

import random
import datetime
from datetime import timedelta, timezone
import pandas as pd

random.seed(10)

startloc = (52.0907, 5.1214)
startdate = datetime.datetime(2023, 5, 17, 14, 16, 00, tzinfo=timezone.utc)

# Simple testdata - 50 points per track, 3 tracks
id = 0
ids = []
owners = []
devices = []
timestamp = []
lon = []
lat = []
degrees = []
# track 1
for i in range(50):
    ids.append(id)
    id += 1
    owners.append("TelA")
    devices.append("A1")
    timestamp.append((startdate + timedelta(minutes=i)).isoformat(" "))
    lat.append(startloc[0] + (i*0.001))
    lon.append(startloc[1] + (i*0.001))
    degrees.append(0)

# track 2
for i in range(50):
    ids.append(id)
    id += 1
    owners.append("TelA")
    devices.append("A2")
    timestamp.append((startdate + timedelta(minutes=i)).isoformat(" "))
    lat.append(startloc[0] + (i*0.001) + 0.001*random.random())
    lon.append(startloc[1] + (i*0.001) + 0.001*random.random())
    degrees.append(0)

# track 3
for i in range(50):
    ids.append(id)
    id += 1
    owners.append("TelB")
    devices.append("B1")
    timestamp.append((startdate + timedelta(minutes=i)).isoformat(" "))
    lat.append(startloc[0] - (i * 0.001))
    lon.append(startloc[1] - (i * 0.001))
    degrees.append(0)

df = pd.DataFrame({
    'id': ids,
    'owner': owners,
    'device': devices,
    'timestamp': timestamp,
    'celldb.wgs84.lon': lon,
    'celldb.wgs84.lat': lat,
    'celldb.azimuth_degrees': degrees})

df.to_csv('testdata.csv', index=False)

# extended testdata - 3 days with random time increments
number_of_days = 3
id = 0
ids = []
owners = []
devices = []
timestamp = []
lon = []
lat = []
degrees = []
# track 1
for i in range(number_of_days*1440):
    ids.append(id)
    id += 1
    owners.append("TelA")
    devices.append("A1")
    timestamp.append((startdate + timedelta(
        minutes=i+0.5*(random.random()-.5))).isoformat(" "))
    lat.append(round(startloc[0] + (i*0.001), 4))
    lon.append(round(startloc[1] + (i*0.001), 4))
    degrees.append(0)

# track 2
for i in range(number_of_days*1440):
    ids.append(id)
    id += 1
    owners.append("TelA")
    devices.append("A2")
    timestamp.append((startdate + timedelta(
        minutes=i+0.5*(random.random()-.5))).isoformat(" "))
    lat.append(round(startloc[0] + (i*0.001) + 0.001*random.random(), 4))
    lon.append(round(startloc[1] + (i*0.001) + 0.001*random.random(), 4))
    degrees.append(0)

# track 3
for i in range(number_of_days*1440):
    ids.append(id)
    id += 1
    owners.append("TelB")
    devices.append("B1")
    timestamp.append((startdate + timedelta(
        minutes=i + 0.5 * (random.random() - .5))).isoformat(" "))
    lat.append(round(startloc[0] - (i * 0.001), 4))
    lon.append(round(startloc[1] - (i * 0.001), 4))
    degrees.append(0)

df = pd.DataFrame({
    'id': ids,
    'owner': owners,
    'device': devices,
    'timestamp': timestamp,
    'celldb.wgs84.lon': lon,
    'celldb.wgs84.lat': lat,
    'celldb.azimuth_degrees': degrees})

df.to_csv('testdata_3days.csv', index=False)
