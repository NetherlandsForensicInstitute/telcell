import random
from collections import Counter
from datetime import datetime, time, timedelta
from itertools import chain
from typing import Tuple, List, Mapping, Any, Optional

import matplotlib
import pandas as pd
import pydeck as pdk
import pyproj
import streamlit as st
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydeck.types import String
from pyproj import Transformer
from telcell.data.models import Track
from telcell.data.parsers import parse_measurements_csv
from telcell.utils.transform import get_switches, \
    sort_pairs_based_on_rarest_location, slice_track_pairs_to_intervals, \
    create_track_pairs

rd_to_wgs84 = Transformer.from_crs(crs_from="EPSG:28992", crs_to="EPSG:4326")
GEOD = pyproj.Geod(ellps='WGS84')
PAIR_COLUMNS = ["interval", "lat_a", "lon_a", "timestamp_a", "device_owner_a",
                "lat_b", "lon_b", "timestamp_b", "device_owner_b", "distance",
                "time_diff", "rarest_pair", "location_count"]

# functions for travel speed
@st.cache_data
def load_data_for_travel_speed(file_name: str) -> pd.DataFrame:
    """
    Load the csv data for the travel speed dashboard and format timestamp column.
    :param: file_name: name of the file to load registrations from.
    :return: a dataframe of registrations, with timestamp set to UTC timezone.
    """
    df = pd.read_csv(file_name)
    df['timestamp'] = df['timestamp'].apply(lambda x: pd.to_datetime(x, utc=True))
    df['device_owner'] = df['device'].astype(str) + "_" + df['owner'].astype(str)
    return df


def prepare_df_for_travel_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create columns with previous timestamps and locations.
    :param: dataframe containing all registrations.
    :return: dataframe of subsequent registrations of the same device
    """
    df = df.sort_values(by=['owner', 'timestamp'])
    df[['prev_timestamp', 'prev_lat', 'prev_lon', 'prev_device', 'prev_owner',
        'prev_device_owner', 'prev_cell']] = df[
        ['timestamp', 'cellinfo.wgs84.lat', 'cellinfo.wgs84.lon', 'device',
         'owner', 'device_owner', 'cell']].shift(1)
    df = df.iloc[1:]
    df = df.loc[df['owner'] == df[
        'prev_owner']]  # only interested in subsequent registrations of the same owner
    return df


def get_travel_speed_data(df: pd.DataFrame, tolerance: int,
                          max_speed_km_h: int) -> Tuple[pd.DataFrame, List[float]]:
    """
    Validate 'travel speed': the displacement in space and time for all
    subsequent registrations. We consider the location of the antenna of a
    registration and the location of the previous antenna registration, and the
    time passed between the registrations.
    :param: df: dataframe containing subsequent registrations
    :param: tolerance: tolerance is subtracted from the distance, to compensate
                      for the fact that there is a distance between the location
                      of the telephone and the location of the registered antenna
    :param: max_speed_km_h: any speeds above max_speed_km_h are displayed in the
                            histogram at max_speed_km_h
    :return: dataframe with registrations above the max speed and a list of all
                travel speeds, clipped to maximum speed.
    """
    above_max_speed_rows, travel_speeds = [], []
    for _, row in df.iterrows():
        time_diff = row['timestamp'] - row['prev_timestamp']
        time_diff = time_diff.total_seconds()
        distance = round(calculate_distance_lat_lon(
            (row['cellinfo.wgs84.lat'], row['cellinfo.wgs84.lon']),
            (row['prev_lat'], row['prev_lon'])) - tolerance, 2)

        if distance > 0:
            travel_speed_km_h = distance / time_diff * 3600 / 1000 if time_diff > 0 else float(
                'inf')
            travel_speeds.append(
                travel_speed_km_h if travel_speed_km_h < max_speed_km_h else max_speed_km_h)
            if travel_speed_km_h > max_speed_km_h:
                row['speed'] = round(travel_speed_km_h,
                                     2) if travel_speed_km_h < float(
                    'inf') else 999999
                row['distance'] = distance
                row['time_diff'] = time_diff
                above_max_speed_rows.append(row)

    df_above_max = pd.DataFrame(above_max_speed_rows,
                                columns=df.columns.append(pd.Index(
                                    ['speed', 'distance', 'time_diff'])))
    df_above_max = df_above_max[
        ['device_owner', 'timestamp', 'cell', 'prev_device_owner', 'prev_timestamp', 'prev_cell', 'distance', 'time_diff', 'speed']]
    df_above_max = df_above_max.rename(
        columns={"distance": "distance (m)", "time_diff": "time_diff (s)",
                 "speed": "speed (km/h)"})

    return df_above_max, travel_speeds


def plot_travel_speeds(travel_speeds: List[float], max_speed: float) -> Figure:
    """
    Plot the travel speeds clipped at the 'max_speed' in a histogram.
    :param travel_speeds: list of the travel speeds in km/h to plot.
    :param max_speed: maximum speed in km/h that the speeds are clipped at.
    :return: a histogram with travel speeds in km/h.
    """
    nr_above_max = sum(speed == max_speed for speed in travel_speeds)
    fig, ax = plt.subplots()
    plt.hist(travel_speeds, bins=20)
    fig.suptitle(
        f"Travel speeds between subsequent registrations \n Above max: {nr_above_max}")
    ax.set_xlabel(
        f"Speed in kilometers per hour (clipped at {round(max_speed, 2)} km/h)")
    ax.set_ylabel("Count")
    return fig


# general functions
def calculate_distance_lat_lon(latlon_a: Tuple[float, float],
                               latlon_b: Tuple[float, float]) -> float:
    """
    Calculate the distance (in meters) between a set of lat-lon coordinates.

    :param latlon_a: the latitude and longitude of the first object
    :param latlon_b: the latitude and longitude of the second object
    :return: the calculated distance in meters
    """
    lat_a, lon_a = latlon_a
    lat_b, lon_b = latlon_b
    _, _, distance = GEOD.inv(lon_a, lat_a, lon_b, lat_b)
    return distance


def map_ts_to_day(timestamp: pd.Timestamp | str) -> datetime.date:
    """
    Instead of considering a 'day' as ranging from 0.00-23.59, we will now
    view it as 5.00-4.59. This function is to map a timestamp to a day, since
    we cannot do "timestamp.date()" anymore.
    :param timestamp: an object containing date and time information.
    :return: the day (from 5.00AM to 4.59AM) to which the timestamp is mapped.
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    if timestamp.time() < time(5):
        return timestamp.date() - timedelta(days=1)
    else:
        return timestamp.date()


def find_date_range(registrations_df: pd.DataFrame) -> List[datetime.date]:
    """
    Find the date range where all device-owner combinations have registrations.
    :param registrations_df: A dataframe containing all registrations of all device-owner combinations
    :return: sequence of days between the minumum and maximum dates found.
    """
    overlapping_days = set.intersection(
        *[set(registrations_df.loc[
                  registrations_df['device_owner'] == device]['day'])
          for device in
          registrations_df['device_owner'].unique()])
    min_date, max_date = min(overlapping_days), max(overlapping_days)
    date_range = [day.date() for day in
                  pd.date_range(min_date, max_date)]
    return date_range

# functions for tracks and pairs dashboard
@st.cache_data
def load_measurements_to_df(file_name: str) -> pd.DataFrame:
    """
    Load the measurements.csv data and format the columns.
    :param file_name: file to load measurements from.
    :return: the measurements in a dataframe.
    """
    df = pd.read_csv(file_name)
    df = df.rename(columns={'cellinfo.wgs84.lat': 'lat', 'cellinfo.wgs84.lon': 'lon'})
    df['day'] = df['timestamp'].apply(map_ts_to_day)
    df['timestamp'] = df['timestamp'].apply(lambda x: pd.to_datetime(x, utc=True))
    df['time'] = df['timestamp'].apply(lambda x: x.time())
    df.drop_duplicates(subset=['device', 'owner', 'timestamp', 'lat', 'lon'], inplace=True, ignore_index=True)
    df['device_owner'] = df['device'].astype(str) + "_" + df['owner'].astype(str)
    return df


@st.cache_data
def get_tracks_pairs_from_csv(file_name: str) -> List[Tuple[Track, Track, Mapping[str, Any]]]:
    """
    Loads and parses the file and creates a list of track pairs.
    :param file_name: file to load.
    :return: the paired tracks from the file.
    """
    tracks = parse_measurements_csv(file_name)
    track_pairs = list(slice_track_pairs_to_intervals(create_track_pairs(tracks, all_different_source=True), interval_length_h=24))
    return track_pairs


def get_switches_and_rarest_pairs(data: List[Tuple[Track, Track, Mapping[str, Any]]],
                                  max_delay: int = None) -> pd.DataFrame:
    """
    Load all registration pairs from pairs of tracks, store them in a
    dataframe. We also indicate for each day what pair is selected based on
    rarest location and maximum time difference (if any).
    :param data: the paired tracks for each time interval.
    :param max_delay: maximum time difference (seconds) of a registration pair.
                      Default: no max_delay, all pairs are returned.
    :return: a dataframe of paired measurements.
    """
    df = []
    for track_a, track_b, kwargs in tqdm(data):
        switches = get_switches(track_a, track_b)
        sorted_pairs_by_rarity_b = sort_pairs_based_on_rarest_location(
            switches=switches,
            history_track_b=kwargs['background_b'],
            round_lon_lats=False,
            max_delay=max_delay)

        sorted_pairs_by_rarity_b = {pair: count for count, pair in sorted_pairs_by_rarity_b}
        # we want the pair with rarest location, since the dict is sorted, we take the first.
        rarest_pair = list(sorted_pairs_by_rarity_b.keys())[0] if sorted_pairs_by_rarity_b else None
        device_owner_a, device_owner_b = track_a.device + "_" + track_a.owner, track_b.device + "_" + track_b.owner

        df.extend([
            [kwargs['interval'][0],
             *pair.measurement_a.latlon, pair.measurement_a.timestamp, device_owner_a,
             *pair.measurement_b.latlon, pair.measurement_b.timestamp, device_owner_b,
             pair.distance, pair.time_difference.seconds,
             pair == rarest_pair, sorted_pairs_by_rarity_b.get(pair)]
            for pair in switches
        ])

    df = pd.DataFrame(df, columns=PAIR_COLUMNS)
    df['interval'] = df['interval'].apply(
        lambda x: x.date())  # here we can do .date() since here the interval is already from 5:00AM.
    df['timestamp_a'] = df['timestamp_a'].apply(lambda x: pd.to_datetime(x, utc=True))
    df['timestamp_b'] = df['timestamp_b'].apply(lambda x: pd.to_datetime(x, utc=True))
    return df


# functions to create layers for the tracks dashboard
def prepare_data_for_layers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Set the correct columns that can be used to create layers, for example
    to create the counts layer or set the color column.
    :param data: dataframe to process.
    :return: the processed dataframe containing new useful columns.
    """
    data['time'] = data['time'].astype(str)
    # add a little noise so the texts are not stacked
    data['lat_with_noise'] = data['lat'].apply(
        lambda x: x + random.uniform(-0.008, 0.008))
    data['lon_with_noise'] = data['lon'].apply(
        lambda x: x + random.uniform(-0.008, 0.008))
    # counts per unique lat lon value
    data['latlon'] = data['lat'].astype(str) + "-" + data['lon'].astype(
        str)
    data['counts'] = data['latlon'].map(Counter(data['latlon'])).astype(
        str)
    # select only the relevant columns
    data = data[
        ['color', 'time', 'lon_with_noise', 'lat_with_noise', 'lat', 'lon',
         'counts']]
    return data


def get_pair_layer(pair_data: pd.DataFrame) -> Tuple[pdk.Layer, pdk.Layer]:
    """"
    Create two layers to show a pair, one layer per measurement of the
    provided pair.
    :param pair_data: information about the pair to show.
    :return: two layers, each displaying one measurement of a pair.
    """
    pair_data = pair_data[
        ['lat_a', 'lon_a', 'lat_b', 'lon_b', 'color_a', 'color_b']]
    layer_a = pdk.Layer('ScatterplotLayer', data=pair_data, stroked=True,
                        getLineWidth=10,
                        lineWidthMaxPixels=5,
                        get_position=['lon_a', 'lat_a'],
                        get_color=[0, 255, 0],  # green
                        radius_min_pixels=8, radius_max_pixels=18,
                        get_line_color='color_a')
    layer_b = pdk.Layer('ScatterplotLayer', data=pair_data, stroked=True,
                        getLineWidth=10,
                        lineWidthMaxPixels=5,
                        get_position=['lon_b', 'lat_b'],
                        get_color=[0, 255, 0],  # green
                        radius_min_pixels=8, radius_max_pixels=18,
                        get_line_color='color_b')
    return layer_a, layer_b


def get_layers(data: pd.DataFrame, chosen_pair: Optional[pd.DataFrame],
               show_timestamps: bool,
               highlight_selected_pair: bool) -> List[pdk.Layer]:
    """
    Create a text layer with timestamps of the registrations, a scatter layer
    of the positions and a counts layer, showing how many registrations are
    stacked at one position.
    :param data: dataframe containing the registrations to show.
    :param chosen_pair: chosen registration pair to show
    :param show_timestamps: boolean whether to generate the timestamp layer
    :param highlight_selected_pair: boolean whether to highlight the selected pair
    :return: all the layers
    """
    data = prepare_data_for_layers(data)
    # Scatter points
    layers = [pdk.Layer('ScatterplotLayer', data=data, stroked=True,
                            getLineWidth=10,
                            lineWidthMaxPixels=1, get_line_color=[0, 0, 0],
                            get_position=['lon', 'lat'],
                            get_color='color', radius_min_pixels=8,
                            radius_max_pixels=18)]
    # Timestamps
    if show_timestamps:
        layers.append(pdk.Layer(type="TextLayer", data=data, pickable=True,
                                get_position=['lon_with_noise',
                                              'lat_with_noise'],
                                get_text='time', get_size=20,
                                get_color='color',
                                get_text_anchor=String("middle"),
                                get_alignment_baseline=String("bottom")))
    # Highlighted pair
    if not chosen_pair.empty and highlight_selected_pair:
        layers.append(get_pair_layer(chosen_pair))
    # Counts
    layers.append(pdk.Layer(type="TextLayer", data=data, pickable=True,
                            get_position=['lon', 'lat'], get_text='counts',
                            get_size=15, get_color=[0, 0, 0]))
    return layers


# functions for coloring
def get_colormap(colormap_names: List[str]) -> List[List[int]]:
    """
    Generates a list of RGB formatted colors from the matplotlib colormap names
    """
    all_colors = chain.from_iterable(matplotlib.colormaps[c].colors for c in colormap_names)
    return [list(map(lambda x: int(x * 255), c)) for c in all_colors]


def get_html_color_legend(i, color, device_name, n):
    """Generates the color coded legend based on the device name"""
    return """
                <style>
                .dot{} {{
                height: 15px;
                width: 15px;
                background-color: {};
                border-radius: 50%;
                display: inline-block;
                }}

                </style>
                </head>
                <body>
                <div style="text-align:left">
                <span class="dot{}"></span>  {} : {} registrations <br>
                </div>
                </body>
                """.format(i, color, i, device_name, n)


