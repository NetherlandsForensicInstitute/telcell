"""This script is used to create a dashboard showing the tracks on a map per day.
Please be aware that we consider a 'day' as ranging from 5.00AM to 5.00AM the
next day. Additionally, we set all timestamps to the UTC timezone. This means
that in the dashboard, we may see the first registrations on a day at e.g.
3:15AM+00:00, which is correct, since this timestamp was originally at
5:15AM+02:00 (or +01:00, either way, Dutch timezone). Pairs were chosen within
a day interval ranging from 5:00AM in Dutch timezone to 5:00AM Dutch timezone
the next day."""
import random
from datetime import timedelta

import matplotlib
import pydeck as pdk
import streamlit as st

from data_analysis.utils import get_layers, get_tracks_pairs_from_csv, \
    load_measurements_to_df, get_colormap, get_html_color_legend, \
    get_switches_and_rarest_pairs, find_date_range


class TrackDashboard:
    def __init__(self, file_name: str, max_delay: int):
        self.file_name = file_name
        self.max_delay = max_delay  # in seconds
        random.seed(123)

    def app(self):
        # read the data
        registrations_df = load_measurements_to_df(self.file_name)
        track_pairs = get_tracks_pairs_from_csv(self.file_name)
        registration_pairs = get_switches_and_rarest_pairs(track_pairs, self.max_delay)

        # Assign a color to each device/owner combination
        colormap = get_colormap(['tab10', 'Set3', 'Dark2', 'tab20b'])
        colors = {device_owner: colormap[i] for i, device_owner in enumerate(sorted(registrations_df['device_owner'].unique()))}
        registrations_df['color'] = registrations_df['device_owner'].map(colors)
        registration_pairs['color_a'] = registration_pairs['device_owner_a'].map(colors)
        registration_pairs['color_b'] = registration_pairs['device_owner_b'].map(colors)

        # find time span, s.t. all device-owner combinations have registrations in this range
        date_range = find_date_range(registrations_df)

        # set up dashboard
        with st.sidebar:
            # Selections
            selected_sources = st.multiselect("Select device - owner combinations:", registrations_df['device_owner'].unique(),
                                              default=registrations_df['device_owner'].unique())

            if 'day' not in st.session_state:
                st.session_state['day'] = min(date_range)
            disable_next = False
            if st.session_state['day'] == max(date_range):
                disable_next = True
            if st.button('Next day', disabled=disable_next):
                st.session_state['day'] += timedelta(days=1)

            day = st.date_input('Select day:',
                                min_value=min(date_range), max_value=max(date_range),
                                key='day', value=min(date_range))

            min_value, max_value = registrations_df[registrations_df['day'] == day]['timestamp'].agg(['min', 'max']).dt.to_pydatetime()
            ts_min, ts_max = st.slider('Select timestamp:', min_value=min_value, max_value=max_value, value=(min_value, max_value),
                                       step=timedelta(minutes=15), format="HH:mm")
            show_timestamps = st.checkbox('Show timestamp labels', value=True)
            highlight_selected_pair = st.checkbox('Show selected pair', value=True)

            # Filter dataframes
            registrations_df = registrations_df.loc[
                (registrations_df['device_owner'].isin(selected_sources)) & (registrations_df['day'] == day) & (
                    registrations_df['timestamp'].between(ts_min, ts_max))].reset_index(drop=True)
            registration_pairs = registration_pairs.loc[registration_pairs['interval'] == day].reset_index(drop=True)
            device_owners_count = registrations_df.groupby(['device_owner']).count().to_dict()['id']

            # Select pair to show
            pair_idx = st.selectbox("Select pair:",
                                    options=[idx if not registration_pairs.iloc[idx]['rarest_pair'] else f'{idx}=rarest_pair' for idx
                                             in registration_pairs.index], index=0)
            if not isinstance(pair_idx, int) and pair_idx is not None:
                pair_idx = int(pair_idx.split('=')[0])

            # Create pair layers
            chosen_pair = registration_pairs[registration_pairs.index == pair_idx] if pair_idx is not None else None
            layers = get_layers(registrations_df, chosen_pair, show_timestamps, highlight_selected_pair)

        with st.container():
            st.title(f'Tracks dashboard to analyse tracks and registration pairs')
            st.write(f'File name: {self.file_name}')
            st.write(f'### Chosen day and time: {day} {ts_min.hour}:{ts_min.minute:02d} - '
                     f'{ts_max.hour}:{ts_max.minute:02d}')

            for i, (device_name, n) in enumerate(device_owners_count.items()):
                color = matplotlib.colors.rgb2hex([float(c) / 255 for c in colors.get(device_name)])
                legend = get_html_color_legend(i, color, device_name, n)
                st.markdown(legend, unsafe_allow_html=True)

            st.write("### Pair:")
            st.write(registration_pairs[
                         registration_pairs.index == pair_idx] if pair_idx is not None else "No pair found on this day.")

            st.pydeck_chart(pdk.Deck(tooltip={'text': '{lat} {lon} {time}'}, map_style=None,
                                     initial_view_state=pdk.ViewState(latitude=52.55, longitude=6.1, zoom=7), layers=layers))

            with st.expander("See registrations in order of timestamp. "):
                st.write(registrations_df[['device_owner', 'lat', 'lon', 'timestamp']].sort_values(
                    by=['timestamp']))

            with st.expander(f"See all pairs in order of timestamp, {len(registration_pairs)} pairs."):
                st.write(registration_pairs)

            pairs_within_max_time = registration_pairs.loc[registration_pairs['location_count'].notnull()]
            with st.expander(
                    f"See pairs within 2 min time difference (if any) in order of rarest location, {len(pairs_within_max_time)} pairs."):
                st.write(pairs_within_max_time.sort_values(by=['location_count', 'time_diff']))

