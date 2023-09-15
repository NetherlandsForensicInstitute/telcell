import argparse

import streamlit as st

from data_analysis.multiapp import MultiApp
from data_analysis.applications.travel_speed import TravelspeedDashboard
from data_analysis.applications.tracks_and_pairs import TrackDashboard

st.set_page_config(layout='wide',
                   page_title="Telcell dashboards",
                   page_icon="🗼")

parser = argparse.ArgumentParser()
parser.add_argument('--file-name', '-f',
                    help="Location of the measurement file to analyse",
                    default="scratch/test_measurements.csv")
parser.add_argument('--max-delay', '-d',
                    help="Max time difference (sec) between two registrations "
                         "of a registration pair",
                    default=600)
args = parser.parse_args()


def home_page():
    st.title("Telcell dashboards 🗼")
    st.write("Welcome to the Telcell dashboards application. In the menu on the"
             " left, you can select the dashboard to show. Choose from: \n "
             "- *Tracks and pairs*: to show antenna registrations and pairs of"
             " registrations on a map per day. \n "
             "- *Travel speed*: to analyse abnormally fast travel movements.")


multi_app = MultiApp()
multi_app.add_app("Home page", home_page)
multi_app.add_app("Tracks and pairs",
                  TrackDashboard(args.file_name, args.max_delay).app)
multi_app.add_app("Travel speed", TravelspeedDashboard(args.file_name).app)
multi_app.run()
