import streamlit as st

from data_analysis.utils import load_data_for_travel_speed, \
    get_travel_speed_data, prepare_df_for_travel_speed, plot_travel_speeds


class TravelspeedDashboard:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def app(self):
        st.title('Travel speed dashboard to investigate abnormally fast movements')
        df = load_data_for_travel_speed(self.file_name) # TODO: move function from utils into class? but self is now unhashable
        with st.sidebar:
            selected_sources = st.multiselect("Select device-owner:", df['device_owner'].unique(), default=df['device_owner'].unique())
            df = df.loc[df['device_owner'].isin(selected_sources)]
            df = prepare_df_for_travel_speed(df)
            max_speed_km_h = st.slider(label="Maximum speed (km/h)",
                                       value=130, min_value=0, max_value=500, step=2)
            st.write(f"Max speed (m/s): {round(max_speed_km_h * 1000 / 3600, 2)}")

            # We want a 'distance tolerance' to remove the effect that you physically
            # stay at one position, but change antennas. This way, you only keep the
            # effect of physical movement.
            distance_tolerance = st.slider(label='Distance tolerance (m)',
                                           value=2000, min_value=0, max_value=15000, step=100)

        st.write("""Check the displacement in space and time for all subsequent 
            registrations of the same owner. We consider the location of the antenna of 
            a registration and the location of the previous antenna registration, and 
            the time passed between the registrations. We set the maximum allowable
            speed by default at 130 km/h. Furthermore, a 'tolerance' on 
            the distance can be applied. This tolerance is subtracted from the 
            distance, to compensate for the fact that there is a distance between the 
            location of the telephone and the location of the registered antenna. Using
            the tolerance parameter will alter the speed.""")

        st.write(f"Chosen file: {self.file_name}")
        st.write("The dataframe below shows subsequent registrations of the same owner "
                 f"with a speed greater than the maximum speed (currently set at {max_speed_km_h} km/h).")

        df_above_max_speed, all_travel_speeds = get_travel_speed_data(df, distance_tolerance, max_speed_km_h)
        fig_travel_speed = plot_travel_speeds(all_travel_speeds, max_speed_km_h)
        st.dataframe(df_above_max_speed)
        st.pyplot(fig_travel_speed)


