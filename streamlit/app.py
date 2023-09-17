import streamlit as st
import pandas as pd
import plotly.express as px 
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="GetAround",
    page_icon="🚗",
    layout="wide"
)

DATA_URL = ('get_around_delay_analysis.xlsx')

st.title("GetAround 🚗")

@st.cache_data()
def load_data():
    data = pd.read_excel(DATA_URL)
    return data

df = load_data()

df["delay_at_checkout_in_minutes"] = df["delay_at_checkout_in_minutes"].fillna(0)

st.subheader("What is the average delay at checkout in minutes?")

average_delay = round(df[df["state"] == "ended"]["delay_at_checkout_in_minutes"].mean())

st.write(f"The average delay is {average_delay} minutes.")

avg_delay_by_checkin = df.groupby('checkin_type')['delay_at_checkout_in_minutes'].mean().reset_index()

fig = px.bar(avg_delay_by_checkin, x='checkin_type', y='delay_at_checkout_in_minutes', title='Average Delay at Checkout by Checkin Type')
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

st.subheader("How often are drivers late for the next check-in?")

count = df[df["state"] == "ended"].shape[0]
late_drivers = round((df["delay_at_checkout_in_minutes"] > 0).sum() / count, 2) * 100

st.write(f"{late_drivers}% of drivers are late for the next check-in.")

late_by_checkin = df[df['delay_at_checkout_in_minutes'] > 0].groupby('checkin_type').size().reset_index(name='late_checkin_count')

fig = px.bar(late_by_checkin, x='checkin_type', y='late_checkin_count', title='Count of late check-in by Checkin Type')
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

st.subheader("What are the most common check-in types?")

checkin_types = df.groupby("checkin_type")["checkin_type"].value_counts()

fig = px.pie(checkin_types, values=checkin_types.values, names=checkin_types.index, title="Distribution of Check-in types")
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

st.write("The feature is to implement a minimum delay between two rentals.")

st.subheader("How many rentals would be affected by the feature depending on the threshold and scope we choose?")

df_delta = df[~df['time_delta_with_previous_rental_in_minutes'].isna()]

delay_threshold = st.slider('Delay threshold (minutes)', min_value=0, max_value=300, value=60, key="delay_threshold")

scope_options = ['All cars', 'Connect cars']
selected_scope = st.radio('Scope: ', scope_options, key="selected_scope")

if selected_scope == 'All cars':
    total = df_delta.shape[0]
    filtered_rentals = df_delta[df_delta['time_delta_with_previous_rental_in_minutes'] > delay_threshold]
else:
    total = df_delta[df_delta['checkin_type'] == 'connect'].shape[0]
    filtered_rentals = df_delta[(df_delta['delay_at_checkout_in_minutes'] > delay_threshold) & (df_delta['checkin_type'] == 'connect')]

affected_rentals_count = round((filtered_rentals.shape[0] / total) * 100, 2) 
st.write(f"Percentage of rentals affected by the feature (threshold: {delay_threshold} minutes, scope: {selected_scope}) : {affected_rentals_count}%")


st.subheader("How many problematic cases will it solve depending on the chosen threshold and scope?")

delay_threshold = st.slider('Delay threshold (minutes)', min_value=0, max_value=300, value=60, key="delay_threshold2")

scope_options = ['All cars', 'Connect cars']
selected_scope = st.radio('Scope: ', scope_options, key="selected_scope2")

canceled_df = df_delta[df_delta["state"] == "canceled"]

if selected_scope == 'All cars':
    total = canceled_df.shape[0]
    filtered_cases = canceled_df[canceled_df["time_delta_with_previous_rental_in_minutes"] < delay_threshold]
else:
    total = canceled_df[canceled_df['checkin_type'] == 'connect'].shape[0]
    filtered_cases = canceled_df[(canceled_df["time_delta_with_previous_rental_in_minutes"] < delay_threshold) & (canceled_df['checkin_type'] == 'connect')]

affected_rentals_count = round((filtered_cases.shape[0] / total) * 100)
st.write(f"Percentage of problematic cases solved (threshold: {delay_threshold} minutes, scope: {selected_scope}) : {affected_rentals_count}%")



