#using database with gradio

import os
import pandas as pd
import matplotlib.pyplot as plt

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
PORT = 8080
DB_NAME = "bikeshare"

connection_string = f"sqlite://{DB_USER}:{DB_PASSWORD}@{DB_HOST}?port={PORT}&dbname={DB_NAME}"

def get_count_ride_type():
    df = pd.read_sql(
    """
        SELECT COUNT(ride_id) as n, rideable_type
        FROM rides
        GROUP BY rideable_type
        ORDER BY n DESC
    """,
    con=connection_string
    )
    fig_m, ax = plt.subplots()
    ax.bar(x=df['rideable_type'], height=df['n'])
    ax.set_title("Number of rides by bycycle type")
    ax.set_ylabel("Number of Rides")
    ax.set_xlabel("Bicycle Type")
    return fig_m


def get_most_popular_stations():

    df = pd.read_sql(
        """
    SELECT COUNT(ride_id) as n, MAX(start_station_name) as station
    FROM RIDES
    WHERE start_station_name is NOT NULL
    GROUP BY start_station_id
    ORDER BY n DESC
    LIMIT 5
    """,
    con=connection_string
    )
    fig_m, ax = plt.subplots()
    ax.bar(x=df['station'], height=df['n'])
    ax.set_title("Most popular stations")
    ax.set_ylabel("Number of Rides")
    ax.set_xlabel("Station Name")
    ax.set_xticklabels(
        df['station'], rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.tick_params(axis="x", labelsize=8)
    fig_m.tight_layout()
    return fig_m

#defining the interface
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        bike_type = gr.Plot()
        station = gr.Plot()

    demo.load(get_count_ride_type, inputs=None, outputs=bike_type)
    demo.load(get_most_popular_stations, inputs=None, outputs=station)

demo.launch()

#for huggingface deployment, go to its ui setting and set database config
#many other databases can be used and functions can be defined to utilize them
