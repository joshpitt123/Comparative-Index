# import json
# import geopandas as gpd
# import pandas as pd
# import numpy as np
# from unidecode import unidecode
# import plotly.express as px
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State
# import os

# # Variables with '6' for 6 months == csv_path_1 
# # Variables with '3' for 3 months == csv_path_2

# # ──────────────────────────────────────────────────────────────
# # File paths — update to your local files
# # ──────────────────────────────────────────────────────────────
# shapefile_path = 'data/shapefiles/NER_admbnda_adm3_IGNN_20230720.shp'
# csv_path_1    = 'data/Weighted_Vulnerability_Index_Linear.csv'
# csv_path_2    = 'data/Weighted_Vulnerability_Index_Uniform.csv'

# # ──────────────────────────────────────────────────────────────
# # Helper: load & prepare data (with log transform)
# # ──────────────────────────────────────────────────────────────
# def load_data(csv_path):
#     df = pd.read_csv(
#         csv_path,
#         usecols=['Date', 'Commune', 'WeightedIndex'],
#         parse_dates=['Date'],
#         dtype={'Commune':'string','WeightedIndex':'float32'}
#     )
#     df = (
#         df
#         .query("Date >= '2000-01-01' and Date <= '2023-12-31'")
#         .assign(
#             Commune=lambda d: d['Commune'].str.strip().str.upper().map(unidecode)
#         )
#     )
#     df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
#     # Mean index per month
#     agg = df.groupby(['Commune','YearMonth'], sort=False)['WeightedIndex'].mean().reset_index()
#     # full grid
#     months = sorted(agg['YearMonth'].unique())
#     communes = shapefile_gdf['Commune'].unique()
#     grid = pd.MultiIndex.from_product([communes, months], names=['Commune','YearMonth']).to_frame(index=False)
#     full = grid.merge(agg, on=['Commune','YearMonth'], how='left').fillna(0)
#     merged = shapefile_gdf.merge(full, on='Commune', how='left')
#     # apply log1p transform
#     merged['LogIndex'] = np.log1p(merged['WeightedIndex'])
#     # determine log range
#     vmin = merged['LogIndex'].min()
#     vmax = merged['LogIndex'].max()
#     return merged, months, vmin, vmax

# # ──────────────────────────────────────────────────────────────
# # Load & clean shapefile once (geometry simplified)
# # ──────────────────────────────────────────────────────────────
# gdf = gpd.read_file(shapefile_path)[['ADM3_FR','geometry']]
# gdf['Commune'] = gdf['ADM3_FR'].str.strip().str.upper().map(unidecode)
# gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01).buffer(0)
# shapefile_gdf = gdf.dissolve(by='Commune').reset_index()
# geojson = json.loads(shapefile_gdf.to_json())

# # ──────────────────────────────────────────────────────────────
# # Prepare log-index datasets
# # ──────────────────────────────────────────────────────────────
# merged6, months6, vmin6, vmax6 = load_data(csv_path_1)
# merged3, months3, vmin3, vmax3 = load_data(csv_path_2)
# # common timeline
# common_months = [m for m in months6 if m in months3]

# # ──────────────────────────────────────────────────────────────
# # Initialize Dash app
# # ──────────────────────────────────────────────────────────────
# app = Dash(__name__)

# # ──────────────────────────────────────────────────────────────
# # Layout: play/pause, slider, and two log maps
# # ──────────────────────────────────────────────────────────────
# app.layout = html.Div([
#     html.H1('Linear Weighting vs Uniform Weighting - Logarithmic Index', style={'text-align':'center'}),
#     html.Div([
#         html.Button('Play', id='play-btn', n_clicks=0),
#         html.Button('Pause', id='pause-btn', n_clicks=0),
#         dcc.Slider(
#             id='month-slider',
#             min=0, max=len(common_months)-1, value=0,
#             marks={i: m for i, m in enumerate(common_months)},
#             updatemode='drag',
#             tooltip={'always_visible':False, 'placement':'bottom'}
#         ),
#         dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=True)
#     ], style={'width':'80%', 'margin':'auto', 'padding':'20px'}),
#     html.Div([
#         dcc.Graph(id='map-6m', style={'width':'48vw', 'height':'75vh'}),
#         dcc.Graph(id='map-3m', style={'width':'48vw', 'height':'75vh'})
#     ], style={'display':'flex', 'justify-content':'space-around'})
# ])

# # ──────────────────────────────────────────────────────────────
# # Callbacks: control interval, advance slider, update maps
# # ──────────────────────────────────────────────────────────────
# @app.callback(
#     Output('interval', 'disabled'),
#     Input('play-btn', 'n_clicks'),
#     Input('pause-btn', 'n_clicks')
# )
# def toggle_interval(play, pause):
#     return not (play > pause)

# @app.callback(
#     Output('month-slider', 'value'),
#     Input('interval', 'n_intervals'),
#     State('month-slider', 'value')
# )
# def advance_slider(n, current):
#     return (current + 1) % len(common_months)

# @app.callback(
#     [Output('map-6m', 'figure'), Output('map-3m', 'figure')],
#     Input('month-slider', 'value')
# )
# def update_maps(idx):
#     frame = common_months[idx]
#     df6 = merged6[merged6['YearMonth']==frame]
#     df3 = merged3[merged3['YearMonth']==frame]
#     # 6‑month log map
#     fig6 = px.choropleth(
#         df6, geojson=geojson, featureidkey='properties.Commune',
#         locations='Commune', color='LogIndex',
#         range_color=(vmin6, vmax6), color_continuous_scale='YlOrRd',
#         projection='mercator', hover_name='Commune',
#         title=f'Linear weighting – {frame}', width=800, height=800
#     )
#     fig6.update_geos(fitbounds='locations', visible=False)
#     fig6.update_layout(margin={'r':10,'t':40,'l':10,'b':10})
#     fig6.update_coloraxes(colorbar=dict(thickness=8, len=0.5, title='Log Index'))
#     # 3‑month log map
#     fig3 = px.choropleth(
#         df3, geojson=geojson, featureidkey='properties.Commune',
#         locations='Commune', color='LogIndex',
#         range_color=(vmin3, vmax3), color_continuous_scale='YlOrRd',
#         projection='mercator', hover_name='Commune',
#         title=f'Uniform weighting – {frame}', width=800, height=800
#     )
#     fig3.update_geos(fitbounds='locations', visible=False)
#     fig3.update_layout(margin={'r':10,'t':40,'l':10,'b':10})
#     fig3.update_coloraxes(colorbar=dict(thickness=8, len=0.5, title='Log Index'))
#     return fig6, fig3

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 8050))
#     app.run(host="0.0.0.0", port=port, debug=False)












































# # app.py  ───────────────────────────────────────────────────────────
# import json
# import os

# import geopandas as gpd
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from dash import Dash, Patch, dcc, html
# from dash.dependencies import Input, Output, State
# from unidecode import unidecode

# # ──────────────────────────────────────────────────────────────
# # File paths — update if your repo uses different folders
# # ──────────────────────────────────────────────────────────────
# SHAPEFILE = "data/shapefiles/NER_admbnda_adm3_IGNN_20230720.shp"
# CSV_LINEAR = "data/Weighted_Vulnerability_Index_Linear.csv"   # 6-month
# CSV_UNIFORM = "data/Weighted_Vulnerability_Index_Uniform.csv" # 3-month

# # ──────────────────────────────────────────────────────────────
# # Load & prepare shapefile (simplified geometry)
# # ──────────────────────────────────────────────────────────────
# gdf = gpd.read_file(SHAPEFILE)[["ADM3_FR", "geometry"]]
# gdf["Commune"] = gdf["ADM3_FR"].str.strip().str.upper().map(unidecode)
# gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01).buffer(0)
# shapefile_gdf = gdf.dissolve(by="Commune").reset_index()
# geojson = json.loads(shapefile_gdf.to_json())

# # ──────────────────────────────────────────────────────────────
# # Helper to load one CSV and return a *full* (Commune, Month) grid
# # with log-transformed values + global vmin/vmax for the series
# # ──────────────────────────────────────────────────────────────
# def load_data(path: str):
#     df = pd.read_csv(
#         path,
#         usecols=["Date", "Commune", "WeightedIndex"],
#         parse_dates=["Date"],
#         dtype={"Commune": "string", "WeightedIndex": "float32"},
#     )

#     df = (
#         df.query("Date >= '2000-01-01' and Date <= '2023-12-31'")
#         .assign(Commune=lambda d: d["Commune"].str.strip().str.upper().map(unidecode))
#     )
#     df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

#     # Mean index per (Commune, Year-Month)
#     agg = (
#         df.groupby(["Commune", "YearMonth"], sort=False)["WeightedIndex"]
#         .mean()
#         .reset_index()
#     )

#     # Build a full grid so every commune has a row for every month
#     months = sorted(agg["YearMonth"].unique())
#     communes = shapefile_gdf["Commune"].unique()
#     grid = (
#         pd.MultiIndex.from_product([communes, months], names=["Commune", "YearMonth"])
#         .to_frame(index=False)
#     )
#     full = grid.merge(agg, on=["Commune", "YearMonth"], how="left").fillna(0)

#     merged = shapefile_gdf.merge(full, on="Commune", how="left")
#     merged["LogIndex"] = np.log1p(merged["WeightedIndex"])

#     return merged, months, merged["LogIndex"].min(), merged["LogIndex"].max()


# # ──────────────────────────────────────────────────────────────
# # Load both datasets
# # ──────────────────────────────────────────────────────────────
# merged6, months6, vmin6, vmax6 = load_data(CSV_LINEAR)
# merged3, months3, vmin3, vmax3 = load_data(CSV_UNIFORM)
# common_months = [m for m in months6 if m in months3]  # timeline intersection

# # Ensure row order is always the same so the *z* vectors line up
# ordered_communes = shapefile_gdf["Commune"]
# def z_vector(df, month):
#     return (
#         df.loc[df.YearMonth == month, ["Commune", "LogIndex"]]
#         .set_index("Commune")
#         .reindex(ordered_communes)
#         .fillna(0)["LogIndex"]
#         .tolist()
#     )

# # ──────────────────────────────────────────────────────────────
# # Build the initial figures (first month only)
# # ──────────────────────────────────────────────────────────────
# frame0 = common_months[0]
# df6_0 = merged6[merged6["YearMonth"] == frame0]
# df3_0 = merged3[merged3["YearMonth"] == frame0]

# fig6_init = px.choropleth(
#     df6_0,
#     geojson=geojson,
#     featureidkey="properties.Commune",
#     locations="Commune",
#     color="LogIndex",
#     range_color=(vmin6, vmax6),
#     color_continuous_scale="YlOrRd",
#     projection="mercator",
#     hover_name="Commune",
#     title=f"Linear weighting – {frame0}",
#     width=800,
#     height=800,
# )
# fig6_init.update_geos(fitbounds="locations", visible=False)
# fig6_init.update_layout(margin=dict(r=10, t=40, l=10, b=10))
# fig6_init.update_coloraxes(colorbar=dict(thickness=8, len=0.5, title="Log Index"))

# fig3_init = px.choropleth(
#     df3_0,
#     geojson=geojson,
#     featureidkey="properties.Commune",
#     locations="Commune",
#     color="LogIndex",
#     range_color=(vmin3, vmax3),
#     color_continuous_scale="YlOrRd",
#     projection="mercator",
#     hover_name="Commune",
#     title=f"Uniform weighting – {frame0}",
#     width=800,
#     height=800,
# )
# fig3_init.update_geos(fitbounds="locations", visible=False)
# fig3_init.update_layout(margin=dict(r=10, t=40, l=10, b=10))
# fig3_init.update_coloraxes(colorbar=dict(thickness=8, len=0.5, title="Log Index"))

# # ──────────────────────────────────────────────────────────────
# # Dash app
# # ──────────────────────────────────────────────────────────────
# app = Dash(__name__)

# app.layout = html.Div(
#     [
#         html.H1(
#             "Linear Weighting vs Uniform Weighting – Logarithmic Index",
#             style={"textAlign": "center"},
#         ),
#         html.Div(
#             [
#                 html.Button("Play", id="play-btn", n_clicks=0),
#                 html.Button("Pause", id="pause-btn", n_clicks=0),
#                 dcc.Slider(
#                     id="month-slider",
#                     min=0,
#                     max=len(common_months) - 1,
#                     value=0,
#                     marks={i: m for i, m in enumerate(common_months)},
#                     updatemode="drag",
#                     tooltip={"always_visible": False, "placement": "bottom"},
#                 ),
#                 dcc.Interval(
#                     id="interval", interval=1000, n_intervals=0, disabled=True
#                 ),
#             ],
#             style={"width": "80%", "margin": "auto", "padding": "20px"},
#         ),
#         html.Div(
#             [
#                 dcc.Graph(id="map-6m", figure=fig6_init, style={"width": "48vw", "height": "75vh"}),
#                 dcc.Graph(id="map-3m", figure=fig3_init, style={"width": "48vw", "height": "75vh"}),
#             ],
#             style={"display": "flex", "justifyContent": "space-around"},
#         ),
#     ]
# )

# # ──────────────────────────────────────────────────────────────
# # Callbacks
# # ──────────────────────────────────────────────────────────────
# @app.callback(
#     Output("interval", "disabled"),
#     Input("play-btn", "n_clicks"),
#     Input("pause-btn", "n_clicks"),
# )
# def toggle_interval(play, pause):
#     """Enable the Interval when Play has been clicked more times than Pause."""
#     return not (play > pause)


# @app.callback(
#     Output("month-slider", "value"),
#     Input("interval", "n_intervals"),
#     State("month-slider", "value"),
# )
# def advance_slider(n_intervals, current_value):
#     """Move to the next month every tick (wrapping around)."""
#     return (current_value + 1) % len(common_months)


# @app.callback(
#     [Output("map-6m", "figure"), Output("map-3m", "figure")],
#     Input("month-slider", "value"),
#     prevent_initial_call=True,  # we already showed the first frame
# )
# def update_maps(idx):
#     """Return two Patch objects that only update z-values + titles."""
#     frame = common_months[idx]

#     # Build the new z vectors in the exact same order as the initial figures
#     z6 = z_vector(merged6, frame)
#     z3 = z_vector(merged3, frame)

#     # Patch for the 6-month map
#     patch6 = Patch()
#     patch6["data"][0]["z"] = z6
#     patch6["layout"]["title"]["text"] = f"Linear weighting – {frame}"

#     # Patch for the 3-month map
#     patch3 = Patch()
#     patch3["data"][0]["z"] = z3
#     patch3["layout"]["title"]["text"] = f"Uniform weighting – {frame}"

#     return patch6, patch3


# # ──────────────────────────────────────────────────────────────
# # Main
# # ──────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8050))
#     app.run(host="0.0.0.0", port=port, debug=False)



















import json, os
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Patch, dcc, html
from dash.dependencies import Input, Output, State
from unidecode import unidecode

# Paths
SHAPEFILE   = "data/shapefiles/NER_admbnda_adm3_IGNN_20230720.shp"
CSV_LINEAR  = "data/Weighted_Vulnerability_Index_Linear.csv"    
CSV_UNIFORM = "data/Weighted_Vulnerability_Index_Uniform.csv"   

# Read shapefile and retain only the commune name and polygon geometry
gdf = gpd.read_file(SHAPEFILE)[["ADM3_FR", "geometry"]]
# Clean up the commune names to be more easily readable
gdf["Commune"]  = gdf["ADM3_FR"].str.strip().str.upper().map(unidecode)
# Simplify geometry to speed up plotting. .buffer(0) fixes any invalid geometry from simplification
gdf["geometry"] = gdf["geometry"].simplify(0.01).buffer(0)
# Combine polygons with the same commune name and reset the index
shapefile_gdf   = gdf.dissolve("Commune").reset_index()
# Convert to GeoJSON format as this is used by Plotly maps
geojson         = json.loads(shapefile_gdf.to_json())

## Helper function: Load and Process CSV files
# Define a function to read and format a weighted index CSV file
def load_csv(path: str):
    # Load selected columns, parse the dat column and set data types
    df = pd.read_csv(
        path, usecols=["Date", "Commune", "WeightedIndex"],
        parse_dates=["Date"], dtype={"Commune": "string", "WeightedIndex": "float32"}
    )
    # Filter dates to our period we are considering (2000-2023) and clean up commune names again
    df = (
        df.query("Date >= '2000-01-01' and Date <= '2023-12-31'")
          .assign(Commune=lambda d: d["Commune"].str.strip().str.upper().map(unidecode))
    )
    # Extract the "year-month" (eg., 2021-06) from the date
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    # Aggregate by commune and month (in case of mutliple entries), taking the average index
    agg = df.groupby(["Commune", "YearMonth"], sort=False)["WeightedIndex"].mean().reset_index()

    # Retrieve a list of all months and communes
    months   = sorted(agg["YearMonth"].unique())
    communes = shapefile_gdf["Commune"].unique()
    # Creates a full grid of all commune-month combinations (ensures every location has all timepoints)
    grid     = pd.MultiIndex.from_product([communes, months],
                                          names=["Commune", "YearMonth"]).to_frame(index=False)
    # Merge actual data with the full grid, and missing entries are set to zero
    full     = grid.merge(agg, on=["Commune", "YearMonth"], how="left").fillna(0)

    # Add spatial geometry to the dataset by merging with the shapefile
    merged = shapefile_gdf.merge(full, "left", "Commune")
    # Apply log(1 + x) transformation to stabilise skewed data for colour mapping
    merged["LogIndex"] = np.log1p(merged["WeightedIndex"])
    # Return the processed GeoDataFrame, list of months, and min/max values of the log transformed index 
    return merged, months, merged["LogIndex"].min(), merged["LogIndex"].max()

# Load both datasets 
merged6, months6, vmin6, vmax6 = load_csv(CSV_LINEAR)
merged3, months3, vmin3, vmax3 = load_csv(CSV_UNIFORM)
# Get a list of months present in both datasets
common_months = [m for m in months6 if m in months3]

# Save the order of communes for consistent mapping
ordered_communes = shapefile_gdf["Commune"]
# For a given month:
# Gget the log index values by commune
# Re-index so all communes are present and consistently ordered
# Fill the missing with 0
# Return this as a list (later used for updating the map)
def z_vec(df, month):
    return (
        df.loc[df.YearMonth == month, ["Commune", "LogIndex"]]
          .set_index("Commune").reindex(ordered_communes).fillna(0)["LogIndex"]
          .tolist()
    )

# Get data for the first available month
frame0  = common_months[0]
df6_0   = merged6[merged6["YearMonth"] == frame0]
df3_0   = merged3[merged3["YearMonth"] == frame0]

# Function to build a choropleth map with Plotly
def base_fig(data, rng, title):
    fig = px.choropleth(
        data, geojson=geojson, featureidkey="properties.Commune",
        locations="Commune", color="LogIndex",
        range_color=rng, color_continuous_scale="YlOrRd",
        projection="mercator", hover_name="Commune",
        title=title, width=800, height=800,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(r=10, t=40, l=10, b=10))
    fig.update_coloraxes(colorbar=dict(thickness=8, len=0.5, title="Log Index"))
    return fig

# Create the initial two maps for display
fig6_init = base_fig(df6_0, (vmin6, vmax6), f"Linear weighting – {frame0}")
fig3_init = base_fig(df3_0, (vmin3, vmax3), f"Uniform weighting – {frame0}")

# Create the Dash app
app = Dash(__name__)

# Slider marks: label only every January (-01)
slider_marks = {i: m for i, m in enumerate(common_months) if m.endswith("-01")}

# Sets the layout, the html.Div is a container that holds all UI elements
app.layout = html.Div(
    [   # Adds the header at the top of the page centred 
        html.H1("Linear vs Uniform Weighting – Logarithmic Index",
                style={"textAlign": "center"}),

        # A nested <div> that groups all interactive controls, so play\pause, slider, and label
        html.Div(
            [   # Two clickable buttons
                # Each is assigned a unique id so Dash callbacks can reference them
                # n_clicks = 0 initialises a counter for how many times the button has been pressed
                html.Button("Play",  id="play-btn",  n_clicks=0),
                html.Button("Pause", id="pause-btn", n_clicks=0),

                # Text element to display the currently selected month
                # children=frame0 starts with the first month displayed
                # The other options is just the format
                html.Span(id="month-label",
                          children=frame0,
                          style={"marginLeft": "1rem",
                                 "fontWeight": 600,
                                 "fontSize": "1.1rem"}),

                # A slider UI element to select the time frame (by index)
                dcc.Slider(
                    # Referenced by callbacks
                    id="month-slider",
                    # Slider moves from first to last month index with only whole months (step=1). Value = 0 specifies to start at the first month
                    min=0, max=len(common_months) - 1, step=1, value=0,
                    # Show selected labels
                    marks=slider_marks,
                    # Value changes continuously while dragging
                    updatemode="drag",
                    # Tooltip appears on hover, not always
                    tooltip={"always_visible": False, "placement": "bottom"},
                ),
                # This controls the animation. A timer component that triggers callbacks at regular intervals
                dcc.Interval(id="interval",
                             # interval=100: ticks every 1000ms (1 second)
                             # disabled=True means its initially off, only turns on when play is clicked
                             interval=1000, n_intervals=0, disabled=True),
            ],
            # Formatting the style of the entire control block
            style={"width": "80%", "margin": "auto", "padding": "20px"},
        ),

        # Another container <div> to hold the two maps in parallel
        html.Div(
            [   # 1st map
                dcc.Graph(id="map-6m", figure=fig6_init,
                          style={"width": "48vw", "height": "75vh"}),
                # 2nd map
                dcc.Graph(id="map-3m", figure=fig3_init,
                          style={"width": "48vw", "height": "75vh"}),
            ],
            # Displays the two maps horizontally
            style={"display": "flex", "justifyContent": "space-around"},
        ),
    ]
)

# Controls whether the animation is running
@app.callback(
    Output("interval", "disabled"),
    Input("play-btn", "n_clicks"),
    Input("pause-btn", "n_clicks"),
)
# Only enable the interval if Play has been clicked more times than pause
def toggle_interval(play, pause):
    return not (play > pause)

# Advances the month slider everytime the interval ticks (every second)
@app.callback(
    Output("month-slider", "value"),
    Input("interval", "n_intervals"),
    State("month-slider", "value"),
)
# Moves forward by one step
def advance_slider(_, current):
    return (current + 1) % len(common_months)

# Updates both maps and labels when slider changes
@app.callback(
    [Output("map-6m", "figure"),
     Output("map-3m", "figure"),
     Output("month-label", "children")],            # NEW: update label
    Input("month-slider", "value"),
    prevent_initial_call=True,
)
# Gets the current YYYY-MM month string based on the index from the slider
def update_maps(idx):
    frame = common_months[idx]

    # Get the updated log index values for each commune, for this month, for both datasets
    z6 = z_vec(merged6, frame)
    z3 = z_vec(merged3, frame)

    # Patch is used to do partial figure updates rather than fully updating the figure each interval (the latter proves too much for Dash)
    patch6 = Patch()
    patch6["data"][0]["z"]           = z6
    patch6["layout"]["title"]["text"] = f"Linear weighting – {frame}"

    # Same thing here
    patch3 = Patch()
    patch3["data"][0]["z"]           = z3
    patch3["layout"]["title"]["text"] = f"Uniform weighting – {frame}"

    # Returns the two updated maps and the current month (for the label)
    return patch6, patch3, frame      # label text

# Ensures the app runs only when the script is executed directly (not when imported)
if __name__ == "__main__":
    # Get the port from the environment used for hosting (Render)
    port = int(os.environ.get("PORT", 8050))
    # Run the dash app on the specified port
    app.run(host="0.0.0.0", port=port, debug=False)












# Find a commune with non-zero index values
active_communes = (
    merged6[merged6["WeightedIndex"] > 0]["Commune"]
    .unique()
)
import random
sample_commune = random.choice(list(active_communes))
print("Selected commune:", sample_commune)

# Get time series for the selected commune
df6_ts = merged6[merged6["Commune"] == sample_commune][["YearMonth", "WeightedIndex"]].copy()
df6_ts["Weighting"] = "Linear"

df3_ts = merged3[merged3["Commune"] == sample_commune][["YearMonth", "WeightedIndex"]].copy()
df3_ts["Weighting"] = "Uniform"

# Combine for plotting
ts_combined = pd.concat([df6_ts, df3_ts])

import plotly.express as px

fig = px.line(
    ts_combined, x="YearMonth", y="WeightedIndex", color="Weighting",
    title=f"Time Series of Weighted Index – {sample_commune}",
    labels={"WeightedIndex": "Index", "YearMonth": "Month"},
    width=900, height=450
)
fig.update_layout(xaxis_tickangle=45)
fig.show()