# import json
# import geopandas as gpd
# import pandas as pd
# import numpy as np
# from unidecode import unidecode
# import plotly.express as px
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State
# import os

# os.environ['DASH_REQUESTS_PATHNAME_PREFIX'] = '/'

# # Variables with '6' for 6 months == csv_path_1 
# # Variables with '3' for 3 months == csv_path_2

# # ──────────────────────────────────────────────────────────────
# # File paths — update to your local files
# # ──────────────────────────────────────────────────────────────
# shapefile_path = 'Niger shapefiles - Git/NER_admbnda_adm3_IGNN_20230720.shp'
# csv_path_1    = 'Weighted_Vulnerability_Index_Linear.csv'
# csv_path_2    = 'Weighted_Vulnerability_Index_Uniform.csv'

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
























import json
import geopandas as gpd
import pandas as pd
import numpy as np
from unidecode import unidecode
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import os

os.environ['DASH_REQUESTS_PATHNAME_PREFIX'] = '/'

# ──────────────────────────────────────────────────────────────
# File paths — relative to repo
# ──────────────────────────────────────────────────────────────
shapefile_path = 'data/Niger shapefiles - Git/NER_admbnda_adm3_IGNN_20230720.shp'
csv_path_1    = 'data/Weighted_Vulnerability_Index_Linear.csv'
csv_path_2    = 'data/Weighted_Vulnerability_Index_Uniform.csv'

# ──────────────────────────────────────────────────────────────
# Lazy loading helper
# ──────────────────────────────────────────────────────────────
def get_data():
    gdf = gpd.read_file(shapefile_path)[['ADM3_FR','geometry']]
    gdf['Commune'] = gdf['ADM3_FR'].str.strip().str.upper().map(unidecode)
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01).buffer(0)
    shapefile_gdf = gdf.dissolve(by='Commune').reset_index()
    geojson = json.loads(shapefile_gdf.to_json())

    def load_data(csv_path):
        df = pd.read_csv(
            csv_path,
            usecols=['Date', 'Commune', 'WeightedIndex'],
            parse_dates=['Date'],
            dtype={'Commune':'string','WeightedIndex':'float32'}
        )
        df = (
            df
            .query("Date >= '2000-01-01' and Date <= '2023-12-31'")
            .assign(
                Commune=lambda d: d['Commune'].str.strip().str.upper().map(unidecode)
            )
        )
        df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
        agg = df.groupby(['Commune','YearMonth'], sort=False)['WeightedIndex'].mean().reset_index()
        months = sorted(agg['YearMonth'].unique())
        communes = shapefile_gdf['Commune'].unique()
        grid = pd.MultiIndex.from_product([communes, months], names=['Commune','YearMonth']).to_frame(index=False)
        full = grid.merge(agg, on=['Commune','YearMonth'], how='left').fillna(0)
        merged = shapefile_gdf.merge(full, on='Commune', how='left')
        merged['LogIndex'] = np.log1p(merged['WeightedIndex'])
        vmin = merged['LogIndex'].min()
        vmax = merged['LogIndex'].max()
        return merged, months, vmin, vmax

    merged6, months6, vmin6, vmax6 = load_data(csv_path_1)
    merged3, months3, vmin3, vmax3 = load_data(csv_path_2)
    common_months = [m for m in months6 if m in months3]

    return merged6, merged3, vmin6, vmax6, vmin3, vmax3, common_months, geojson

# ──────────────────────────────────────────────────────────────
# Initialize Dash app
# ──────────────────────────────────────────────────────────────
app = Dash(__name__)

# ──────────────────────────────────────────────────────────────
# Layout: play/pause, slider, and two log maps
# ──────────────────────────────────────────────────────────────
merged6, merged3, vmin6, vmax6, vmin3, vmax3, common_months, geojson = get_data()

app.layout = html.Div([
    html.H1('Linear Weighting vs Uniform Weighting - Logarithmic Index', style={'text-align':'center'}),
    html.Div([
        html.Button('Play', id='play-btn', n_clicks=0),
        html.Button('Pause', id='pause-btn', n_clicks=0),
        dcc.Slider(
            id='month-slider',
            min=0, max=len(common_months)-1, value=0,
            marks={i: m for i, m in enumerate(common_months)},
            updatemode='drag',
            tooltip={'always_visible':False, 'placement':'bottom'}
        ),
        dcc.Interval(id='interval', interval=1000, n_intervals=0, disabled=True)
    ], style={'width':'80%', 'margin':'auto', 'padding':'20px'}),
    html.Div([
        dcc.Graph(id='map-6m', style={'width':'48vw', 'height':'75vh'}),
        dcc.Graph(id='map-3m', style={'width':'48vw', 'height':'75vh'})
    ], style={'display':'flex', 'justify-content':'space-around'})
])

# ──────────────────────────────────────────────────────────────
# Callbacks: control interval, advance slider, update maps
# ──────────────────────────────────────────────────────────────
@app.callback(
    Output('interval', 'disabled'),
    Input('play-btn', 'n_clicks'),
    Input('pause-btn', 'n_clicks')
)
def toggle_interval(play, pause):
    return not (play > pause)

@app.callback(
    Output('month-slider', 'value'),
    Input('interval', 'n_intervals'),
    State('month-slider', 'value')
)
def advance_slider(n, current):
    _, _, _, _, _, _, common_months, _ = get_data()
    return (current + 1) % len(common_months)

@app.callback(
    [Output('map-6m', 'figure'), Output('map-3m', 'figure')],
    Input('month-slider', 'value')
)
def update_maps(idx):
    merged6, merged3, vmin6, vmax6, vmin3, vmax3, common_months, geojson = get_data()
    frame = common_months[idx]
    df6 = merged6[merged6['YearMonth'] == frame]
    df3 = merged3[merged3['YearMonth'] == frame]

    fig6 = px.choropleth(
        df6, geojson=geojson, featureidkey='properties.Commune',
        locations='Commune', color='LogIndex',
        range_color=(vmin6, vmax6), color_continuous_scale='YlOrRd',
        projection='mercator', hover_name='Commune',
        title=f'Linear weighting – {frame}', width=800, height=800
    )
    fig6.update_geos(fitbounds='locations', visible=False)
    fig6.update_layout(margin={'r':10,'t':40,'l':10,'b':10})
    fig6.update_coloraxes(colorbar=dict(thickness=8, len=0.5, title='Log Index'))

    fig3 = px.choropleth(
        df3, geojson=geojson, featureidkey='properties.Commune',
        locations='Commune', color='LogIndex',
        range_color=(vmin3, vmax3), color_continuous_scale='YlOrRd',
        projection='mercator', hover_name='Commune',
        title=f'Uniform weighting – {frame}', width=800, height=800
    )
    fig3.update_geos(fitbounds='locations', visible=False)
    fig3.update_layout(margin={'r':10,'t':40,'l':10,'b':10})
    fig3.update_coloraxes(colorbar=dict(thickness=8, len=0.5, title='Log Index'))

    return fig6, fig3

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
