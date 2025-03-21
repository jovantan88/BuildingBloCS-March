import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree

st.set_page_config(
    page_title="Australia Wildfire Biodiversity Impact Analysis",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Australia Wildfire Impact on Biodiversity")
st.markdown("""
This interactive dashboard analyzes the impact of Australian wildfires on biodiversity,
with a focus on koala populations. We explore wildfire patterns, environmental conditions,
and their correlation with koala sightings to understand threats to biodiversity.
""")

st.markdown("---")

st.header("📊 Data Sources")
st.markdown("""
- **modis_2000-2019_Australia.csv** is obtained from [Kaggle - Satellite Data on Australia Fires](https://www.kaggle.com/datasets/gabrielbgutierrez/satellite-data-on-australia-fires)
- **wildnetkoalalocations.csv** is obtained from Australian wildlife conservation databases
- **weatherAUS.csv** is obtained from Australian Bureau of Meteorology
""")

st.markdown("---")

@st.cache_data
def load_data():
    df_fires = pd.read_csv('modis_2000-2019_Australia.csv')
    df_fires['acq_date'] = pd.to_datetime(df_fires['acq_date'])
    df_fires['year'] = df_fires['acq_date'].dt.year

    try:
        df_weather = pd.read_csv('weatherAUS.csv', parse_dates=['Date'])
        df_weather['RainToday'] = df_weather['RainToday'].map({'No': 0, 'Yes': 1})
        df_weather['Year'] = df_weather['Date'].dt.year
        df_weather['Month'] = df_weather['Date'].dt.month
    except:
        df_weather = None

    try:
        df_koalas = pd.read_csv('wildnetkoalalocations.csv', parse_dates=['StartDate'], dayfirst=True)
        df_koalas = df_koalas[(df_koalas['StartDate'].dt.year >= 2007) &
                    (df_koalas['StartDate'].dt.year <= 2016)]
        df_koalas['Year'] = df_koalas['StartDate'].dt.year
        df_koalas['Month'] = df_koalas['StartDate'].dt.month
    except:
        df_koalas = None

    return df_fires, df_weather, df_koalas

df_fires, df_weather, df_koalas = load_data()

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Wildfire Visualization", "Burned Area Analysis", "Temperature Analysis", "Koala & Climate Analysis", "Fire Impact on Koalas", "Summary & Conclusion"]
)

if section == "Wildfire Visualization":
    st.header("🗺️ Australian Wildfires (2000-2019)")

    st.markdown("""
    This visualization shows the distribution and intensity of wildfires across Australia from 2000 to 2019.
    The data is aggregated into grid cells to make visualization more efficient.

    - **Color intensity** represents the total fire radiative power (FRP) - a measure of fire intensity
    - **Circle size** represents the number of fire detections in that area
    - **Animation** shows the progression of fires over the years
    """)

    with st.spinner('Generating wildfire visualization...'):
        df = df_fires.copy()
        df = df[df['frp'] >= 0]
        df = df.sort_values('acq_date')

        resolution = st.slider("Grid Resolution (degrees)", 0.1, 1.0, 0.25, 0.05)

        df['grid_lat'] = np.floor(df['latitude'] / resolution) * resolution
        df['grid_lon'] = np.floor(df['longitude'] / resolution) * resolution

        grid_data = df.groupby(['year', 'grid_lat', 'grid_lon']).agg({
            'frp': ['count', 'sum', 'mean', 'max'],
            'acq_date': 'min'
        }).reset_index()

        grid_data.columns = [
            'year', 'grid_lat', 'grid_lon',
            'fire_count', 'frp_sum', 'frp_avg', 'frp_max',
            'first_fire'
        ]

        grid_data['lat'] = grid_data['grid_lat'] + resolution/2
        grid_data['lon'] = grid_data['grid_lon'] + resolution/2

        st.write(f"Reduced to {len(grid_data):,} data points")

        years = sorted(grid_data['year'].unique())

        fig = px.scatter_geo(
            grid_data,
            lat='lat',
            lon='lon',
            color='frp_sum', 
            size='fire_count',  
            animation_frame='year',
            color_continuous_scale='Inferno',  
            scope='world',
            projection='natural earth',
            title='Australian Wildfires (2000-2019)',
            hover_data={
                'grid_lat': False,
                'grid_lon': False,
                'lat': ':.2f',
                'lon': ':.2f',
                'fire_count': True,
                'frp_sum': ':.1f',
                'frp_avg': ':.2f',
                'frp_max': ':.1f',
                'first_fire': True
            },
            height=700
        )

        fig.update_layout(
            geo=dict(
                lataxis_range=[-45, -10],
                lonaxis_range=[110, 155],
                showland=True,
                landcolor='rgb(217, 217, 217)',
                coastlinecolor='white',
                countrycolor='white',
                showocean=True,
                oceancolor='rgb(204, 229, 255)',
                showcountries=True,
                showsubunits=True,
                subunitcolor='white'
            ),
            font=dict(size=12),
            title_font=dict(size=20),
            margin=dict(l=0, r=0, t=80, b=0),
            coloraxis_colorbar_title='Total FRP'
        )

        fig.update_traces(
            marker=dict(
                opacity=0.75,
                line=dict(width=0.5, color='DarkSlateGrey'),
                sizemode='area',
                sizeref=2.*max(grid_data['fire_count'])/(30.**2),
                sizemin=2
            )
        )

        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1200
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 600
        fig.layout.sliders[0].currentvalue.prefix = 'Year: '

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    From the timelapse above, we can observe that Australia experiences significant wildfire activity,
    particularly during the period 2011-2013. This high level of fire activity potentially poses a severe
    threat to local wildlife and biodiversity.
    """)

elif section == "Burned Area Analysis":
    st.header("🌋 Annual Burned Area Analysis")

    st.markdown("""
    This section analyzes the annual pattern of burned areas in Australia from 2000-2019.
    We can observe trends and identify particularly severe fire seasons.
    """)

    with st.spinner('Analyzing burned area...'):
        df = df_fires.copy()
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        df['year'] = df['acq_date'].dt.year

        scale_factor = st.number_input("Area Scale Factor (km² per detection)", 0.5, 2.0, 1.0, 0.1,
                                     help="Adjust this value to calibrate the burned area estimate")

        annual_area = df.groupby('year').size().reset_index(name='detections')
        annual_area['burned_area'] = annual_area['detections'] * scale_factor

        fig = px.line(
            annual_area,
            x='year',
            y='burned_area',
            markers=True,
            title='Annual Burned Area in Australia',
            labels={'burned_area': f'Burned Area Estimate (km²)', 'year': 'Year'}
        )

        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='lightgrey'),
            hovermode='x unified'
        )

        fig.update_traces(
            line=dict(width=2.5, color='#FF5722'),
            marker=dict(size=8, color='#E64A19')
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    We can observe a striking pattern in the burned area data:

    - Years seem to alternate between high and low fire activity
    - The period 2011-2012 shows particularly high levels of fire activity
    - These peaks in fire activity could be attributed to:
        - Harsher wind conditions
        - Lower rainfall
        - Reduced humidity
        - Climate anomalies

    These severe fire seasons potentially have a devastating impact on biodiversity, especially on vulnerable species like koalas that are dependent on specific forest habitats.
    """)

elif section == "Temperature Analysis":
    st.header("🌡️ Temperature Trends Analysis")

    if df_weather is not None:
        st.markdown("""
        Analyzing temperature trends across Australia to understand if there's a correlation between
        overall temperature patterns and fire activity. This helps us understand if rising temperatures
        are contributing to increased fire risk.
        """)

        with st.spinner('Analyzing temperature data...'):
            yearly_monthly = df_weather.groupby(['Year', 'Month'])[['MinTemp', 'MaxTemp']].mean().reset_index()

            temp_type = st.radio("Temperature Measurement", ["MaxTemp", "MinTemp"], horizontal=True)

            fig = px.line(yearly_monthly,
                        x='Month',
                        y=temp_type,
                        color='Year',
                        title=f'Yearly {temp_type} Trends by Month',
                        labels={temp_type: 'Temperature (°C)', 'Month': ''},
                        markers=True)

            fig.update_xaxes(
                tickvals=list(range(1,13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )

            fig.update_layout(
                yaxis=dict(
                    title="Temperature (°C)",
                    showgrid=True
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        From the line graph above, we can observe that the temperature patterns across years don't show
        dramatic changes. This suggests that the overall temperature across Australia is not the only
        factor affecting biodiversity decline or fire intensity. Local climate conditions, seasonal
        variations, and specific fire events likely have more direct impacts on wildlife populations than
        broad temperature trends.
        """)
    else:
        st.error("Weather data not available. Please ensure 'weatherAUS.csv' is in the correct location.")

elif section == "Koala & Climate Analysis":
    st.header("🐨 Koala Population & Climate Analysis")

    if df_koalas is not None and df_weather is not None:
        st.markdown("""
        This section analyzes the relationship between koala sightings and climate conditions.
        We explore how temperature and rainfall patterns affect koala populations across Australia.
        """)

        with st.spinner('Analyzing koala and climate data...'):
            weather_station_coords = {
                'Albury': (-36.0748, 146.9240),
                'BadgerysCreek': (-33.8844, 150.7833),
                'Cobar': (-31.4967, 145.8344),
                'CoffsHarbour': (-30.2963, 153.1135),
                'Moree': (-29.4628, 149.8416),
                'Newcastle': (-32.9267, 151.7789),
                'NorahHead': (-33.2815, 151.5680),
                'NorfolkIsland': (-29.0408, 167.9547),
                'Penrith': (-33.7500, 150.7000),
                'Richmond': (-33.6000, 150.7500),
                'Sydney': (-33.8688, 151.2093),
                'SydneyAirport': (-33.9399, 151.1753),
                'WaggaWagga': (-35.1086, 147.3697),
                'Williamtown': (-32.8064, 151.8436),
                'Wollongong': (-34.4240, 150.8935),
                'Canberra': (-35.2809, 149.1300),
                'Tuggeranong': (-35.4244, 149.0888),
                'MountGinini': (-35.5299, 148.7726),
                'Ballarat': (-37.5622, 143.8503),
                'Bendigo': (-36.7578, 144.2784),
                'Sale': (-38.1040, 147.0675),
                'MelbourneAirport': (-37.6690, 144.8410),
                'Melbourne': (-37.8136, 144.9631),
                'Mildura': (-34.2066, 142.1350),
                'Nhil': (-36.3333, 141.6500),
                'Portland': (-38.3333, 141.6000),
                'Watsonia': (-37.7167, 145.0833),
                'Dartmoor': (-37.9167, 141.2667),
                'Brisbane': (-27.4698, 153.0251),
                'Cairns': (-16.9203, 145.7710),
                'GoldCoast': (-28.0167, 153.4000),
                'Townsville': (-19.2590, 146.8169),
                'Adelaide': (-34.9285, 138.6007),
                'MountGambier': (-37.8316, 140.7652),
                'Nuriootpa': (-34.4682, 139.0047),
                'Woomera': (-31.1999, 136.8326),
                'Albany': (-35.0225, 117.8911),
                'Witchcliffe': (-34.0333, 115.1000),
                'PearceRAAF': (-31.6678, 116.0250),
                'PerthAirport': (-31.9403, 115.9669),
                'Perth': (-31.9505, 115.8605),
                'SalmonGums': (-32.9833, 121.6333),
                'Walpole': (-34.9833, 116.7333),
                'Hobart': (-42.8821, 147.3272),
                'Launceston': (-41.4332, 147.1441),
                'AliceSprings': (-23.6980, 133.8807),
                'Darwin': (-12.4634, 130.8456),
                'Katherine': (-14.4652, 132.2635),
                'Uluru': (-25.3444, 131.0369)
            }

            df_weather_with_coords = df_weather.copy()
            df_weather_with_coords['Latitude'] = df_weather_with_coords['Location'].map(lambda x: weather_station_coords.get(x, (np.nan, np.nan))[0])
            df_weather_with_coords['Longitude'] = df_weather_with_coords['Location'].map(lambda x: weather_station_coords.get(x, (np.nan, np.nan))[1])
            df_weather_with_coords = df_weather_with_coords.dropna(subset=['Latitude', 'Longitude'])

            def create_geometry(df, lat_col, lon_col):
                gdf = gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col])
                )
                return gdf.set_crs(epsg=4326)

            koala_gdf = create_geometry(df_koalas, 'Latitude', 'Longitude')
            weather_gdf = create_geometry(df_weather_with_coords.groupby('Location').first().reset_index(),
                                        'Latitude', 'Longitude')

            def find_nearest(source_points, target_points):
                source_points['rad_lat'] = np.radians(source_points['Latitude'])
                source_points['rad_lon'] = np.radians(source_points['Longitude'])
                target_points['rad_lat'] = np.radians(target_points['Latitude'])
                target_points['rad_lon'] = np.radians(target_points['Longitude'])

                tree = BallTree(target_points[['rad_lat', 'rad_lon']].values, leaf_size=15)
                distances, indices = tree.query(source_points[['rad_lat', 'rad_lon']].values)
                return indices.flatten(), distances.flatten()

            indices, distances = find_nearest(koala_gdf, weather_gdf)
            koala_gdf['Nearest_Location'] = weather_gdf.iloc[indices]['Location'].values
            koala_gdf['Distance_km'] = distances * 6371  # Convert to kilometers

            merged = pd.merge(koala_gdf, df_weather_with_coords,
                              left_on=['Nearest_Location', 'Year', 'Month'],
                              right_on=['Location', 'Year', 'Month'],
                              how='left')

            annual_analysis = merged.groupby(['Year', 'Nearest_Location', 'Latitude_y', 'Longitude_y']).agg(
                Koala_Count=('ScientificName', 'count'),
                Avg_MaxTemp=('MaxTemp', 'mean'),
                Total_Rainfall=('Rainfall', 'sum')
            ).reset_index()

            tab1, tab2, tab3 = st.tabs(["Koala Density by Climate", "Koala & Temperature Trends", "Koala Distribution Map"])

            with tab1:
                st.subheader("Koala Density by Temperature and Rainfall")

                fig1 = px.density_heatmap(
                    annual_analysis,
                    x='Avg_MaxTemp',
                    y='Total_Rainfall',
                    z='Koala_Count',
                    histfunc='avg',
                    title='Koala Sightings Density by Temperature and Rainfall',
                    labels={'Avg_MaxTemp': 'Average Max Temperature (°C)',
                            'Total_Rainfall': 'Annual Rainfall (mm)'}
                )

                st.plotly_chart(fig1, use_container_width=True)

                st.markdown("""
                The heatmap above shows that koalas tend to be sighted more frequently in areas with:
                - Higher rainfall amounts
                - Cooler temperatures (below ~28-29°C)

                This suggests koalas prefer temperate, well-watered environments, which aligns with their evolved habitat preferences.
                """)

            with tab2:
                st.subheader("Koala Sightings vs Temperature Trends")

                yearly_data = annual_analysis.groupby('Year').mean(numeric_only=True).reset_index()

                fig2 = px.line(yearly_data, x='Year', y='Koala_Count',
                            title='Annual Koala Sightings vs Temperature Trends',
                            labels={"Koala_Count": "Koala Count"},
                            custom_data=["Year"])

                fig2.data[0].name = "Koala Count"

                fig2.add_trace(
                    go.Scatter(
                        x=yearly_data['Year'],
                        y=yearly_data['Avg_MaxTemp'],
                        name='Avg Max Temp',
                        yaxis="y2"
                    )
                )

                fig2.update_layout(
                    yaxis=dict(title='Koala Count'),
                    yaxis2=dict(
                        title='Temperature (°C)',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(
                        x=0.01,
                        y=0.99,
                        title=None
                    )
                )

                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("""
                From the line graph above, we can observe an inverse correlation between temperature and koala sightings:
                - When temperatures increase, koala sightings typically decrease
                - When temperatures decrease, koala sightings increase

                This pattern suggests that koalas thrive better in cooler temperatures, which is consistent with their biological needs and habitat preferences.
                """)

            with tab3:
                st.subheader("Koala Distribution Map")

                max_count = annual_analysis['Koala_Count'].max()
                min_count = annual_analysis['Koala_Count'].min()

                def scale_radius(count, min_val=min_count, max_val=max_count, min_radius=8, max_radius=30):
                    if count <= min_val:
                        return min_radius

                    normalized_count = (count - min_val) / (max_val - min_val)
                    scaled = min_radius + (max_radius - min_radius) * np.sqrt(normalized_count)
                    return scaled

                m = folium.Map(
                    location=[-28, 135],  
                    zoom_start=4,         
                    tiles='CartoDB positron'  
                )

                koala_group = folium.FeatureGroup(name="Koala Sightings")

                for idx, row in annual_analysis.iterrows():
                    radius = scale_radius(row['Koala_Count'])

                    temp_color = '#FF5722' if row['Avg_MaxTemp'] > 30 else '#4CAF50'

                    folium.CircleMarker(
                        location=[row['Latitude_y'], row['Longitude_y']],
                        radius=radius,
                        color='black',
                        weight=1.5,
                        fill=True,
                        fill_color=temp_color,
                        fill_opacity=0.8,
                        popup=f"<strong>{row['Nearest_Location']}</strong><br>Count: {int(row['Koala_Count'])}<br>Avg Temp: {row['Avg_MaxTemp']:.1f}°C<br>Rainfall: {row['Total_Rainfall']:.1f}mm"
                    ).add_to(koala_group)

                koala_group.add_to(m)

                st.write("""
                **Map Legend:**
                - **Circle Size**: Larger circles indicate more koala sightings
                - **Circle Color**:
                  - 🔴 Red: Hot areas (>30°C)
                  - 🟢 Green: Moderate temperature areas (≤30°C)
                """)

                folium_static(m)

                st.markdown("""
                The map shows the distribution of koala sightings across Australia:
                - The eastern coast (marked predominantly with green circles) has cooler temperatures and higher koala counts
                - Inland areas (with more red circles) have higher temperatures and fewer koala sightings

                This spatial pattern reinforces our finding that koala populations favor cooler, coastal forest habitats.
                """)

    else:
        st.error("Koala or Weather data not available. Please ensure data files are in the correct location.")

elif section == "Fire Impact on Koalas":
    st.header("🔥🐨 Fire Impact on Koala Populations")

    if df_koalas is not None and df_fires is not None:
        st.markdown("""
        This section investigates the direct relationship between fire incidents and koala populations.
        We analyze how proximity to fires and fire intensity affect koala sightings across Australia.
        """)

        with st.spinner('Analyzing fire impact on koalas...'):
            koalas = df_koalas.copy()
            fires = df_fires.copy()

            koalas = koalas[(koalas["StartDate"].dt.year >= 2007) & (koalas["StartDate"].dt.year <= 2016)]
            fires = fires[(fires["acq_date"].dt.year >= 2007) & (fires["acq_date"].dt.year <= 2016)]

            koalas_geometry = [Point(xy) for xy in zip(koalas["Longitude"], koalas["Latitude"])]
            koalas_gdf = gpd.GeoDataFrame(koalas, geometry=koalas_geometry, crs="EPSG:4326")

            fires_geometry = [Point(xy) for xy in zip(fires["longitude"], fires["latitude"])]
            fires_gdf = gpd.GeoDataFrame(fires, geometry=fires_geometry, crs="EPSG:4326")

            buffer_distance = st.slider("Buffer distance around koala sightings (degrees)", 0.01, 0.10, 0.05, 0.01,
                                      help="Approximately 5.5km at these latitudes when set to 0.05")

            koalas_gdf["buffer"] = koalas_gdf.geometry.buffer(buffer_distance)

            buffer_gdf = koalas_gdf.copy()
            buffer_gdf.geometry = buffer_gdf["buffer"]

            with st.spinner("Performing spatial join (this may take a moment)..."):
                st.info("Finding fires within koala habitat buffers...")
                fires_within_buffer = gpd.sjoin(fires_gdf, buffer_gdf, how="inner", predicate="within")
                st.success(f"Found {len(fires_within_buffer)} fires within koala habitat buffers")

                st.info("Calculating distances between fires and koalas...")

                if len(fires_within_buffer) > 5000:
                    st.warning(f"Large dataset detected ({len(fires_within_buffer)} points). Using a sample for distance calculation.")
                    fires_sample = fires_within_buffer.sample(5000)
                    fires_sample["distance_to_koala"] = fires_sample.apply(
                        lambda row: row.geometry.distance(koalas_gdf.loc[row.index_right, "geometry"]), axis=1
                    )
                    mean_distance = fires_sample["distance_to_koala"].mean()
                    st.write(f"Average distance from fire to nearest koala (sample): {mean_distance:.5f} degrees")
                    fires_within_buffer["distance_to_koala"] = np.nan
                else:
                    fires_within_buffer["distance_to_koala"] = fires_within_buffer.apply(
                        lambda row: row.geometry.distance(koalas_gdf.loc[row.index_right, "geometry"]), axis=1
                    )
                    mean_distance = fires_within_buffer["distance_to_koala"].mean()
                    st.write(f"Average distance from fire to nearest koala: {mean_distance:.5f} degrees")

            fires_within_buffer["fire_impact"] = fires_within_buffer["brightness"] * fires_within_buffer["frp"]


            with tab1:
                st.subheader("Koala Sightings and Fire Incidents Over Time")

                koalas_by_year = koalas.groupby(koalas["StartDate"].dt.year).size()
                fires_by_year = fires_within_buffer.groupby(fires_within_buffer["acq_date"].dt.year).size()

                trend_df = pd.DataFrame({
                    'Year': koalas_by_year.index,
                    'Koala_Count': koalas_by_year.values,
                })

                for year in trend_df['Year']:
                    if year in fires_by_year.index:
                        trend_df.loc[trend_df['Year'] == year, 'Fire_Count'] = fires_by_year[year]
                    else:
                        trend_df.loc[trend_df['Year'] == year, 'Fire_Count'] = 0

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=trend_df['Year'],
                    y=trend_df['Koala_Count'],
                    name='Koala Sightings',
                    mode='lines+markers',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ))

                fig.add_trace(go.Scatter(
                    x=trend_df['Year'],
                    y=trend_df['Fire_Count'],
                    name='Fire Incidents',
                    mode='lines+markers',
                    line=dict(color='red', width=3),
                    marker=dict(size=10),
                    yaxis='y2'
                ))

                fig.update_layout(
                    title='Koala Sightings and Fire Incidents Over Time (2007-2016)',
                    xaxis=dict(title='Year'),
                    yaxis=dict(title='Koala Sightings', side='left'),
                    yaxis2=dict(title='Fire Incidents', overlaying='y', side='right'),
                    legend=dict(x=0.01, y=0.99)
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                The line graph above reveals a clear inverse correlation between fire incidents and koala sightings:

                - When fire incidents increase, koala sightings decrease
                - When fire incidents decrease, koala sightings increase

                This pattern strongly suggests that wildfire events have a direct negative impact on koala populations,
                either through habitat destruction, direct mortality, or displacement of koala populations to areas
                where they are less likely to be observed.
                """)

            with tab2:
                st.subheader("Fire Intensity vs. Distance to Koala Sightings")

                if len(fires_within_buffer) > 5000:
                    plot_sample = fires_within_buffer.sample(5000)
                    fig = px.scatter(
                        plot_sample.dropna(subset=["distance_to_koala", "fire_impact"]),
                        x="distance_to_koala",
                        y="fire_impact",
                        title="Fire Intensity vs. Distance to Koala Sightings (Sample)",
                        labels={
                            "distance_to_koala": "Distance to Koala Sighting (degrees)",
                            "fire_impact": "Fire Impact (brightness * frp)"
                        },
                        opacity=0.7
                    )

                    fig.update_traces(marker=dict(size=8))
                    fig.update_layout(xaxis_range=[0, buffer_distance])

                else:
                    fig = px.scatter(
                        fires_within_buffer.dropna(subset=["distance_to_koala", "fire_impact"]),
                        x="distance_to_koala",
                        y="fire_impact",
                        title="Fire Intensity vs. Distance to Koala Sightings",
                        labels={
                            "distance_to_koala": "Distance to Koala Sighting (degrees)",
                            "fire_impact": "Fire Impact (brightness * frp)"
                        },
                        opacity=0.7,
                        trendline="lowess"
                    )

                    fig.update_traces(marker=dict(size=8))
                    fig.update_layout(xaxis_range=[0, buffer_distance])

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                The scatter plot reveals a critical finding:

                - Higher intensity fires tend to occur closer to koala sightings
                - This proximity poses a significant threat to koala populations

                This disturbing pattern explains the inverse correlation we observed between fire incidents and koala populations.
                When fires occur, they often impact areas with koala habitats, directly threatening these vulnerable marsupials.
                """)

                st.subheader("Fire Impact Summary Statistics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Fires Near Koalas", f"{len(fires_within_buffer):,}")

                with col2:
                    avg_impact = fires_within_buffer["fire_impact"].mean()
                    st.metric("Average Fire Intensity", f"{avg_impact:.1f}")

                with col3:
                    avg_dist = mean_distance * 111  
                    st.metric("Avg Distance (km)", f"{avg_dist:.2f}")

    else:
        st.error("Koala or Fire data not available. Please ensure data files are in the correct location.")

elif section == "Summary & Conclusion":
    st.header("📋 Summary & Conclusion")

    st.markdown("""
    ## Key Findings

    Our analysis of wildfire impacts on Australian biodiversity, particularly koala populations, has revealed several important insights:

    1. **Wildfire Patterns:**
       - Australia experiences significant wildfire activity, with peaks during 2011-2013
       - Fire activity shows an alternating pattern of high and low years

    2. **Koala Habitat Preferences:**
       - Koalas thrive in environments with moderate temperatures (below 28-29°C)
       - Higher rainfall areas show increased koala sightings
       - Coastal regions of eastern Australia provide optimal koala habitat conditions

    3. **Fire Impact on Koalas:**
       - Clear inverse relationship between fire incidents and koala sightings
       - Higher intensity fires tend to occur closer to koala habitats
       - Years with increased fire activity show decreased koala populations

    ## Implications for Biodiversity Conservation

    These findings have significant implications for biodiversity conservation efforts:

    - **Habitat Protection:** Preserving temperate forest areas with moderate rainfall is crucial for koala conservation
    - **Fire Management:** Strategic fire management in koala habitats should be prioritized
    - **Climate Monitoring:** Ongoing monitoring of local climate conditions can help predict threats to koala populations
    - **Rescue Operations:** During severe fire seasons, rescue and rehabilitation efforts should be intensified in key koala habitats

    ## Recommendations

    Based on our analysis, we recommend:

    1. **Enhanced Fire Monitoring:** Implement improved early warning systems for fires in critical koala habitats
    2. **Habitat Corridors:** Create and maintain wildlife corridors to allow koalas to escape fire-affected areas
    3. **Climate-Adaptive Management:** Develop conservation strategies that account for changing climate patterns
    4. **Data Integration:** Integrate biodiversity, weather, and fire data for more effective conservation planning
    5. **Community Engagement:** Involve local communities in koala habitat protection and fire prevention efforts

    ## Future Research Directions

    To build on this work, future research should:

    - Incorporate more detailed vegetation and habitat data
    - Study post-fire recovery patterns of koala populations
    - Analyze the genetic diversity impacts of population fragmentation due to fires
    - Develop predictive models for koala population changes based on fire and climate data
    """)

    st.info("""
    **About This Dashboard**

    This interactive dashboard was created as part of a data science hackathon addressing the challenge:

    *"Analyse pertinent data trends concerning the biodiversity sector to improve the sector's efficiency and abilities."*

    The analysis focuses on Australian koala populations as an indicator species for overall biodiversity health in relation to wildfire impacts.
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    Created for the Data Science Hackathon 2025<br>
    Data sources: Australian wildlife conservation databases, satellite fire monitoring, and weather records
</div>
""", unsafe_allow_html=True)