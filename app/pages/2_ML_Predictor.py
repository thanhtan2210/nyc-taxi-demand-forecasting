import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys
import plotly.express as px
from google.cloud import bigquery
from dotenv import load_dotenv

# Robust path setup
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path)

from src.inference import DemandPredictor

st.set_page_config(page_title="AI Forecast Center", page_icon="🔮", layout="wide")

# --- CSS for Professional Look ---
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight:bold; color: #00D1FF; }
    .prediction-box { background: #1E2130; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #00D1FF; }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Travel Demand Forecaster")
st.markdown("Easily predict taxi demand for any NYC neighborhood using our trained AI models.")

# --- Data Acquisition: Neighborhood Map ---
@st.cache_data
def get_zone_map():
    client = bigquery.Client()
    query = f"SELECT Zone, Location_ID FROM `{os.getenv('BQ_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID', 'nyc_taxi_dw')}.Dim_Location` ORDER BY Zone"
    return client.query(query).to_dataframe()

zones_df = get_zone_map()

# --- UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📍 Where & When?")
    selected_zone = st.selectbox("Neighborhood", options=zones_df['Zone'].tolist())
    zone_id = zones_df[zones_df['Zone'] == selected_zone]['Location_ID'].iloc[0]
    selected_date = st.date_input("Date")
    time_str = st.text_input("Enter Time (HH:MM:SS)", value="12:00:00")
    try:
        selected_time = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        st.error("Invalid format! Use HH:MM:SS.")
        selected_time = datetime.time(12, 0, 0)

with col2:
    st.subheader("🌤️ Weather conditions")
    weather_temp = st.number_input("Temperature (°C)", value=20.0, step=0.5)
    weather_precip = st.number_input("Precipitation (mm)", min_value=0.0, value=0.0, step=0.1)

st.markdown("---")

# --- Inference (Exactly your requested logic and map coloring) ---
if st.button("🚀 Run Prediction"):
    dt = datetime.datetime.combine(selected_date, selected_time)
    input_data = pd.DataFrame([{
        'PULocation_Key': zone_id, 'total_demand': 50, 'Hour': dt.hour,
        'DayOfWeek': dt.weekday(), 'Is_Weekend': 1 if dt.weekday() >= 5 else 0,
        'hour_sin': np.sin(dt.hour * (2. * np.pi / 24)),
        'hour_cos': np.cos(dt.hour * (2. * np.pi / 24)),
        'lag_1h': 45, 'lag_2h': 40, 'lag_24h': 110, 'lag_168h': 105,
        'rolling_mean_6h': 55, 
        'Temperature': weather_temp,
        'Precipitation': weather_precip
    }])
    
    try:
        model = DemandPredictor()
        pred = int(model.predict(input_data)[0])
        st.markdown(f"""
            <div class='prediction-box'>
                <p>Predicted demand for <b>{selected_zone}</b> on {selected_date}:</p>
                <p class='big-font'>{pred} trips</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Add Map Visualization (Restored and Kept Coloring)
        st.subheader("📍 Prediction Location Map")
        
        # Load coordinates from CSV
        coord_file_path = os.path.join(root_dir, 'dataset/taxi_zone_lookup_cordinates/taxi_zone_lookup_coordinates.csv')
        df_coords = pd.read_csv(coord_file_path)
        
        # Get coordinates for the selected zone
        zone_info = df_coords[df_coords['Zone'] == selected_zone]
        
        if not zone_info.empty:
            lat = zone_info['latitude'].iloc[0]
            lon = zone_info['longitude'].iloc[0]
        else:
            lat, lon = 40.7128, -74.0060
            
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon], 'Zone': [selected_zone], 'Pred': [pred]})
        fig_map = px.scatter_mapbox(
            map_df, lat="lat", lon="lon", size=[20],
            zoom=12, height=400, mapbox_style="open-street-map", template="plotly_dark"
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
