"""Streamlit entry point: `streamlit run app.py`."""
import streamlit as st

st.set_page_config(page_title="NYC ride demand", layout="wide")

pages = [
    st.Page("src/app_pages/overview.py", title="Overview", default=True),
    st.Page("src/app_pages/patterns.py", title="Patterns"),
    st.Page("src/app_pages/forecast.py", title="Forecast"),
]
st.navigation(pages).run()
