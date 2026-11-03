"""Entry point: ``uv run streamlit run app/streamlit_app.py``."""

import streamlit as st
from common import apply_style

st.set_page_config(
    page_title="Coronary artery disease risk",
    page_icon=":material/cardiology:",
    layout="wide",
)
apply_style()

navigation = st.navigation(
    [
        st.Page("predict.py", title="Risk estimate", icon=":material/monitor_heart:", default=True),
        st.Page("explore.py", title="Explore the data", icon=":material/query_stats:"),
        st.Page("performance.py", title="Model performance", icon=":material/fact_check:"),
        st.Page("about.py", title="About the project", icon=":material/info:"),
    ],
    position="top",
)
navigation.run()
