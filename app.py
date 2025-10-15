"""
Unified Trading Dashboard
Main application with Stock Analysis, Screener, Watchlist, and Comparison
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 20px;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Main navigation
st.markdown('<h1 class="main-header">Trading Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Stock Analysis & Trading Tools</p>', unsafe_allow_html=True)

# Page selection
page = st.sidebar.radio(
    "Navigate to:",
    ["Stock Analysis", "Stock Screener", "Watchlist", "Stock Comparison"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Stock Analysis**: Deep dive into individual stocks\n\n"
    "**Stock Screener**: Filter stocks by criteria\n\n"
    "**Watchlist**: Monitor favorite stocks\n\n"
    "**Stock Comparison**: Compare stocks side-by-side"
)

# Load the selected page
if page == "Stock Analysis":
    # Import and run main dashboard
    import subprocess
    import sys
    st.info("The main Stock Analysis dashboard runs separately.")
    st.markdown("### To use Stock Analysis:")
    st.code("streamlit run live_dashboard.py", language="bash")
    st.markdown("Or click the button below:")
    if st.button("Open Stock Analysis Dashboard"):
        st.info("Please run: streamlit run live_dashboard.py")

elif page == "Stock Screener":
    import stock_screener
    stock_screener.main()

elif page == "Watchlist":
    import watchlist
    watchlist.main()

elif page == "Stock Comparison":
    import stock_comparison
    stock_comparison.main()

