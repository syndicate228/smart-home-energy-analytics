# ─── SECTION 1: IMPORTS ──────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ─── SECTION 2: PAGE CONFIGURATION ───────────────────────────────────────────
st.set_page_config(
    page_title="Smart Home Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── SECTION 3: CUSTOM CSS STYLING ───────────────────────────────────────────
st.markdown("""
<style>
    /* Import Clean Font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Main Background */
    .stApp {
        background-color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Main Container */
    .main > div {
        background-color: #ffffff;
        padding: 0;
        max-width: 100%;
    }
    
    /* Main Header - Bold & Large */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #000000;
        text-align: left;
        padding: 2.5rem 0 1rem 0;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    
    .sub-header {
        font-size: 1.25rem;
        color: #666666;
        text-align: left;
        margin-bottom: 2.5rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 800px;
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #000000;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #fafafa;
        border-right: 1px solid #e5e5e5;
    }
    
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    
    /* Navigation */
    .stRadio > label {
        color: #333333;
        font-weight: 500;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        transition: all 0.2s ease;
    }
    
    .stRadio > label:hover {
        background-color: #f0f0f0;
    }
    
    /* Metric Cards - DopplePress Style */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 16px;
        padding: 2rem;
        text-align: left;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        height: 100%;
    }
    
    .metric-card:hover {
        border-color: #000000;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
        margin: 0.75rem 0;
        letter-spacing: -0.5px;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #000000;
        border-radius: 12px;
        padding: 2rem;
        color: #ffffff;
        margin: 1.5rem 0;
    }
    
    .info-box h4 {
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .info-box p {
        color: #cccccc;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background-color: #ffffff;
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 2rem;
        color: #000000;
        margin: 1.5rem 0;
    }
    
    .warning-box h4 {
        color: #dc3545;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background-color: #ffffff;
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 2rem;
        color: #000000;
        margin: 1.5rem 0;
    }
    
    .success-box h4 {
        color: #28a745;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #000000;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.2s ease;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        background-color: #333333;
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 2px solid #e5e5e5;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
        font-weight: 500;
        color: #666666;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #000000;
        background-color: #f5f5f5;
    }
    
    .stTabs [aria-selected="true"] {
        color: #000000;
        background-color: #ffffff;
        border-bottom: 2px solid #ffffff;
        font-weight: 600;
        border: 1px solid #e5e5e5;
        border-bottom: none;
    }
    
    /* DataFrames */
    .dataframe {
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 5rem;
        padding: 3rem;
        color: #999999;
        font-size: 0.85rem;
        border-top: 1px solid #e5e5e5;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e5e5e5;
        margin: 2.5rem 0;
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        border: 1px solid #e5e5e5;
        border-radius: 8px;
    }
    
    /* Slider */
    .stSlider > div {
        color: #000000;
    }
    
    /* Plotly Charts Container */
    .chart-container {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── SECTION 4: HEADER DISPLAY ───────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ Smart Home Energy Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered load forecasting and conservation advisor for smart grid optimization. Predict consumption, detect anomalies, and optimize energy usage.</div>', unsafe_allow_html=True)

# ─── SECTION 5: SIDEBAR NAVIGATION ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### Menu", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Select Page",
        [
            "Home",
            "EDA",
            "Model Training",
            "Anomaly Detection",
            "Model Comparison"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### Project Details")
    st.markdown("""
    - **Subject:** Python for Data Science
    - **Team:** SY ECE A1
    - **Dataset:** HomeC (UCI/Kaggle)
    - **Models:** LR, Ridge, RF
    """)
    st.markdown("---")
    st.caption("v2.0 | 2024")

# ─── SECTION 6: DATA LOADING ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('HomeC_sample.csv')
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['hour'] = df['time'].dt.hour
            df['month'] = df['time'].dt.month
            df['dayofweek'] = df['time'].dt.dayofweek
        
        numeric_cols = ['use [kW]', 'gen [kW]', 'temperature', 'humidity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if 'use [kW]' in df.columns and 'gen [kW]' in df.columns:
            df['net_consumption'] = df['use [kW]'] - df['gen [kW]']
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# ─── SECTION 7: PAGE LOGIC ───────────────────────────────────────────────────
if df is not None:

    # ─── PAGE 1: HOME ────────────────────────────────────────────────────────
    if page == "Home":
        st.markdown("---")
        
        # Project Overview Cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>🔋 Task 1: Smart Grid Load Balancing</h4>
                <p><strong>Type:</strong> Supervised Regression</p>
                <p><strong>Goal:</strong> Predict energy consumption based on environmental and usage factors</p>
                <p><strong>Models:</strong> Linear Regression, Ridge, Random Forest</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>🚨 Task 2: Electricity Theft Detection</h4>
                <p><strong>Type:</strong> Anomaly Detection</p>
                <p><strong>Goal:</strong> Identify abnormal consumption patterns</p>
                <p><strong>Method:</strong> IQR Statistical Method</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Key Metrics
        st.markdown("### Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Consumption</div>
                <div class="metric-value">{df['use [kW]'].mean():.2f} kW</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Consumption</div>
                <div class="metric-value">{df['use [kW]'].max():.2f} kW</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Temperature</div>
                <div class="metric-value">{df['temperature'].mean():.1f}°C</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Dataset Sample")
        st.dataframe(df.head(10), use_container_width=True)

    # ─── PAGE 2: EDA ─────────────────────────────────────────────────────────
    elif page == "EDA":
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Trends"])

        with tab1:
            st.markdown("### Energy Consumption Distribution")
            fig = px.histogram(df, x='use [kW]', nbins=50, color_discrete_sequence=['#000000'])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### Feature Correlation Heatmap")
            key_features = ['use [kW]', 'gen [kW]', 'temperature', 'humidity', 'hour', 'month', 'dayofweek']
            correlation_df = df[key_features]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
            st.pyplot(fig)
            
            st.markdown("""
            <div class="success-box">
                <h4>Key Insights</h4>
                <p>• Temperature & Humidity show correlation with energy consumption</p>
                <p>• Hour of day reveals usage patterns (peak vs off-peak)</p>
                <p>• Net consumption = Usage minus Generation</p>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("### Hourly Consumption")
            hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
            fig = px.bar(hourly, x='hour', y='use [kW]', color_discrete_sequence=['#000000'])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─── PAGE 3: MODEL TRAINING ──────────────────────────────────────────────
    elif page == "Model Training":
        st.markdown("---")
        
        features = ['temperature', 'humidity', 'hour', 'month', 'dayofweek']
        target = 'use [kW]'
        
        if all(f in df.columns for f in features) and target in df.columns:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            model_choice = st.selectbox("Select Algorithm", ["Linear Regression", "Ridge Regression", "Random Forest"])
            
            if model_choice == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
            elif model_choice == "Ridge Regression":
                alpha = st.slider("Ridge Alpha", 0.01, 10.0, 1.0)
                model = Ridge(alpha=alpha)
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
            else:
                n_trees = st.slider("Number of Trees", 10, 200, 100, step=10)
                model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.markdown("### Model Performance")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MAE (Error)</div>
                    <div class="metric-value">{mae:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">RMSE (Error)</div>
                    <div class="metric-value">{rmse:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">R² (Accuracy)</div>
                    <div class="metric-value">{r2:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Actual vs Predicted")
            result_df = pd.DataFrame({'Actual': y_test.values[:200], 'Predicted': y_pred[:200]})
            fig = px.line(result_df, color_discrete_sequence=['#000000', '#666666'])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if model_choice == "Random Forest":
                st.markdown("---")
                st.markdown("### Feature
