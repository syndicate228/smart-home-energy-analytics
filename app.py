# IMPORTS
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

# PAGE CONFIG
st.set_page_config(page_title="Smart Home Energy Analytics", page_icon="⚡", layout="wide")

# CSS STYLING
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #000000; padding: 1rem 0; border-bottom: 3px solid #000000; }
    .metric-card { background-color: #ffffff; border: 2px solid #e5e5e5; border-radius: 12px; padding: 1.5rem; }
    .info-box { background-color: #000000; border-radius: 12px; padding: 1.5rem; color: #ffffff; margin: 1rem 0; }
    .footer { text-align: center; margin-top: 3rem; padding: 2rem; color: #999999; border-top: 1px solid #e5e5e5; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="main-header">⚡ Smart Home Energy Analytics</div>', unsafe_allow_html=True)
st.write("AI-Powered Load Forecasting & Conservation Advisor")

# SIDEBAR
with st.sidebar:
    st.write("### Menu")
    st.markdown("---")
    page = st.radio("Select Page", ["Home", "EDA", "Model Training", "Anomaly Detection", "Model Comparison"])
    st.markdown("---")
    st.write("### Project Info")
    st.write("- Subject: Python for Data Science")
    st.write("- Team: SY ECE A1")
    st.write("- Dataset: HomeC (Kaggle)")
    st.write("- Models: LR, Ridge, RF")

# DATA LOADING
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('HomeC_sample.csv')
        
        # Rename columns if needed
        df = df.rename(columns={
            'use_kW': 'use [kW]', 'gen_kW': 'gen [kW]',
            'Temperature': 'temperature', 'Humidity': 'humidity'
        })
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['hour'] = df['time'].dt.hour
            df['month'] = df['time'].dt.month
            df['dayofweek'] = df['time'].dt.dayofweek
        
        for col in ['use [kW]', 'gen [kW]', 'temperature', 'humidity']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if 'use [kW]' in df.columns and 'gen [kW]' in df.columns:
            df['net_consumption'] = df['use [kW]'] - df['gen [kW]']
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = load_data()

# CHECK DATA
if df is None:
    st.error("❌ Data failed to load. Check if HomeC_sample.csv exists in GitHub repo.")
    st.stop()

# DEBUG INFO
st.markdown("---")
with st.expander("🔍 Debug Info (Click to expand)"):
    st.write(f"**Data Shape:** {df.shape}")
    st.write(f"**Columns:** {df.columns.tolist()}")
    st.write(f"**Sample:**")
    st.dataframe(df.head(3))

# PAGE LOGIC
if page == "Home":
    st.markdown("---")
    st.info("🔋 **Task 1:** Load Balancing (Regression) | 🚨 **Task 2:** Anomaly Detection (IQR)")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg Consumption", f"{df['use [kW]'].mean():.2f} kW")
    c3.metric("Max Consumption", f"{df['use [kW]'].max():.2f} kW")
    c4.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
    
    st.markdown("---")
    st.write("### Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "EDA":
    st.markdown("---")
    st.info("📊 **Exploratory Data Analysis** — Understanding data patterns before building models")
    
    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Trends"])
    
    with tab1:
        st.write("### Energy Consumption Distribution")
        st.write("Shows how consumption values are distributed. Most values cluster between 0.5-2.0 kW.")
        fig = px.histogram(df, x='use [kW]', nbins=50, title="Consumption Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Statistics:**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{df['use [kW]'].mean():.2f} kW")
        c2.metric("Median", f"{df['use [kW]'].median():.2f} kW")
        c3.metric("Std Dev", f"{df['use [kW]'].std():.2f} kW")
    
    with tab2:
        st.write("### Feature Correlation")
        st.write("Red = positive correlation, Blue = negative correlation")
        
        cols = ['use [kW]', 'temperature', 'humidity', 'hour', 'month']
        cols = [c for c in cols if c in df.columns]
        
        if len(cols) > 1:
            corr_df = df[cols]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
            st.pyplot(fig)
            st.success("✅ Temperature and hour are strong predictors!")
        else:
            st.error("Not enough columns for correlation")
    
    with tab3:
        st.write("### Hourly Consumption")
        st.write("Peak hours: 6-9 AM and 5-10 PM")
        
        if 'hour' in df.columns and 'use [kW]' in df.columns:
            hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
            fig = px.bar(hourly, x='hour', y='use [kW]', title="Hourly Pattern")
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("---")
        st.write("### Monthly Consumption")
        st.write("Summer = higher (AC), Winter = elevated (heating)")
        
        if 'month' in df.columns and 'use [kW]' in df.columns:
            monthly = df.groupby('month')['use [kW]'].mean().reset_index()
            fig = px.line(monthly, x='month', y='use [kW]', markers=True, title="Monthly Pattern")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Model Training":
    st.markdown("---")
    st.write("### Model Training")
    
    features = ['temperature', 'humidity', 'hour', 'month', 'dayofweek']
    features = [f for f in features if f in df.columns]
    target = 'use [kW]'
    
    if target in df.columns and len(features) >= 2:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_choice = st.selectbox("Select Model", ["Linear Regression", "Ridge", "Random Forest"])
        
        if model_choice == "Linear Regression":
            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict(X_test)
        elif model_choice == "Ridge":
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            pred = model.predict(X_test)
        else:
            model = RandomForestRegressor(n_estimators=50)
            model.fit(X, y)
            pred = model.predict(X_test)
        
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        
        c1, c2 = st.columns(2)
        c1.metric("R² Score", f"{r2:.4f}")
        c2.metric("MAE", f"{mae:.4f}")
        
        st.success(f"✅ {model_choice} trained successfully!")
    else:
        st.error("Missing required columns for training")

elif page == "Anomaly Detection":
    st.markdown("---")
    st.info("🔍 **Method:** IQR (Interquartile Range) — Outliers may indicate theft")
    
    if 'use [kW]' in df.columns:
        Q1 = df['use [kW]'].quantile(0.25)
        Q3 = df['use [kW]'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df['anomaly'] = ((df['use [kW]'] < lower) | (df['use [kW]'] > upper))
        anomalies = df[df['anomaly'] == True]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"{len(df):,}")
        c2.metric("🚨 Anomalies", f"{len(anomalies):,}")
        c3.metric("✅ Normal", f"{len(df) - len(anomalies):,}")
        
        fig = px.scatter(df.head(1000), y='use [kW]', color='anomaly', title="Anomaly Detection")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Column 'use [kW]' not found")

elif page == "Model Comparison":
    st.markdown("---")
    st.write("### Model Comparison")
    
    features = ['temperature', 'humidity', 'hour', 'month']
    features = [f for f in features if f in df.columns]
    target = 'use [kW]'
    
    if target in df.columns and len(features) >= 2:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        results['Linear'] = r2_score(y_test, lr.predict(X_test))
        
        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        results['Ridge'] = r2_score(y_test, ridge.predict(X_test))
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=50)
        rf.fit(X_train, y_train)
        results['RF'] = r2_score(y_test, rf.predict(X_test))
        
        # Display
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'R² Score'])
        st.dataframe(results_df, use_container_width=True)
        
        fig = px.bar(results_df, x='Model', y='R² Score', color='R² Score', title="Model Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        best = max(results, key=results.get)
        st.success(f"🏆 Best Model: {best}")
    else:
        st.error("Missing required columns")

# FOOTER
st.markdown("---")
st.markdown('<div class="footer">Deployed on Streamlit Cloud | Python for Data Science Project | 2024</div>', unsafe_allow_html=True)
