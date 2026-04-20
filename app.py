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
    .stApp {
        background-color: #ffffff;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #000000;
        text-align: left;
        padding: 1.5rem 0;
        border-bottom: 3px solid #000000;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        text-align: left;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #ffffff;
        border: 2px solid #e5e5e5;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #000000;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .info-box {
        background-color: #000000;
        border-radius: 12px;
        padding: 1.5rem;
        color: #ffffff;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #ffffff;
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 1.5rem;
        color: #000000;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #f0fff4;
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        color: #000000;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        color: #999999;
        font-size: 0.85rem;
        border-top: 1px solid #e5e5e5;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── SECTION 4: HEADER DISPLAY ───────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ Smart Home Energy Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered load forecasting and conservation advisor for smart grid optimization</div>', unsafe_allow_html=True)

# ─── SECTION 5: SIDEBAR NAVIGATION ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### Menu")
    st.markdown("---")
    page = st.radio(
        "Select Page",
        ["Home", "EDA", "Model Training", "Anomaly Detection", "Model Comparison"]
    )
    st.markdown("---")
    st.markdown("### Project Details")
    st.markdown("- **Subject:** Python for Data Science")
    st.markdown("- **Team:** SY ECE A1")
    st.markdown("- **Dataset:** HomeC (Kaggle)")
    st.markdown("- **Models:** LR, Ridge, RF")
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔋 Task 1: Smart Grid Load Balancing")
            st.info("**Type:** Supervised Regression\n\n**Goal:** Predict energy consumption\n\n**Models:** LR, Ridge, Random Forest")
        with col2:
            st.markdown("#### 🚨 Task 2: Electricity Theft Detection")
            st.warning("**Type:** Anomaly Detection\n\n**Goal:** Identify abnormal patterns\n\n**Method:** IQR Statistical")

        st.markdown("---")
        st.markdown("### Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Avg Consumption", f"{df['use [kW]'].mean():.2f} kW")
        c3.metric("Max Consumption", f"{df['use [kW]'].max():.2f} kW")
        c4.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")

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
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### Feature Correlation Heatmap")
            key_features = ['use [kW]', 'gen [kW]', 'temperature', 'humidity', 'hour', 'month', 'dayofweek']
            correlation_df = df[key_features]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
            st.pyplot(fig)
            
            st.success("**Key Insights:** Temperature & Humidity affect consumption | Hour shows usage patterns")

        with tab3:
            st.markdown("### Hourly Consumption")
            hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
            fig = px.bar(hourly, x='hour', y='use [kW]', color_discrete_sequence=['#000000'])
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
            c1.metric("MAE", f"{mae:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")
            c3.metric("R²", f"{r2:.4f}")

            st.markdown("---")
            st.markdown("### Actual vs Predicted")
            result_df = pd.DataFrame({'Actual': y_test.values[:200], 'Predicted': y_pred[:200]})
            fig = px.line(result_df, color_discrete_sequence=['#000000', '#666666'])
            st.plotly_chart(fig, use_container_width=True)
            
            if model_choice == "Random Forest":
                st.markdown("---")
                st.markdown("### Feature Importance")
                fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                fi = fi.sort_values('Importance', ascending=True)
                fig = px.bar(fi, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#000000'])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Required columns not found in dataset")

    # ─── PAGE 4: ANOMALY DETECTION ───────────────────────────────────────────
    elif page == "Anomaly Detection":
        st.markdown("---")
        
        st.info("🔍 **Detection Method:** IQR (Interquartile Range) — Outliers may indicate electricity theft")

        Q1 = df['use [kW]'].quantile(0.25)
        Q3 = df['use [kW]'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df['anomaly'] = ((df['use [kW]'] < lower) | (df['use [kW]'] > upper))
        anomalies = df[df['anomaly'] == True]
        normal = df[df['anomaly'] == False]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("🚨 Anomalies", f"{len(anomalies):,}")
        c3.metric("✅ Normal", f"{len(normal):,}")

        st.markdown("---")
        st.markdown("### Anomaly Visualization")
        plot_df = df.head(1000).reset_index(drop=True)
        fig = px.scatter(plot_df, x=plot_df.index, y='use [kW]', color='anomaly',
                         color_discrete_map={False: '#000000', True: '#dc3545'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Sample Anomalous Records")
        st.dataframe(anomalies[['time', 'use [kW]', 'hour']].head(10), use_container_width=True)

    # ─── PAGE 5: MODEL COMPARISON ────────────────────────────────────────────
    elif page == "Model Comparison":
        st.markdown("---")
        
        st.info("📊 **Model Comparison:** Comparing all three models on the same test set")

        features = ['temperature', 'humidity', 'hour', 'month', 'dayofweek']
        target = 'use [kW]'
        X = df[features]; y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        results = {}
        models = [
            ("Linear Regression", LinearRegression(), True),
            ("Ridge Regression", Ridge(alpha=1.0), True),
            ("Random Forest", RandomForestRegressor(n_estimators=50, random_state=42), False)
        ]

        for name, model, use_scale in models:
            if use_scale:
                model.fit(X_train_sc, y_train)
                pred = model.predict(X_test_sc)
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            
            results[name] = {
                "MAE": round(mean_absolute_error(y_test, pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, pred)), 4),
                "R²": round(r2_score(y_test, pred), 4)
            }

        comparison_df = pd.DataFrame(results).T.reset_index()
        comparison_df.columns = ['Model', 'MAE', 'RMSE', 'R²']
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown("---")
        fig = px.bar(comparison_df, x='Model', y='R²', color='R²', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

        best = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        st.success(f"🏆 **Best Performing Model: {best}**")

# ─── SECTION 8: FOOTER ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="footer">Deployed on Streamlit Cloud | Python for Data Science Project | 2024</div>', unsafe_allow_html=True)
