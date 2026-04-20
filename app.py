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
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Main Container */
    .main > div {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Headers */
.main-header {
    font-size: 3rem;
    font-weight: 800;
    color: #000000;
    text-align: left;
    padding: 1.5rem 0;
    border-bottom: 3px solid #000000;
    margin-bottom: 1rem;
}

.sub-header {
    font-size: 1.3rem;
    color: #666666;
    text-align: left;
    margin-bottom: 2rem;
    font-weight: 500;
}

    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 2px solid #eee;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── SECTION 4: HEADER DISPLAY ───────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ Smart Home Energy Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Load Forecasting & Conservation Advisor</div>', unsafe_allow_html=True)

# ─── SECTION 5: SIDEBAR NAVIGATION ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    st.markdown("---")
    page = st.radio(
        "Select Page",
        [
            "🏠 Home Dashboard",
            "📊 EDA & Visualizations",
            "🤖 Model Training",
            "🔍 Anomaly Detection",
            "📈 Model Comparison"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📋 Project Info")
    st.info("""
    **Subject:** Python for Data Science  
    **Team:** SY ECE A1  
    **Dataset:** HomeC (UCI/Kaggle)  
    **Models:** LR, Ridge, Random Forest
    """)
    
    # Add a visual element
    st.markdown("---")
    st.markdown("### 🌟 Quick Stats")
    st.metric("App Version", "2.0")
    st.metric("Last Updated", "2026")

# ─── SECTION 6: DATA LOADING & PREPROCESSING ─────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('HomeC_sample.csv')
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['hour'] = df['time'].dt.hour
            df['month'] = df['time'].dt.month
            df['dayofweek'] = df['time'].dt.dayofweek
        else:
            df['hour'] = np.random.randint(0, 24, len(df))
            df['month'] = np.random.randint(1, 13, len(df))
            df['dayofweek'] = np.random.randint(0, 7, len(df))

        numeric_cols = ['use [kW]', 'gen [kW]', 'temperature', 'humidity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if 'use [kW]' in df.columns and 'gen [kW]' in df.columns:
            df['net_consumption'] = df['use [kW]'] - df['gen [kW]']
            
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'HomeC_sample.csv' not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

df = load_data()

# ─── SECTION 7: PAGE LOGIC ───────────────────────────────────────────────────
if df is not None:

    # ─── PAGE 1: HOME DASHBOARD ──────────────────────────────────────────────
    if page == "🏠 Home Dashboard":
        # Project Overview Cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>🔋 Task 1: Smart Grid Load Balancing</h3>
                <ul>
                    <li><strong>Type:</strong> Supervised Regression</li>
                    <li><strong>Goal:</strong> Predict energy consumption</li>
                    <li><strong>Models:</strong> LR, Ridge, Random Forest</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h3>🚨 Task 2: Electricity Theft Detection</h3>
                <ul>
                    <li><strong>Type:</strong> Anomaly Detection</li>
                    <li><strong>Goal:</strong> Identify abnormal patterns</li>
                    <li><strong>Method:</strong> IQR Statistical Method</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Key Metrics with Custom Cards
        st.subheader("📊 Dataset Overview")
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
        st.subheader("📄 Dataset Sample")
        st.dataframe(df.head(10), use_container_width=True)

    # ─── PAGE 2: EDA ─────────────────────────────────────────────────────────
    elif page == "EDA":
            # ─── PAGE 2: EDA ─────────────────────────────────────────────────────────
    elif page == "EDA":
        st.markdown("---")
        
        # DEBUG: Show data info
        st.write("🔍 **Debug Info:**")
        st.write(f"Data loaded: {df is not None}")
        st.write(f"Data shape: {df.shape if df is not None else 'N/A'}")
        st.write(f"Available columns: {df.columns.tolist() if df is not None else 'N/A'}")
        
        # Check if required columns exist
        required_cols = ['use [kW]', 'temperature', 'hour', 'month']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
            st.info(f"Available: {df.columns.tolist()}")
        else:
            st.success("✅ All required columns found!")
        
        st.markdown("---")
        
        st.info("📊 **Exploratory Data Analysis** — Understanding data patterns before building models")
        
        # Check if data exists
        if df is None:
            st.error("❌ Data not loaded. Please check if HomeC_sample.csv exists.")
            st.stop()
        
        # Check required columns
        required_cols = ['use [kW]', 'temperature', 'hour', 'month']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            st.error(f"❌ Missing columns: {missing}")
            st.write("Available columns:", df.columns.tolist())
            st.stop()
        
        st.success("✅ Data loaded successfully!")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Trends"])
        
        # TAB 1: DISTRIBUTION
        with tab1:
            st.write("### Energy Consumption Distribution")
            st.write("Shows how energy consumption values are distributed across all records.")
            
            try:
                fig = px.histogram(df, x='use [kW]', nbins=50, title="Energy Consumption Distribution")
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Chart rendered successfully!")
            except Exception as e:
                st.error(f"❌ Chart error: {e}")
        
        # TAB 2: CORRELATION
        with tab2:
            st.write("### Feature Correlation")
            st.write("Shows relationships between features (Red = positive, Blue = negative).")
            
            try:
                key_features = ['use [kW]', 'temperature', 'humidity', 'hour', 'month']
                corr_df = df[key_features]
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
                st.pyplot(fig)
                st.success("✅ Chart rendered successfully!")
            except Exception as e:
                st.error(f"❌ Chart error: {e}")
        
        # TAB 3: TRENDS
        with tab3:
            st.write("### Hourly Consumption")
            st.write("Average consumption for each hour of the day.")
            
            try:
                hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
                fig = px.bar(hourly, x='hour', y='use [kW]', title="Hourly Consumption")
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Chart rendered successfully!")
            except Exception as e:
                st.error(f"❌ Chart error: {e}")
            
            st.write("---")
            st.write("### Monthly Consumption")
            st.write("Average consumption across months.")
            
            try:
                monthly = df.groupby('month')['use [kW]'].mean().reset_index()
                fig = px.line(monthly, x='month', y='use [kW]', markers=True, title="Monthly Consumption")
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Chart rendered successfully!")
            except Exception as e:
                st.error(f"❌ Chart error: {e}")

    # ─── PAGE 3: MODEL TRAINING ──────────────────────────────────────────────
    elif page == "🤖 Model Training":
        st.header("🤖 Machine Learning Model Training")
        
        features = ['temperature', 'humidity', 'hour', 'month', 'dayofweek']
        target = 'use [kW]'
        
        missing_cols = [col for col in features + [target] if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.info("Available: " + ", ".join(df.columns.tolist()))
        else:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            model_choice = st.selectbox("Select Algorithm", 
                                        ["Linear Regression", "Ridge Regression", "Random Forest"])
            
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

            st.markdown("### 📏 Model Performance")
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

            st.subheader("Actual vs Predicted Values")
            result_df = pd.DataFrame({'Actual': y_test.values[:200], 'Predicted': y_pred[:200]})
            fig = px.line(result_df, title="Actual vs Predicted (First 200 Samples)",
                          color_discrete_sequence=['#667eea', '#764ba2'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            if model_choice == "Random Forest":
                st.subheader("Feature Importance")
                fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                fi = fi.sort_values('Importance', ascending=True)
                fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                             color_discrete_sequence=['#667eea'])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

    # ─── PAGE 4: ANOMALY DETECTION ───────────────────────────────────────────
    elif page == "🔍 Anomaly Detection":
        st.header("🔍 Electricity Theft / Anomaly Detection")
        
        st.markdown("""
        <div class="info-box">
            <strong>Method:</strong> IQR (Interquartile Range) — Outliers may indicate theft or leakage
        </div>
        """, unsafe_allow_html=True)

        Q1 = df['use [kW]'].quantile(0.25)
        Q3 = df['use [kW]'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df['anomaly'] = ((df['use [kW]'] < lower) | (df['use [kW]'] > upper))
        anomalies = df[df['anomaly'] == True]
        normal = df[df['anomaly'] == False]

        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="metric-label">🚨 Anomalies</div>
                <div class="metric-value">{len(anomalies):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="metric-label">✅ Normal</div>
                <div class="metric-value">{len(normal):,}</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Anomaly Visualization")
        plot_df = df.head(1000).reset_index(drop=True)
        fig = px.scatter(plot_df, x=plot_df.index, y='use [kW]',
                         color='anomaly', 
                         color_discrete_map={False: '#667eea', True: '#f5576c'},
                         title="Energy Consumption (Red = Anomaly)")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sample Anomalous Records")
        st.dataframe(anomalies[['time', 'use [kW]', 'hour', 'temperature']].head(10), 
                     use_container_width=True)

    # ─── PAGE 5: MODEL COMPARISON ────────────────────────────────────────────
    elif page == "📈 Model Comparison":
        st.header("📈 Algorithm Comparison")
        st.markdown("""
        <div class="info-box">
            Comparing all three models on the same test set to determine the best performer
        </div>
        """, unsafe_allow_html=True)

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

        fig = px.bar(comparison_df, x='Model', y='R²', color='R²',
                     color_continuous_scale='Blues', 
                     title="R² Score Comparison (Higher is Better)")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        best = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        st.markdown(f"""
        <div class="success-box">
            <h3>🏆 Best Performing Model: {best}</h3>
        </div>
        """, unsafe_allow_html=True)

# ─── SECTION 8: FOOTER ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🚀 Deployed on Streamlit Cloud | 🎓 Python for Data Science Project | © 2026 Smart Home Energy Analytics</p>
</div>
""", unsafe_allow_html=True)
