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
    /* Main Background - Clean White */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Main Container */
    .main > div {
        background-color: #ffffff;
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Headers - Clean & Bold */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: left;
        padding: 2rem 0 1rem 0;
        border-bottom: 2px solid #000000;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666666;
        text-align: left;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Sidebar - Minimalist */
    .css-1d391kg {
        background-color: #fafafa;
        border-right: 1px solid #e5e5e5;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: left;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #fafafa;
        border-left: 3px solid #1a1a1a;
        border-radius: 8px;
        padding: 1.5rem;
        color: #333333;
        margin: 1.5rem 0;
    }
    
    .warning-box {
        background-color: #fff9f9;
        border-left: 3px solid #dc3545;
        border-radius: 8px;
        padding: 1.5rem;
        color: #333333;
        margin: 1.5rem 0;
    }
    
    .success-box {
        background-color: #f8fff9;
        border-left: 3px solid #28a745;
        border-radius: 8px;
        padding: 1.5rem;
        color: #333333;
        margin: 1.5rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #ffffff;
        color: #1a1a1a;
        border: 1px solid #1a1a1a;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        border-bottom: 1px solid #e5e5e5;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1a1a1a;
        border-bottom: 2px solid #1a1a1a;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        color: #999999;
        font-size: 0.85rem;
        border-top: 1px solid #e5e5e5;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── SECTION 4: HEADER DISPLAY ───────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ Smart Home Energy Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Load Forecasting & Conservation Advisor for Smart Grid Optimization</div>', unsafe_allow_html=True)

# ─── SECTION 5: SIDEBAR NAVIGATION ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation", unsafe_allow_html=True)
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
    st.markdown("### Project Information")
    st.markdown("""
    - **Subject:** Python for Data Science
    - **Team:** SY ECE A1
    - **Dataset:** HomeC (Kaggle)
    - **Models:** LR, Ridge, RF
    """)
    st.markdown("---")
    st.markdown("### Version")
    st.caption("v2.0 | 2026")

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
        
        # Key Metrics with Clean Cards
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

    # ─── PAGE 2: EDA & VISUALIZATIONS ────────────────────────────────────────
    elif page == "📊 EDA & Visualizations":
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Time Trends"])

        with tab1:
            st.markdown("### Energy Consumption Distribution")
            fig = px.histogram(df, x='use [kW]', nbins=50, 
                               color_discrete_sequence=['#1a1a1a'],
                               title="Distribution of Energy Consumption (kW)")
            fig.update_layout(
                plot_bgcolor='rgba(255,255,255,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(family='Inter', size=12, color='#333333'),
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### Feature Correlation Heatmap")
            
            key_features = ['use [kW]', 'gen [kW]', 'temperature', 'humidity', 
                            'hour', 'month', 'dayofweek', 'net_consumption']
            
            correlation_df = df[key_features]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', 
                        ax=ax, fmt='.2f', square=True, linewidths=0.5,
                        cbar_kws={'shrink': 0.8})
            plt.title('Correlation Matrix - Key Features', fontsize=14, pad=20, fontweight='600')
            st.pyplot(fig)
            
            st.markdown("""
            <div class="success-box">
                <strong>Key Insights:</strong>
                <ul>
                    <li>Temperature & Humidity show correlation with energy consumption</li>
                    <li>Hour of day reveals usage patterns (peak vs off-peak)</li>
                    <li>Net consumption = Usage minus Generation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("### Consumption Trends")
            col_a, col_b = st.columns(2)
            with col_a:
                hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
                fig = px.bar(hourly, x='hour', y='use [kW]', 
                             color_discrete_sequence=['#1a1a1a'],
                             title="Average Consumption by Hour")
                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    font=dict(family='Inter', size=12, color='#333333'),
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                monthly = df.groupby('month')['use [kW]'].mean().reset_index()
                fig = px.line(monthly, x='month', y='use [kW]', markers=True,
                              color_discrete_sequence=['#1a1a1a'],
                              title="Average Consumption by Month")
                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    font=dict(family='Inter', size=12, color='#333333'),
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                st.plotly_chart(fig, use_container_width=True)

    # ─── PAGE 3: MODEL TRAINING ──────────────────────────────────────────────
    elif page == "🤖 Model Training":
        st.markdown("---")
        
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

            st.markdown("### Model Performance Metrics")
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
            st.markdown("### Actual vs Predicted Values")
            result_df = pd.DataFrame({'Actual': y_test.values[:200], 'Predicted': y_pred[:200]})
            fig = px.line(result_df, title="Actual vs Predicted (First 200 Samples)",
                          color_discrete_sequence=['#1a1a1a', '#666666'])
            fig.update_layout(
                plot_bgcolor='rgba(255,255,255,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(family)
            )
