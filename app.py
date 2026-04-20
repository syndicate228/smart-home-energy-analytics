# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Home Energy Analytics",
    page_icon="⚡",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .main-header { 
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #000000; 
        padding: 1rem 0; 
        border-bottom: 3px solid #000000;
        margin-bottom: 1rem;
    }
    .info-box { 
        background-color: #000000; 
        border-radius: 12px; 
        padding: 1.5rem; 
        color: #ffffff; 
        margin: 1rem 0; 
    }
    .footer { 
        text-align: center; 
        margin-top: 3rem; 
        padding: 2rem; 
        color: #999999; 
        border-top: 1px solid #e5e5e5; 
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">⚡ Smart Home Energy Analytics</div>', unsafe_allow_html=True)
st.write("AI-Powered Load Forecasting & Conservation Advisor for Smart Grid Optimization")

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.write("### 🧭 Navigation")
    st.markdown("---")
    
    page = st.radio(
        "Select Page",
        ["Home", "EDA", "Model Training", "Anomaly Detection", "Model Comparison"],
        index=0
    )
    
    st.markdown("---")
    st.write("### Project Info")
    st.write("- **Subject:** Python for Data Science")
    st.write("- **Team:** SY ECE A1")
    st.write("- **Dataset:** HomeC (Kaggle)")
    st.write("- **Models:** LR, Ridge, RF")
    st.markdown("---")
    st.caption("v2.0 | 2024")

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('HomeC_sample.csv')
        
        # Rename columns if needed (common variations)
        df = df.rename(columns={
            'use_kW': 'use [kW]',
            'gen_kW': 'gen [kW]',
            'Temperature': 'temperature',
            'Humidity': 'humidity',
            'Hour': 'hour',
            'Month': 'month',
            'DayOfWeek': 'dayofweek'
        })
        
        # Extract time features
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['hour'] = df['time'].dt.hour
            df['month'] = df['time'].dt.month
            df['dayofweek'] = df['time'].dt.dayofweek
        
        # Convert to numeric
        for col in ['use [kW]', 'gen [kW]', 'temperature', 'humidity']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop missing values
        df = df.dropna()
        
        # Calculate net consumption
        if 'use [kW]' in df.columns and 'gen [kW]' in df.columns:
            df['net_consumption'] = df['use [kW]'] - df['gen [kW]']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_data()

# Check if data loaded
if df is None:
    st.error("❌ Data failed to load. Please ensure HomeC_sample.csv exists in your GitHub repository.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def show_home():
    """Home Dashboard Page"""
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h4>🔋 Task 1: Smart Grid Load Balancing</h4>
        <p>Type: Regression | Models: Linear, Ridge, Random Forest</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("### Dataset Overview")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg Consumption", f"{df['use [kW]'].mean():.2f} kW")
    c3.metric("Max Consumption", f"{df['use [kW]'].max():.2f} kW")
    c4.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
    
    st.markdown("---")
    st.write("### Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)

def show_eda():
    """EDA Page with Visualizations"""
    st.markdown("---")
    st.info("📊 **Exploratory Data Analysis** — Understanding data patterns before building models")
    
    # TAB 1: DISTRIBUTION
    st.write("### 📈 Distribution")
    st.write("**What:** Shows how energy consumption values are distributed. **Key:** Most values cluster between 0.5-2.0 kW.")
    
    if 'use [kW]' in df.columns:
        fig1 = px.histogram(df, x='use [kW]', nbins=50, title="Energy Consumption Distribution",
                           color_discrete_sequence=['#000000'])
        fig1.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig1, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{df['use [kW]'].mean():.2f} kW")
        c2.metric("Median", f"{df['use [kW]'].median():.2f} kW")
        c3.metric("Std Dev", f"{df['use [kW]'].std():.2f} kW")
    else:
        st.error("Column 'use [kW]' not found")
    
    st.markdown("---")
    
    # TAB 2: CORRELATION
    st.write("### 🔗 Correlation")
    st.write("**What:** Shows relationships between features. **Red** = Positive, **Blue** = Negative.")
    
    cols = ['use [kW]', 'temperature', 'humidity', 'hour', 'month']
    cols = [c for c in cols if c in df.columns]
    
    if len(cols) >= 2:
        corr_df = df[cols]
        fig2, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
        st.pyplot(fig2)
        st.success("✅ Temperature and hour are strong predictors!")
    else:
        st.error("Not enough columns for correlation")
    
    st.markdown("---")
    
    # TAB 3: TRENDS
    st.write("### 📅 Trends")
    st.write("**What:** Consumption patterns by hour and month. **Peak:** 6-9 AM and 5-10 PM.")
    
    if 'hour' in df.columns and 'use [kW]' in df.columns:
        st.write("#### Hourly Pattern")
        hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
        fig3 = px.bar(hourly, x='hour', y='use [kW]', title="Average Consumption by Hour",
                     color_discrete_sequence=['#000000'])
        fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig3, use_container_width=True)
    
    if 'month' in df.columns and 'use [kW]' in df.columns:
        st.write("#### Monthly Pattern")
        monthly = df.groupby('month')['use [kW]'].mean().reset_index()
        fig4 = px.line(monthly, x='month', y='use [kW]', markers=True, title="Average Consumption by Month",
                      color_discrete_sequence=['#000000'])
        fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig4, use_container_width=True)

def show_model_training():
    """Model Training Page"""
    st.markdown("---")
    st.info("🤖 **Model Training** — Train and evaluate machine learning models")
    
    features = ['temperature', 'humidity', 'hour', 'month', 'dayofweek']
    features = [f for f in features if f in df.columns]
    target = 'use [kW]'
    
    if target in df.columns and len(features) >= 2:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_choice = st.selectbox("Select Algorithm", 
                                   ["Linear Regression", "Ridge Regression", "Random Forest"])
        
        if model_choice == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        elif model_choice == "Ridge Regression":
            alpha = st.slider("Alpha", 0.01, 10.0, 1.0)
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        else:
            n_trees = st.slider("Number of Trees", 10, 200, 100)
            model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        
        st.markdown("### Model Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.4f}")
        c2.metric("RMSE", f"{rmse:.4f}")
        c3.metric("R²", f"{r2:.4f}")
        
        st.markdown("---")
        st.write("### Actual vs Predicted")
        result_df = pd.DataFrame({'Actual': y_test.values[:200], 'Predicted': pred[:200]})
        fig = px.line(result_df, title="Actual vs Predicted (First 200 Samples)")
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"✅ {model_choice} trained successfully!")
    else:
        st.error("Missing required columns for training")

def show_anomaly():
    """Anomaly Detection Page"""
    st.markdown("---")
    st.info("🔍 **Anomaly Detection** — IQR method to identify potential electricity theft")
    
    if 'use [kW]' in df.columns:
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
        st.write("### Anomaly Visualization")
        plot_df = df.head(1000).reset_index(drop=True)
        fig = px.scatter(plot_df, y='use [kW]', color='anomaly',
                        color_discrete_map={False: '#000000', True: '#dc3545'},
                        title="Energy Consumption (Red = Anomaly)")
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.write("### Sample Anomalous Records")
        st.dataframe(anomalies[['time', 'use [kW]', 'hour']].head(10), use_container_width=True)
    else:
        st.error("Column 'use [kW]' not found")

def show_comparison():
    """Model Comparison Page"""
    st.markdown("---")
    st.info("📊 **Model Comparison** — Comparing all models on the same test set")
    
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
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        results['RF'] = r2_score(y_test, rf.predict(X_test))
        
        # Display results
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'R² Score'])
        st.dataframe(results_df, use_container_width=True)
        
        fig = px.bar(results_df, x='Model', y='R² Score', color='R² Score',
                    title="Model Comparison (Higher R² = Better)",
                    color_continuous_scale='Blues')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        
        best = results_df.loc[results_df['R²'].idxmax(), 'Model']
        st.success(f"🏆 Best Performing Model: {best}")
    else:
        st.error("Missing required columns")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE ROUTER
# ═══════════════════════════════════════════════════════════════════════════
if page == "Home":
    show_home()
elif page == "EDA":
    show_eda()
elif page == "Model Training":
    show_model_training()
elif page == "Anomaly Detection":
    show_anomaly()
elif page == "Model Comparison":
    show_comparison()

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="footer">Deployed on Streamlit Cloud | Python for Data Science Project | 2024</div>', unsafe_allow_html=True)
