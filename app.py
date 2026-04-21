# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
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
st.set_page_config(page_title="Smart Home Energy Analytics", page_icon="⚡", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #000000; padding: 1rem 0; border-bottom: 3px solid #000000; margin-bottom: 1rem; }
    
    /* Modern Task Cards */
    .task-card-dark {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        color: #ffffff;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        border-left: 5px solid #00d9ff;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .task-card-dark:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
    }
    
    .task-card-light {
        background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 2rem;
        color: #1a1a2e;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(220, 53, 69, 0.1);
        border-left: 5px solid #dc3545;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .task-card-light:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(220, 53, 69, 0.2);
    }
    
    .task-card-dark h4, .task-card-light h4 {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .task-card-dark p, .task-card-light p {
        font-size: 0.95rem;
        margin: 0.75rem 0;
        line-height: 1.6;
    }
    
    .task-card-dark strong, .task-card-light strong {
        color: #00d9ff;
        font-weight: 600;
    }
    
    .task-card-light strong {
        color: #dc3545;
        font-weight: 600;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 2px solid #e5e5e5;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #000000;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: #ffffff;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #f0fff4;
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff9f9;
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
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
    page = st.radio("Select Page", ["Home", "EDA", "Model Training", "Anomaly Detection", "Model Comparison", "Live Prediction"], index=0)
    st.markdown("---")
    st.write("### 📋 Project Info")
    st.write("- **Subject:** Python for Data Science")
    st.write("- **Team:** SY ECE A1")
    st.write("- **Dataset:** HomeC (Kaggle)")
    st.write("- **Models:** LR, Ridge, RF")
    st.write("### 🔄 Auto-Refresh")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 30)
    if auto_refresh:
        st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds")
        time.sleep(refresh_interval)
        st.rerun()
    st.markdown("---")
    st.caption("v2.0 | 2026")

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('HomeC_sample.csv')
        df = df.rename(columns={'use_kW': 'use_kW', 'gen_kW': 'gen_kW', 'Temperature': 'temperature', 'Humidity': 'humidity'})
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

if df is None:
    st.error("Data failed to load. Check HomeC_sample.csv exists.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def show_home():
    """Home Page - Modern Cards + Metrics + Info"""
    st.markdown("---")
    
    # Welcome Message
    st.write("### 👋 Welcome to Smart Home Energy Analytics")
    st.write("""
    This application helps utility companies and homeowners:
    - **Predict** energy consumption using machine learning
    - **Detect** anomalies that may indicate electricity theft
    - **Optimize** energy usage for cost savings
    - **Balance** grid load for better efficiency
    """)
    
    st.markdown("---")
    
    # Modern Task Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="task-card-dark">
            <h4>🔋 Task 1: Smart Grid Load Balancing</h4>
            <p><strong>Type:</strong> Supervised Regression</p>
            <p><strong>Goal:</strong> Predict energy consumption based on environmental and usage factors</p>
            <p><strong>Models:</strong> Linear Regression, Ridge, Random Forest</p>
            <p><strong>Metrics:</strong> R² Score, MAE, RMSE</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="task-card-light">
            <h4>🚨 Task 2: Electricity Theft Detection</h4>
            <p><strong>Type:</strong> Unsupervised Anomaly Detection</p>
            <p><strong>Goal:</strong> Identify abnormal consumption patterns</p>
            <p><strong>Method:</strong> IQR (Interquartile Range) Statistical Method</p>
            <p><strong>Output:</strong> Flagged records for investigation</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Statistics
    st.write("### 📊 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📁 Total Records", f"{len(df):,}")
    c2.metric("⚡ Avg Consumption", f"{df['use [kW]'].mean():.2f} kW")
    c3.metric("🌡️ Avg Temperature", f"{df['temperature'].mean():.1f}°C")
    c4.metric("💧 Avg Humidity", f"{df['humidity'].mean():.1f}%")
    c5.metric("📋 Features", f"{len(df.columns)}")
    
    st.markdown("---")
    
    # Dataset Sample
    with st.expander("📄 View Dataset Sample (First 10 Records)"):
        st.dataframe(df.head(10), width='stretch')
    
    # Column Information
    with st.expander("📋 View Column Descriptions"):
        st.write("""
        | Column | Description | Type |
        |--------|-------------|------|
        | time | Timestamp of measurement | DateTime |
        | use [kW] | Energy consumption in kilowatts | Numeric |
        | gen [kW] | Energy generated (solar) in kilowatts | Numeric |
        | temperature | Outdoor temperature in Celsius | Numeric |
        | humidity | Relative humidity percentage | Numeric |
        | hour | Hour of day (0-23) | Numeric |
        | month | Month of year (1-12) | Numeric |
        | dayofweek | Day of week (0=Monday, 6=Sunday) | Numeric |
        """)

def show_eda():
    """EDA Page - Mix of Charts, Text, Tables, Insights"""
    st.markdown("---")
    
    # Introduction
    st.write("### 📊 Exploratory Data Analysis")
    st.write("""
    EDA helps us understand data patterns before building models. We analyze:
    - **Distribution** of energy consumption values
    - **Correlations** between features
    - **Temporal trends** (hourly, monthly patterns)
    """)
    
    st.markdown("---")
    
    # Key Statistics
    st.write("### 📈 Key Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Consumption", f"{df['use [kW]'].mean():.2f} kW")
    c2.metric("Median Consumption", f"{df['use [kW]'].median():.2f} kW")
    c3.metric("Std Deviation", f"{df['use [kW]'].std():.2f} kW")
    c4.metric("Range", f"{df['use [kW]'].min():.1f} - {df['use [kW]'].max():.1f} kW")
    
    st.markdown("---")
    
    # Distribution Chart
    st.write("### 📉 Consumption Distribution")
    fig1 = px.histogram(df, x='use [kW]', nbins=50, title="Energy Consumption Distribution",
                       color_discrete_sequence=['#000000'])
    st.plotly_chart(fig1, width='stretch')
    
    st.markdown("""
    <div class="success-box">
        <strong>Observation:</strong> Most consumption values cluster between 0.5-2.0 kW (typical household usage). 
        Some high-consumption outliers exist (appliance spikes, AC usage).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Correlation Heatmap
    st.write("### 🔗 Feature Correlation")
    
    key_features = ['use [kW]', 'temperature', 'humidity', 'hour', 'month']
    key_features = [f for f in key_features if f in df.columns]
    
    if len(key_features) >= 2:
        corr_df = df[key_features]
        corr_matrix = corr_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f',
                   linewidths=2, linecolor='white', annot_kws={'size': 14, 'weight': 'bold'})
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation Insights (TEXT + TABLE)
        st.write("#### 🔝 Top Correlations with Consumption")
        correlations = corr_matrix['use [kW]'].abs().sort_values(ascending=False)
        corr_table = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation': correlations.values.round(2)
        })
        st.dataframe(corr_table.head(5), width='stretch')
        
        if len(correlations) > 1:
            st.success(f"💡 **Strongest Predictor:** {correlations.index[1]} (correlation: {correlations.iloc[1]:.2f})")
    
    st.markdown("---")
    
    # Time Trends (TABS)
    st.write("### 📅 Time-Based Patterns")
    
    tab1, tab2 = st.tabs(["Hourly Pattern", "Monthly Pattern"])
    
    with tab1:
        st.write("#### Consumption by Hour of Day")
        hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
        fig = px.bar(hourly, x='hour', y='use [kW]', title="Average Consumption by Hour",
                    color_discrete_sequence=['#000000'])
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("""
        <div class="info-box">
            <strong>Peak Hours:</strong> 6-9 AM (morning) and 5-10 PM (evening)<br>
            <strong>Low Hours:</strong> 11 PM - 5 AM (night)<br>
            <strong>Recommendation:</strong> Shift high-energy appliances to off-peak hours
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.write("#### Consumption by Month")
        monthly = df.groupby('month')['use [kW]'].mean().reset_index()
        fig = px.line(monthly, x='month', y='use [kW]', markers=True, title="Average Consumption by Month",
                     color_discrete_sequence=['#000000'])
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("""
        <div class="info-box">
            <strong>Summer:</strong> Higher consumption (AC usage)<br>
            <strong>Winter:</strong> Elevated consumption (heating)<br>
            <strong>Spring/Fall:</strong> Lower consumption (moderate weather)
        </div>
        """, unsafe_allow_html=True)

def show_model_training():
    """Model Training Page - Interactive + Metrics + Results"""
    st.markdown("---")
    
    st.write("### 🤖 Machine Learning Model Training")
    st.write("""
    Train and evaluate regression models to predict energy consumption.
    Select a model, adjust parameters, and see performance metrics.
    """)
    
    st.markdown("---")
    
    # Model Selection
    st.write("### ⚙️ Model Configuration")
    
    features = ['temperature', 'humidity', 'hour', 'month', 'dayofweek']
    features = [f for f in features if f in df.columns]
    target = 'use [kW]'
    
    if target in df.columns and len(features) >= 2:
        model_choice = st.selectbox("Select Algorithm", 
                                   ["Linear Regression", "Ridge Regression", "Random Forest"])
        
        # Model-specific parameters
        if model_choice == "Ridge Regression":
            alpha = st.slider("Regularization Strength (Alpha)", 0.01, 10.0, 1.0, 0.1)
        elif model_choice == "Random Forest":
            n_trees = st.slider("Number of Trees", 10, 200, 100, 10)
        
        # Train button
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                elif model_choice == "Ridge Regression":
                    model = Ridge(alpha=alpha)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                else:
                    model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                
                # Metrics
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)
                
                st.markdown("---")
                st.write("### 📏 Model Performance")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE (Error)", f"{mae:.4f}")
                c2.metric("RMSE (Error)", f"{rmse:.4f}")
                c3.metric("R² (Accuracy)", f"{r2:.4f}")
                
                # Interpretation
                st.markdown("---")
                st.write("### 📊 Performance Interpretation")
                
                if r2 > 0.7:
                    st.success(f"✅ **Excellent!** R² = {r2:.4f} indicates strong predictive power")
                elif r2 > 0.5:
                    st.info(f"👍 **Good!** R² = {r2:.4f} indicates moderate predictive power")
                else:
                    st.warning(f"⚠️ **Fair** R² = {r2:.4f} - consider adding more features")
                
                # Feature Importance (for RF)
                if model_choice == "Random Forest":
                    st.markdown("---")
                    st.write("### 🎯 Feature Importance")
                    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                    fi = fi.sort_values('Importance', ascending=True)
                    fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                                title="Which Features Matter Most?",
                                color_discrete_sequence=['#000000'])
                    st.plotly_chart(fig, width='stretch')
                
                st.success(f"✅ {model_choice} trained successfully!")
    else:
        st.error("Missing required columns for training")

def show_anomaly():
    """Anomaly Detection Page - Metrics + Scatter + Table"""
    st.markdown("---")
    
    st.write("### 🔍 Anomaly Detection")
    st.write("""
    Identify unusual consumption patterns that may indicate:
    - **Electricity theft** (tampering with meters)
    - **Equipment malfunction** (faulty appliances)
    - **Energy leakage** (wastage)
    """)
    
    st.markdown("---")
    
    if 'use [kW]' in df.columns:
        # IQR Calculation
        Q1 = df['use [kW]'].quantile(0.25)
        Q3 = df['use [kW]'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df['anomaly'] = ((df['use [kW]'] < lower) | (df['use [kW]'] > upper))
        anomalies = df[df['anomaly'] == True]
        normal = df[df['anomaly'] == False]
        
        # Metrics
        st.write("### 📊 Detection Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("📁 Total Records", f"{len(df):,}")
        c2.metric("🚨 Anomalies Detected", f"{len(anomalies):,}")
        c3.metric("✅ Normal Records", f"{len(normal):,}")
        
        # Percentage
        anomaly_pct = (len(anomalies) / len(df)) * 100
        st.write(f"**Anomaly Rate:** {anomaly_pct:.2f}%")
        
        if anomaly_pct > 5:
            st.warning("⚠️ High anomaly rate - recommend investigation")
        else:
            st.success("✅ Normal anomaly rate")
        
        st.markdown("---")
        
        # Method Explanation
        with st.expander("📖 How IQR Method Works"):
            st.write("""
            **Step 1:** Calculate Q1 (25th percentile) and Q3 (75th percentile)
            
            **Step 2:** Calculate IQR = Q3 - Q1
            
            **Step 3:** Define bounds:
            - Lower Bound = Q1 - 1.5 × IQR
            - Upper Bound = Q3 + 1.5 × IQR
            
            **Step 4:** Flag values outside bounds as anomalies
            """)
        
        st.markdown("---")
        
        # Visualization
        st.write("### 📈 Anomaly Visualization")
        plot_df = df.head(1000).reset_index(drop=True)
        fig = px.scatter(plot_df, y='use [kW]', color='anomaly',
                        color_discrete_map={False: '#000000', True: '#dc3545'},
                        title="Energy Consumption (Red = Anomaly)")
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # Sample Anomalous Records
        st.write("### 📋 Sample Anomalous Records")
        if len(anomalies) > 0:
            st.dataframe(anomalies[['time', 'use [kW]', 'hour', 'temperature']].head(10), width='stretch')
            st.warning("⚠️ These records should be investigated for potential theft or equipment issues")
        else:
            st.success("✅ No anomalies found in sample")
    else:
        st.error("Column 'use [kW]' not found")

def show_comparison():
    """Model Comparison Page - Table + Chart + Winner"""
    st.markdown("---")
    
    st.write("### 📊 Model Comparison")
    st.write("""
    Compare all three models on the same test set to determine the best performer.
    All models trained with 80-20 train-test split.
    """)
    
    st.markdown("---")
    
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
        
        # Results Table
        st.write("###  Performance Comparison")
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'R2_Score'])
        results_df['R2_Score'] = results_df['R2_Score'].round(4)
        st.dataframe(results_df.style.highlight_max(color='lightgreen'), width='stretch')
        
        st.markdown("---")
        
        # Bar Chart
        st.write("### 📈 Visual Comparison")
        fig = px.bar(results_df, x='Model', y='R2_Score', color='R2_Score',
                    title="R² Score Comparison (Higher = Better)",
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # Winner Announcement
        best = results_df.loc[results_df['R2_Score'].idxmax(), 'Model']
        st.markdown(f"""
        <div class="success-box">
            <h3>🏆 Best Performing Model: {best}</h3>
            <p>R² Score: {results_df.loc[results_df['Model'] == best, 'R2_Score'].values[0]:.4f}</p>
            <p><strong>Recommendation:</strong> Use this model for production deployment</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Why This Model Won
        with st.expander("💡 Why Did This Model Win?"):
            if best == 'Linear':
                st.write("Linear Regression works well when relationships are linear and features are not highly correlated.")
            elif best == 'Ridge':
                st.write("Ridge Regression handles multicollinearity better through L2 regularization.")
            else:
                st.write("Random Forest captures non-linear relationships and handles complex patterns better.")
    else:
        st.error("Missing required columns")

def show_live_prediction():
    """Live Energy Prediction Calculator"""
    st.markdown("---")
    st.write("### ⚡ Live Energy Prediction")
    st.write("Enter current conditions to predict energy consumption:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.slider("Temperature (°C)", 10.0, 40.0, 25.0, 0.5)
    with col2:
        humidity = st.slider("Humidity (%)", 30.0, 90.0, 60.0, 5.0)
    with col3:
        hour = st.slider("Hour of Day", 0, 23, 12)
    
    # Make prediction
    if st.button("🔮 Predict Now"):
        # Use your trained model
        input_data = pd.DataFrame({
            'temperature': [temp],
            'humidity': [humidity],
            'hour': [hour],
            'month': [datetime.now().month],
            'dayofweek': [datetime.now().weekday()]
        })
        
        # Load your best model (Random Forest)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        features = ['temperature', 'humidity', 'hour', 'month', 'dayofweek']
        X = df[features]
        y = df['use [kW]']
        model.fit(X, y)
        
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.markdown(f"""
        <div class="success-box" style="text-align: center; padding: 2rem;">
            <h2 style="font-size: 3rem; margin: 0;">⚡ {prediction:.2f} kW</h2>
            <p style="font-size: 1.2rem;">Predicted Energy Consumption</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation
        if prediction > 2.0:
            st.warning("⚠️ **High Consumption Alert:** Consider turning off non-essential appliances")
        elif prediction > 1.5:
            st.info("ℹ️ **Moderate Consumption:** Normal usage pattern")
        else:
            st.success("✅ **Low Consumption:** Efficient energy usage!")

# Add this to your navigation
# In sidebar, add: "Live Prediction"
# Then add: elif page == "Live Prediction": show_live_prediction()

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
elif page == "Live Prediction": 
    show_live_prediction()
# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="footer">Deployed on Streamlit Cloud | Python for Data Science Project | 2026</div>', unsafe_allow_html=True)
