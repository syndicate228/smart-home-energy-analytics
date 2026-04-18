# ─── SECTION 1: IMPORTS ──────────────────────────────────────────────────────
# Import all necessary libraries for data, modeling, and visualization
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

# Ignore warnings to keep the app interface clean
warnings.filterwarnings('ignore')

# ─── SECTION 2: PAGE CONFIGURATION ───────────────────────────────────────────
# Set the browser tab title, icon, and layout width
st.set_page_config(
    page_title="Smart Home Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── SECTION 3: CUSTOM CSS STYLING ───────────────────────────────────────────
# Add custom CSS to make the app look professional and branded
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 10px; }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 0.8rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ─── SECTION 4: HEADER DISPLAY ───────────────────────────────────────────────
# Display the main title and subtitle of the project
st.markdown('<div class="main-header">⚡ Smart Home Energy Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Smart Grid Load Balancing & Anomaly Detection</div>', unsafe_allow_html=True)

# ─── SECTION 5: SIDEBAR NAVIGATION ───────────────────────────────────────────
# Create the sidebar menu for navigating between different sections of the app
with st.sidebar:
    st.image("https://img.icons8.com/color/96/smart-home.png", width=80)
    st.title("Navigation")
    page = st.radio("Go to", [
        "🏠 Home",
        "📊 EDA & Visualizations",
        "🤖 Model Training & Results",
        "🔍 Anomaly Detection",
        "📈 Model Comparison"
    ])
    st.markdown("---")
    st.markdown("**Project:** Python for Data Science")
    st.markdown("**Dataset:** HomeC (Energy Consumption)")
    st.markdown("**Models:** LR, Ridge, Random Forest")

# ─── SECTION 6: DATA LOADING & PREPROCESSING ─────────────────────────────────
# Function to load data. Uses caching to improve performance.
@st.cache_data
def load_data():
    try:
        # Attempt to load the local CSV file
        # IMPORTANT: Ensure 'HomeC.csv' is in the same folder as app.py
        df = pd.read_csv('HomeC_sample.csv')

        
        # --- Data Cleaning & Feature Engineering ---
        # Convert time column to datetime if it exists
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['hour'] = df['time'].dt.hour
            df['month'] = df['time'].dt.month
            df['dayofweek'] = df['time'].dt.dayofweek
        else:
            # Fallback if time column is missing
            df['hour'] = np.random.randint(0, 24, len(df))
            df['month'] = np.random.randint(1, 13, len(df))
            df['dayofweek'] = np.random.randint(0, 7, len(df))

        # Ensure numeric columns are numeric (handle potential string errors)
        numeric_cols = ['use [kW]', 'gen [kW]', 'temperature', 'humidity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing values created during conversion
        df = df.dropna()
        
        # Calculate net consumption (Usage - Generation)
        if 'use [kW]' in df.columns and 'gen [kW]' in df.columns:
            df['net_consumption'] = df['use [kW]'] - df['gen [kW]']
            
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'HomeC.csv' not found. Please ensure the dataset is uploaded to GitHub.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# Load the data
df = load_data()

# ─── SECTION 7: PAGE LOGIC ───────────────────────────────────────────────────
# Only render pages if data loaded successfully
if df is not None:

    # ─── PAGE 1: HOME DASHBOARD ──────────────────────────────────────────────
    if page == "🏠 Home":
        st.subheader("📌 Project Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **🔋 Task 1: Smart Grid Load Balancing**
            - **Type:** Supervised Regression
            - **Goal:** Predict future energy consumption based on weather & time.
            - **Models:** Linear Regression, Ridge, Random Forest.
            """)
        with col2:
            st.warning("""
            **🚨 Task 2: Electricity Theft Detection**
            - **Type:** Unsupervised Anomaly Detection
            - **Goal:** Identify abnormal consumption patterns.
            - **Method:** Interquartile Range (IQR) Statistical Method.
            """)

        st.markdown("---")
        st.subheader("📄 Dataset Snapshot")
        st.dataframe(df.head(), use_container_width=True)
        
        # Key Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Avg Consumption", f"{df['use [kW]'].mean():.2f} kW")
        c3.metric("Max Consumption", f"{df['use [kW]'].max():.2f} kW")
        c4.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")

    # ─── PAGE 2: EDA & VISUALIZATIONS ────────────────────────────────────────
    elif page == "📊 EDA & Visualizations":
        st.header("📊 Exploratory Data Analysis")
        tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Time Trends"])

        with tab1:
            st.subheader("Energy Consumption Distribution")
            fig = px.histogram(df, x='use [kW]', nbins=50, color_discrete_sequence=['#1f77b4'],
                               title="Distribution of Energy Consumption (kW)")
            st.plotly_chart(fig, use_container_width=True)

            with tab2:
        st.subheader("Feature Correlation Heatmap")
        
        # Select only key features (not all appliances)
        key_features = ['use [kW]', 'gen [kW]', 'temperature', 'humidity', 
                        'hour', 'month', 'dayofweek', 'net_consumption']
        
        # Filter dataframe to only include these columns
        correlation_df = df[key_features]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', 
                    ax=ax, fmt='.2f', square=True, linewidths=0.5)
        plt.title('Correlation Matrix - Key Features', fontsize=14, pad=20)
        st.pyplot(fig)
        
        # Add insights
        st.info("""
        **Key Insights:**
        - Temperature & Humidity affect energy consumption
        - Hour of day shows usage patterns
        - Net consumption = Usage - Generation
        """)


        with tab3:
            st.subheader("Consumption Trends")
            col_a, col_b = st.columns(2)
            with col_a:
                hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
                fig = px.bar(hourly, x='hour', y='use [kW]', title="Avg Consumption by Hour")
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                monthly = df.groupby('month')['use [kW]'].mean().reset_index()
                fig = px.line(monthly, x='month', y='use [kW]', markers=True, title="Avg Consumption by Month")
                st.plotly_chart(fig, use_container_width=True)

    # ─── PAGE 3: MODEL TRAINING ──────────────────────────────────────────────
    elif page == "🤖 Model Training & Results":
        st.header("🤖 Machine Learning Model Training")
        
        # Define Features and Target
        # Ensure these column names match your CSV exactly
        features = ['gen [kW]', 'temperature', 'humidity', 'hour', 'month', 'dayofweek']
        target = 'use [kW]'
        
        # Check if features exist in dataframe
        if all(f in df.columns for f in features) and target in df.columns:
            X = df[features]
            y = df[target]
            
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scaling (Required for Linear/Ridge, not strictly for RF but good for comparison)
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            # Model Selection
            model_choice = st.selectbox("Select Algorithm", ["Linear Regression", "Ridge Regression", "Random Forest"])
            
            # Train Model based on selection
            if model_choice == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
            elif model_choice == "Ridge Regression":
                alpha = st.slider("Ridge Alpha (Regularization)", 0.01, 10.0, 1.0)
                model = Ridge(alpha=alpha)
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
            else:
                n_trees = st.slider("Number of Trees", 10, 200, 100, step=10)
                model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
                model.fit(X_train, y_train) # RF doesn't need scaled data
                y_pred = model.predict(X_test)

            # Evaluation Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.markdown("### 📏 Model Performance")
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE (Error)", f"{mae:.4f}")
            c2.metric("RMSE (Error)", f"{rmse:.4f}")
            c3.metric("R² (Accuracy)", f"{r2:.4f}")

            # Visualization
            st.subheader("Actual vs Predicted Values")
            result_df = pd.DataFrame({'Actual': y_test.values[:200], 'Predicted': y_pred[:200]})
            fig = px.line(result_df, title="Actual vs Predicted Energy Consumption (First 200 Samples)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance for Random Forest
            if model_choice == "Random Forest":
                st.subheader("Feature Importance")
                fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                fi = fi.sort_values('Importance', ascending=True)
                fig = px.bar(fi, x='Importance', y='Feature', orientation='h', color='Importance')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Column names in CSV do not match expected features. Please check `app.py` feature list.")

    # ─── PAGE 4: ANOMALY DETECTION ───────────────────────────────────────────
    elif page == "🔍 Anomaly Detection":
        st.header("🔍 Electricity Theft / Anomaly Detection")
        st.info("Using **IQR (Interquartile Range)** method to identify outliers that may indicate theft or leakage.")

        # Calculate IQR
        Q1 = df['use [kW]'].quantile(0.25)
        Q3 = df['use [kW]'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Flag Anomalies
        df['anomaly'] = ((df['use [kW]'] < lower) | (df['use [kW]'] > upper))
        anomalies = df[df['anomaly'] == True]
        normal = df[df['anomaly'] == False]

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("🚨 Anomalies Detected", len(anomalies), delta_color="inverse")
        c3.metric("✅ Normal Records", len(normal))

        # Visualization
        st.subheader("Consumption with Anomalies Highlighted")
        # Reset index for plotting
        plot_df = df.head(1000).reset_index(drop=True)
        fig = px.scatter(plot_df, x=plot_df.index, y='use [kW]',
                         color='anomaly', color_discrete_map={False: '#1f77b4', True: '#d62728'},
                         title="Energy Consumption (Red = Anomaly)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sample Anomalous Records")
        st.dataframe(anomalies[['time', 'use [kW]', 'hour', 'temperature']].head(10), use_container_width=True)

    # ─── PAGE 5: MODEL COMPARISON ────────────────────────────────────────────
    elif page == "📈 Model Comparison":
        st.header("📈 Algorithm Comparison")
        st.write("Comparing all three models on the same test set to determine the best performer.")

        features = ['gen [kW]', 'temperature', 'humidity', 'hour', 'month', 'dayofweek']
        target = 'use [kW]'
        X = df[features]; y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        results = {}
        # Train all models
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

        # Display Comparison Table
        comparison_df = pd.DataFrame(results).T.reset_index()
        comparison_df.columns = ['Model', 'MAE', 'RMSE', 'R²']
        st.dataframe(comparison_df, use_container_width=True)

        # Visual Comparison
        fig = px.bar(comparison_df, x='Model', y='R²', color='R²',
                     color_continuous_scale='Blues', title="R² Score Comparison (Higher is Better)")
        st.plotly_chart(fig, use_container_width=True)

        best_model = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        st.success(f"🏆 Best Performing Model: **{best_model}**")

# ─── SECTION 8: FOOTER ───────────────────────────────────────────────────────
# Display a professional footer
st.markdown("---")
st.markdown('<div class="footer">🚀 Deployed on Streamlit Cloud | 🎓 Python for Data Science Project</div>', unsafe_allow_html=True)
