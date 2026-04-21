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
st.set_page_config(page_title="Smart Home Energy Analytics", page_icon="⚡", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #000000; padding: 1rem 0; border-bottom: 3px solid #000000; margin-bottom: 1rem; }
    .info-box { background-color: #000000; border-radius: 12px; padding: 1.5rem; color: #ffffff; margin: 1rem 0; }
    .footer { text-align: center; margin-top: 3rem; padding: 2rem; color: #999999; border-top: 1px solid #e5e5e5; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">⚡ Smart Home Energy Analytics</div>', unsafe_allow_html=True)
st.write("AI-Powered Load Forecasting & Conservation Advisor")

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.write("### Navigation")
    st.markdown("---")
    page = st.radio("Select Page", ["Home", "EDA", "Model Training", "Anomaly Detection", "Model Comparison"], index=0)
    st.markdown("---")
    st.write("### Project Info")
    st.write("- Subject: Python for Data Science")
    st.write("- Team: SY ECE A1")
    st.write("- Dataset: HomeC (Kaggle)")
    st.write("- Models: LR, Ridge, RF")

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
    st.markdown("---")
    st.info("Task 1: Load Balancing (Regression) | Task 2: Anomaly Detection (IQR)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg Consumption", f"{df['use [kW]'].mean():.2f} kW")
    c3.metric("Max Consumption", f"{df['use [kW]'].max():.2f} kW")
    c4.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
    st.markdown("---")
    st.write("### Dataset Sample")
    st.dataframe(df.head(10), width='stretch')

def show_eda():
    """Enhanced EDA Page with Interactive Features"""
    st.markdown("---")
    st.info("📊 **Exploratory Data Analysis** — Interactive data exploration with smart insights")
    
    # ─── QUICK STATS ─────────────────────────────────────────────────────────
    st.write("### 📱 Quick Statistics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg Consumption", f"{df['use [kW]'].mean():.2f} kW")
    c3.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
    c4.metric("Avg Humidity", f"{df['humidity'].mean():.1f}%")
    c5.metric("Features", f"{len(df.columns)}")
    
    st.markdown("---")
    
    # ─── DROPDOWN: SELECT FEATURE & CHART TYPE ─────────────────────────────
    st.write("### 🔍 Interactive Visualization")
    
    available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_feature = st.selectbox(
            "Choose Feature",
            available_cols,
            index=available_cols.index('use [kW]') if 'use [kW]' in available_cols else 0
        )
    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Histogram", "Bar Chart", "Line Chart", "Box Plot", "Scatter Plot"]
        )
    with col3:
        color_theme = st.selectbox(
            "Color Theme",
            ["Blues", "Viridis", "Plasma", "Inferno", "Dark2", "Set2"]
        )
    
    st.markdown("---")
    
    # ─── RENDER SELECTED CHART ───────────────────────────────────────────────
    st.write(f"### 📈 {selected_feature} - {chart_type}")
    
    try:
        if chart_type == "Histogram":
            fig = px.histogram(df, x=selected_feature, nbins=50, title=f"Distribution of {selected_feature}",
                              color_discrete_sequence=px.colors.qualitative.__dict__.get(color_theme, ['#000000']))
            st.plotly_chart(fig, width='stretch')
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{df[selected_feature].mean():.2f}")
            c2.metric("Median", f"{df[selected_feature].median():.2f}")
            c3.metric("Std Dev", f"{df[selected_feature].std():.2f}")
            c4.metric("Min-Max", f"{df[selected_feature].min():.1f} - {df[selected_feature].max():.1f}")
        
        elif chart_type == "Bar Chart":
            if 'hour' in df.columns and selected_feature != 'hour':
                grouped = df.groupby('hour')[selected_feature].mean().reset_index()
                fig = px.bar(grouped, x='hour', y=selected_feature, 
                            title=f"{selected_feature} by Hour",
                            color_discrete_sequence=px.colors.qualitative.__dict__.get(color_theme, ['#000000']))
                st.plotly_chart(fig, width='stretch')
                st.write("**Insight:** Shows how consumption varies throughout the day.")
            else:
                st.warning("Hour column needed for bar chart grouping")
        
        elif chart_type == "Line Chart":
            if 'month' in df.columns:
                grouped = df.groupby('month')[selected_feature].mean().reset_index()
                fig = px.line(grouped, x='month', y=selected_feature, markers=True,
                             title=f"{selected_feature} by Month",
                             color_discrete_sequence=px.colors.qualitative.__dict__.get(color_theme, ['#000000']))
                st.plotly_chart(fig, width='stretch')
                st.write("**Insight:** Shows seasonal trends in consumption.")
            else:
                st.warning("Month column needed for line chart")
        
        elif chart_type == "Box Plot":
            fig = px.box(df, y=selected_feature, title=f"Box Plot of {selected_feature}",
                        color_discrete_sequence=px.colors.qualitative.__dict__.get(color_theme, ['#000000']))
            st.plotly_chart(fig, width='stretch')
            st.write("**Insight:** Shows distribution, median, and outliers.")
        
        elif chart_type == "Scatter Plot":
            if 'temperature' in df.columns and selected_feature != 'temperature':
                fig = px.scatter(df, x='temperature', y=selected_feature, 
                                title=f"{selected_feature} vs Temperature",
                                color_discrete_sequence=px.colors.qualitative.__dict__.get(color_theme, ['#000000']))
                st.plotly_chart(fig, width='stretch')
                st.write("**Insight:** Shows relationship between temperature and consumption.")
            else:
                st.warning("Temperature column needed for scatter plot")
    
    except Exception as e:
        st.error(f"Chart error: {e}")
    
    st.markdown("---")
    
    # ─── AUTO-GENERATED INSIGHTS ─────────────────────────────────────────────
    st.write("### 💡 Auto-Generated Insights")
    
    insights = []
    
    if 'hour' in df.columns and 'use [kW]' in df.columns:
        peak_hour = df.groupby('hour')['use [kW]'].mean().idxmax()
        insights.append(f"⚡ Peak consumption occurs at {peak_hour}:00 hours")
    
    if 'temperature' in df.columns and 'use [kW]' in df.columns:
        temp_corr = df['temperature'].corr(df['use [kW]'])
        if temp_corr > 0.3:
            insights.append(f"🌡️ Positive correlation between temperature and consumption ({temp_corr:.2f})")
        elif temp_corr < -0.3:
            insights.append(f"🌡️ Negative correlation between temperature and consumption ({temp_corr:.2f})")
    
    avg_consumption = df['use [kW]'].mean()
    insights.append(f"📈 Average household consumption: {avg_consumption:.2f} kW")
    
    for insight in insights:
        st.info(insight)
    
    st.markdown("---")
    
     # ─── CORRELATION HEATMAP ─────────────────────────────────────────────────
    st.write("### 🔗 Feature Correlation Heatmap")
    st.write("Red = Positive | Blue = Negative | Darker = Stronger")
    
    key_features = ['use [kW]', 'temperature', 'humidity', 'hour', 'month']
    key_features = [f for f in key_features if f in df.columns]
    
    if len(key_features) >= 2:
        corr_df = df[key_features]
        corr_matrix = corr_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # ✅ REMOVED mask - show full heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   ax=ax, 
                   fmt='.2f',
                   linewidths=2,
                   linecolor='white',
                   cbar_kws={'shrink': 0.8},
                   annot_kws={'size': 14, 'weight': 'bold'},
                   square=True,
                   vmin=-1, vmax=1)
        
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show insights
        st.write("#### 🔝 Key Insights")
        if 'use [kW]' in key_features:
            correlations = corr_matrix['use [kW]'].abs().sort_values(ascending=False)
            
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**Strongest Predictor:** {correlations.index[1]} ({correlations.iloc[1]:.2f})")
            with c2:
                st.info(f"**Weakest Predictor:** {correlations.index[-1]} ({correlations.iloc[-1]:.2f})")
    else:
        st.error("Not enough features for correlation")
    
    # ─── CONSUMPTION CATEGORIES ──────────────────────────────────────────────
    st.write("### 📊 Consumption Categories")
    
    df['consumption_category'] = pd.cut(df['use [kW]'], 
                                        bins=[0, 1, 2, 3, float('inf')],
                                        labels=['Low (0-1 kW)', 'Medium (1-2 kW)', 'High (2-3 kW)', 'Very High (>3 kW)'])
    
    cat_counts = df['consumption_category'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=cat_counts.values, names=cat_counts.index, 
                    title="Consumption Category Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.write("### Category Statistics")
        for cat in cat_counts.index:
            count = cat_counts[cat]
            pct = (count / len(df)) * 100
            st.write(f"**{cat}:** {count:,} records ({pct:.1f}%)")
    
    st.markdown("---")
    
    # ─── TIME TRENDS ─────────────────────────────────────────────────────────
    st.write("### 📅 Time-Based Trends")
    
    tab1, tab2 = st.tabs(["Hourly", "Monthly"])
    
    with tab1:
        if 'hour' in df.columns and 'use [kW]' in df.columns:
            st.write("#### Consumption by Hour")
            hourly = df.groupby('hour')['use [kW]'].mean().reset_index()
            fig = px.bar(hourly, x='hour', y='use [kW]', 
                        title="Average Consumption by Hour of Day",
                        color_discrete_sequence=['#000000'])
            st.plotly_chart(fig, width='stretch')
            st.write("**Peak Hours:** 6-9 AM (morning) and 5-10 PM (evening)")
            st.write("**Low Hours:** 11 PM - 5 AM (night)")
    
    with tab2:
        if 'month' in df.columns and 'use [kW]' in df.columns:
            st.write("#### Consumption by Month")
            monthly = df.groupby('month')['use [kW]'].mean().reset_index()
            fig = px.line(monthly, x='month', y='use [kW]', markers=True,
                         title="Average Consumption by Month",
                         color_discrete_sequence=['#000000'])
            st.plotly_chart(fig, width='stretch')
            st.write("**Summer:** Higher (AC usage) | **Winter:** Elevated (heating) | **Spring/Fall:** Lower")
    
    st.markdown("---")
    
    # ─── DAY OF WEEK ANALYSIS ────────────────────────────────────────────────
    st.write("### 📆 Day of Week Pattern")
    
    if 'dayofweek' in df.columns and 'use [kW]' in df.columns:
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = df.groupby('dayofweek')['use [kW]'].mean().reset_index()
        daily['day_name'] = daily['dayofweek'].apply(lambda x: day_names[x])
        
        fig = px.bar(daily, x='day_name', y='use [kW]', 
                    title="Average Consumption by Day of Week",
                    color='use [kW]',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, width='stretch')
        
        highest_day = daily.loc[daily['use [kW]'].idxmax(), 'day_name']
        lowest_day = daily.loc[daily['use [kW]'].idxmin(), 'day_name']
        
        c1, c2 = st.columns(2)
        c1.success(f"📈 Highest: {highest_day}")
        c2.info(f"📉 Lowest: {lowest_day}")
    
    st.markdown("---")
    
    # ─── DOWNLOAD BUTTON ─────────────────────────────────────────────────────
    st.write("### 📥 Export EDA Data")
    
    csv_data = df.describe().to_csv()
    st.download_button(
        label="Download Statistics (CSV)",
        data=csv_data,
        file_name="eda_statistics.csv",
        mime="text/csv"
    )

def show_model_training():
    st.markdown("---")
    st.info("Model Training - Train and evaluate ML models")
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
        elif model_choice == "Ridge":
            alpha = st.slider("Alpha", 0.01, 10.0, 1.0)
            model = Ridge(alpha=alpha)
        else:
            n_trees = st.slider("Trees", 10, 200, 100)
            model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.4f}")
        c2.metric("RMSE", f"{rmse:.4f}")
        c3.metric("R2", f"{r2:.4f}")
        st.markdown("---")
        st.write("### Actual vs Predicted")
        result_df = pd.DataFrame({'Actual': y_test.values[:200], 'Predicted': pred[:200]})
        fig = px.line(result_df, title="Actual vs Predicted")
        st.plotly_chart(fig, width='stretch')
        st.success(f"{model_choice} trained!")
    else:
        st.error("Missing columns")

def show_anomaly():
    st.markdown("---")
    st.info("Anomaly Detection - IQR method")
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
        c2.metric("Anomalies", f"{len(anomalies):,}")
        c3.metric("Normal", f"{len(df) - len(anomalies):,}")
        st.markdown("---")
        st.write("### Visualization")
        plot_df = df.head(1000).reset_index(drop=True)
        fig = px.scatter(plot_df, y='use [kW]', color='anomaly', title="Anomalies (Red)")
        st.plotly_chart(fig, width='stretch')
    else:
        st.error("Column not found")

def show_comparison():
    st.markdown("---")
    st.info("Model Comparison - All models on same test set")
    features = ['temperature', 'humidity', 'hour', 'month']
    features = [f for f in features if f in df.columns]
    target = 'use [kW]'
    if target in df.columns and len(features) >= 2:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results = {}
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        results['Linear'] = r2_score(y_test, lr.predict(X_test))
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        results['Ridge'] = r2_score(y_test, ridge.predict(X_test))
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        results['RF'] = r2_score(y_test, rf.predict(X_test))
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'R2'])
        st.dataframe(results_df, width='stretch')
        fig = px.bar(results_df, x='Model', y='R2', color='R2', title="Model Comparison")
        st.plotly_chart(fig, width='stretch')
        best = results_df.loc[results_df['R2'].idxmax(), 'Model']
        st.success(f"Best Model: {best}")
    else:
        st.error("Missing columns")

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
