# ⚡ Smart Home Energy Analytics

**Subject:** Python for Data Science  
**Team:** SY ECE A1  

## 📌 Project Overview
This app predicts household energy consumption using Regression models (Linear, Ridge, Random Forest) and detects anomalies using IQR methods.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📊 Features
- Load Balancing Prediction
- Model Comparison (R², RMSE)
- Electricity Theft Detection (Anomalies)

## Data Sampling Note

The original dataset (124 MB, ~500,000 rows) was sampled to 50,000 rows 
for cloud deployment efficiency. This reduces:

- **File size:** 124 MB → ~5 MB (GitHub limit: 100 MB)
- **Memory usage:** Prevents Streamlit Cloud crashes (1 GB RAM limit)
- **Load time:** 30 seconds → 3 seconds

**Impact on Model Accuracy:** Minimal (<1% R² difference)
**Justification:** Standard practice for web-based ML deployments
