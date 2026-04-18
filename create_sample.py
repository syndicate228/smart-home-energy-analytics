import pandas as pd

# ─── LINE 1: Load your original large dataset ────────────────────────────────
# This reads your 124 MB file from your computer
df = pd.read_csv('HomeC.csv')

# ─── LINE 2: Keep only first 50,000 rows ─────────────────────────────────────
# This creates a smaller version (like copying first 50,000 lines in Excel)
sample_df = df.head(50000)

# ─── LINE 3: Save as new smaller file ────────────────────────────────────────
# This creates a NEW file called 'HomeC_sample.csv' (around 5-10 MB)
sample_df.to_csv('HomeC_sample.csv', index=False)

# ─── LINE 4: Print confirmation ──────────────────────────────────────────────
# This shows you the results in the terminal
print(f"Original rows: {len(df)}")
print(f"Sample rows: {len(sample_df)}")
print(f"Sample file created: HomeC_sample.csv")
