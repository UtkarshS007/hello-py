"""
This validates model performance(output dataframe after the preprocessing)
numerically i.e. Deterministic Reward Calculation
Since scores are deterministics rane is 0-1.
"""
import pandas as pd
import numpy as np

def grading(df, reference_df):
    checks = 0
    total = 8

    # 1. Expected engineered columns exist
    expected_cols = {'Invoice_Year', 'Invoice_Month', 'Revenue'}
    if expected_cols.issubset(df.columns):
        checks += 1

    # 2) InvoiceDate dtype
    if pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']): checks += 1

    # 3) Revenue correctness (mean within 1%)
    ref_mean = (reference_df['Quantity'] * reference_df['UnitPrice']).mean()
    pred_mean = df['Revenue'].mean()
    if np.isfinite(ref_mean) and abs(pred_mean - ref_mean) / (abs(ref_mean) + 1e-9) < 0.01:
        checks += 1

    # 4) Delivery_Time valid (>=0)
    if 'Delivery_Time' in df.columns and (df['Delivery_Time'] >= 0).all(): checks += 1

    # 5) CustomerID non-null
    if 'CustomerID' in df.columns and df['CustomerID'].notna().all(): checks += 1

    # 6) Quantity > 0
    if (df['Quantity'] > 0).all(): checks += 1

    # 7) UnitPrice > 0
    if (df['UnitPrice'] > 0).all(): checks += 1

    # 8) Year/Month derived correctly
    yr_ok = (df['Invoice_Year'] == pd.to_datetime(df['InvoiceDate']).dt.year).all()
    mo_ok = (df['Invoice_Month'] == pd.to_datetime(df['InvoiceDate']).dt.month).all()
    if yr_ok and mo_ok: checks += 1

    reward = round(checks / total, 2)
    print(f" Grading complete: {checks}/{total} checks passed → Reward = {reward}")
    return reward