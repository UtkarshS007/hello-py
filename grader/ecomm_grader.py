"""
This validates model performance(output dataframe after the preprocessing)
numerically i.e. Deterministic Reward Calculation
Since scores are deterministics rane is 0-1.
"""
import pandas as pd

def grading(df, reference_df):
    checks = 0
    total = 5

    # 1. Expected engineered columns exist
    expected_cols = {'Invoice_Year', 'Invoice_Month', 'Revenue'}
    if expected_cols.issubset(df.columns):
        checks += 1

    # 2. Revenue correctness (mean ±1%)
    ref_mean = (reference_df['Quantity'] * reference_df['UnitPrice']).mean()
    pred_mean = df['Revenue'].mean()
    if abs(pred_mean - ref_mean) / (ref_mean + 1e-9) < 0.01:
        checks += 1

    # 3. Nulls check
    if df[['Quantity', 'UnitPrice', 'InvoiceDate']].notnull().all().all():
        checks += 1

    # 4. Type validation
    if pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        checks += 1

    # 5. Delivery_Time (>= 0)
    if 'Delivery_Time' in df.columns and (df['Delivery_Time'] >= 0).all():
        checks += 1

    reward = round(checks / total, 2)
    print(f"Grading complete: {checks}/{total} checks passed → Reward = {reward}")
    return reward