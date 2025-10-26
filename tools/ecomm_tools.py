"""
The Following file contains helper tools/functions for 
cleaning and feature engineering the retail/e-commerce dataset.
"""

import pandas as pd

def preprocess(df):
    """Cleans and engineers key features."""   
    df = df.dropna(subset=['CustomerID', 'Quantity', 'UnitPrice'])
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    df['Invoice_Year'] = df['InvoiceDate'].dt.year
    df['Invoice_Month'] = df['InvoiceDate'].dt.month
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    if 'DeliveryDate' in df.columns:
        df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], errors='coerce')
        df['Delivery_Time'] = (df['DeliveryDate'] - df['InvoiceDate']).dt.days
        df['Delivery_Time'] = df['Delivery_Time'].fillna(0).clip(lower=0)
    else:
        df['Delivery_Time'] = 0

    return df