## This file contains the Data cleaning as well as the Feature Engineering Prompt tasks

import pandas as pd
import os

sampledata_path = "C:\Projects\hello-py\data\ecomm_datasample.csv"
source_path = "C:\Projects\hello-py\data\ecomm_data.csv"    # original data path 

def sampling_taskdata(sample_size = 1000, seed=None):
    """Selects randomly 1000 samples from the original dataset"""
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"{source_path}" not found. Please use the following link to access the data" 
            "https://www.kaggle.com/datasets/carrie1/ecommerce-data" 
            "and save it under /data/"
        )
    
    df = pd.read_csv(source_path, encoding="ISO-8859-1")
    if seed is not None:
        df = df.sample(n=sample_size, random_state=seed)
    else:
        df = df.sample(n=sample_size, random_state=pd.Timestamp.now().value % 2**32)
    os.makedirs("data", exist_ok=True)
    df.to_csv(sampledata_path, index=False)
    print(f"Sample size of {sample_size} observations creates at {sampledata_path}")
    return df

def describe_task():
    return(
        "TASK: Clean and engineer features from retail transactions.\n"
        "Steps:\n"
        "1. Remove rows with invalid Quantity/UnitPrice or missing CustomerID.\n"
        "2. Convert InvoiceDate to datetime.\n"
        "3. Create derived columns:\n"
        "   - Invoice_Year, Invoice_Month\n"
        "   - Revenue = Quantity * UnitPrice\n"
        "   - Delivery_Time = DeliveryDate - InvoiceDate (if DeliveryDate exists)\n"
        "4. Return cleaned DataFrame ready for grading."
    )    


