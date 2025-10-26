## This file contains the Data cleaning as well as the Feature Engineering Prompt tasks

import pandas as pd
import os
from glob import glob
from datetime import datetime


data_dir = "data"
sampledata_path = os.path.join(data_dir, "ecomm_datasample.csv")
source_path = os.path.join(data_dir, "ecomm_data.csv")    # original data path
samples_dir = os.path.join(data_dir, "samples") 

keep_last_n = 5   #aiming to keep 5 archived samples

def old_arch_del():
    os.makedirs(samples_dir, exist_ok=True)
    files = sorted(glob(os.path.join(samples_dir, "ecomm_datasample_*.csv")))
    if len(files) > keep_last_n:
        for f in files[: len(files) - keep_last_n]:
            try:
                os.remove(f)
            except OSError:
                pass

def sampling_taskdata(sample_size: int = 1000, seed: int | None = None) -> pd.DataFrame:
    """Selects randomly 1000 samples from the original dataset"""
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"{source_path} not found. Please use the following link to access the data "
            "https://www.kaggle.com/datasets/carrie1/ecommerce-data" 
            "and save it under /data/")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    df = pd.read_csv(source_path, encoding="ISO-8859-1")
    rs = seed if seed is not None else (pd.Timestamp.now().value % 2**32)
    sample = df.sample(n=sample_size, random_state=rs)

    sample.to_csv(sampledata_path, index=False)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(samples_dir, f"ecomm_datasample_{ts}.csv")
    sample.to_csv(archive_path, index=False)

    old_arch_del()

    print(f"Latest Sample -> {sampledata_path}")
    print(f"Archived copy -> {archive_path}")
    return sample

def describe_task():
    '''
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
    '''
    return (
        "TASK: Prepare a retail transactions CSV for downstream modeling by cleaning it and "
        "engineering essential features.\n\n"
        "Correctness conditions used for evaluation:\n"
        "• All rows must have a valid CustomerID (no missing).\n"
        "• InvoiceDate must be parsed as a datetime type.\n"
        "• Quantity and UnitPrice must be strictly positive (remove invalid rows).\n"
        "• Include the following features derived from the data:\n"
        "  - Invoice_Year, Invoice_Month (from InvoiceDate)\n"
        "  - Revenue = Quantity * UnitPrice\n"
        "  - Delivery_Time = (DeliveryDate - InvoiceDate) in days; if DeliveryDate is absent or "
        "    missing, set Delivery_Time = 0.\n"
        "Return the cleaned DataFrame with these features."
    )
