# RETAIL DATA CLEANING & FEATURE ENGINEERING

This readme file is based on the RL assessment. In this particular case the project belongs to Utkarsh Singh. This repo contains an **end-to-end Reinforcement Learning (RL)** task setup.
  
 ## Github Repo
 For the ease of access and to avoid confusion - The Original repo provided in the assessment was forked to my Personal Repo: https://github.com/UtkarshS007
 
All the project development with respect to my assessment will be in this particular folder. 
### Repo Structure 


## Dataset for this Project 

The Data was sourced from the UCI repository. The Data can be sourced from the following link:
"https://www.kaggle.com/datasets/carrie1/ecommerce-data".

## Approach to the Problem
The cleaning and transformation involved:

   1. Ensuring all rows had valid CustomerID values.
   2. Parsing InvoiceDate as a datetime type.
   3. Removing rows where Quantity or UnitPrice ≤ 0.
   4. Engineering derived columns:
        - Invoice_Year, Invoice_Month (from InvoiceDate)
        - Revenue = Quantity * UnitPrice
        - Delivery_Time = (DeliveryDate − InvoiceDate) in days
            
            → Default Delivery_Time = 0 if DeliveryDate is missing.

### Phase 1 - Initial Grader-Based Approach
            