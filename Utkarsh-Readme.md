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
The grader imported the cleaned DataFrame returned by the task script.

It verified:

 - Presence of required columns.
 - Validity of column datatypes.
 - Logical correctness of computed fields.

Failures triggered detailed feedback messages rather than simple pass/fail signals, helping diagnose issues.

**Components:**

`data_cleaning_tools.py` → Contained helper utilities (e.g., date parsing, revenue computation).

`data_cleaning_task.py` → Defined the task prompt and processing logic.

`data_cleaning_grader.py` → Encoded outcome-based assertions.            

*This approach worked well for deterministic checks.
However, it lacked adaptive learning — i.e., the system didn’t improve or explore multiple strategies for achieving better performance over time.*

### Phase 2 - Agent Trial Integration
Once the basic grader validation was stable, we moved towards agent-based experimentation using the `agent_trial` setup.

The goal here was to make the process autonomous — allowing the agent to:

- Attempt multiple runs.
- Learn from grader feedback.
- Optimize its cleaning and transformation logic iteratively.

**Implementation Details:**

- Introduced `scripts/ecomm_agenttrials.py` as a driver script for executing multiple trials.

- Each trial used the grader’s output as a feedback signal.

- Added logging for:

    - Trial number
    - Pass/fail status
    - Detailed failure reasons (if any)

- Created an out/ directory (gitignored) to store:

  - Intermediate results
  - Trial summaries
  - Grader evaluations

*With `agent_trial`, we began to see measurable improvements in pass rates across multiple runs.
Agents were able to self-correct issues like:*

- *Missing derived columns*
- *Incorrect datetime parsing*
- *Handling of missing DeliveryDate fields*


🪶 Author

**Utkarsh Singh**

`Data Scientist` | M.S. Data Analytics Engineering, Northeastern University