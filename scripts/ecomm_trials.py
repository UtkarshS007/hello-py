"""
This file helps in running multiple trials of the ecommerce task 
prompts that we had set.
Each trial uses a new random 1000-row sample for authenticity.
"""

import pandas as pd
from tasks.ecomm_preprocess import sampling_taskdata, describe_task
from 