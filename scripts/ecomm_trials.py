"""
This file helps in running multiple trials of the ecommerce task 
prompts that we had set.
Each trial uses a new random 1000-row sample for authenticity.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#The above "Path Shim" helps make the repo importable no matter where this file 
#will be run from 
import pandas as pd
from tasks.ecomm_preprocess import sampling_taskdata, describe_task
from tools.ecomm_tools import preprocess
from grader.ecomm_grader import grading

def trials(trial_num: int) -> float:
    print(f"\n Running Trial {trial_num}")
    df_raw = sampling_taskdata(sample_size=1000)
    ref_df = df_raw.copy()
    cleaned_df = preprocess(df_raw)
    reward = grading(cleaned_df, ref_df)
    return reward

def main(num_trials: int=5) -> None:
    print(describe_task())
    rewards = [trials(i+1) for i in range(num_trials)]
    avg_reward = round(sum(rewards)/len(rewards), 2)
    print(f"\n Average Reward over {num_trials} trials: {avg_reward}")

if __name__ == "__main__":
    main()
