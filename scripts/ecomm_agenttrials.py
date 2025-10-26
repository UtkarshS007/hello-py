"""
scripts/ecomm_agenttrials.py
Runs an Anthropic agent through your retail cleaning task N times and reports pass-rate.

It:
  • Samples a fresh 1000-row subset each trial (tasks/ecomm_preprocess.py)
  • Prompts the agent to use the python_expression tool to clean the data
  • Requires the agent to save the cleaned CSV to out/cleaned.csv
  • Grades the cleaned file against the sampled reference using outcome checks
  • Counts trial success if reward >= 0.6 (>= 5/8 checks)
"""

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import asyncio
import pandas as pd
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Load env (so ANTHROPIC_API_KEY from .env works even if main.py doesn't load it)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Importing the tool loop + handlers from main.py
from main import run_agent_loop, python_expression_tool, submit_answer_tool

# task + grader
from tasks.ecomm_preprocess import sampling_taskdata as prepare_task_data, describe_task
from grader.ecomm_grader import grading


OUT_DIR = "out"
CLEANED_PATH = os.path.join(OUT_DIR, "cleaned.csv")

'''
def build_prompt() -> str:
    # Prompt that exactly mirrors the checks enforced by the grader (8 checks)
    return f"""
You are given a CSV file at: data/ecomm_datasample.csv

CLEANING REQUIREMENTS (must match exactly what the grader checks):
1) Drop rows where CustomerID is missing (NaN).
2) Convert InvoiceDate to datetime (rows with unparsable dates may be dropped).
3) Drop rows where Quantity <= 0.
4) Drop rows where UnitPrice <= 0.
5) Create derived columns:
   - Invoice_Year = year(InvoiceDate) as int
   - Invoice_Month = month(InvoiceDate) as int (1..12)
   - Revenue = Quantity * UnitPrice
   - Delivery_Time = (DeliveryDate - InvoiceDate) in days.
     If DeliveryDate column is absent or value missing, set Delivery_Time = 0.

INSTRUCTIONS:
• Use the python_expression tool to write Python (with pandas) that:
  - reads "data/ecomm_datasample.csv",
  - applies the above transformations,
  - saves the cleaned DataFrame to "{CLEANED_PATH}" (create the 'out' folder if needed).
• After saving the file, call the submit_answer tool with this JSON:
  {{ "answer": "{CLEANED_PATH}" }}

IMPORTANT:
• Do not print the entire DataFrame; printing is optional (small summaries ok).
• Your final step must be a submit_answer tool call with the file path shown above.
"""
'''
def build_prompt() -> str:
    return f"""
You are given a CSV file at: data/ecomm_datasample.csv

Goal:
Prepare the data for downstream modeling by cleaning it and engineering essential features.

Correctness conditions (this is how your output will be evaluated):
• All rows must have a valid CustomerID (no missing).
• InvoiceDate must be a proper datetime type.
• Quantity and UnitPrice must be strictly positive (invalid rows removed).
• Include these features in the final data:
  - Invoice_Year, Invoice_Month (derived from InvoiceDate)
  - Revenue = Quantity * UnitPrice
  - Delivery_Time = (DeliveryDate - InvoiceDate) in days; if DeliveryDate is absent or missing,
    set Delivery_Time = 0.

What to do:
• Use the python_expression tool to write Python (pandas) that:
  - reads "data/ecomm_datasample.csv",
  - applies the above requirements,
  - writes the final cleaned DataFrame to "{CLEANED_PATH}" (create the 'out' folder if needed).

Submission:
• After saving the file, call the submit_answer tool with:
  {{ "answer": "{CLEANED_PATH}" }}
"""

def build_tools_and_handlers():
    # Tools descriptors must match what main.run_agent_loop expects (same schemas)
    tools = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python passed to exec(). Use print(...) for stdout. Returns stdout."
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]
    handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }
    return tools, handlers


async def run_one_trial(trial_idx: int) -> tuple[float, bool]:
    print(f"\n Trial {trial_idx}")

    # 1) Prepare fresh sample + reference
    ref_df = prepare_task_data(sample_size=1000)  # saves data/ecomm_datasample.csv
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(CLEANED_PATH):
        try:
            os.remove(CLEANED_PATH)
        except OSError:
            pass

    # 2) Build prompt & tools
    prompt = build_prompt()
    tools, handlers = build_tools_and_handlers()

    # 3) Run the agent loop (uses Anthropic model via AsyncAnthropic inside main.run_agent_loop)
    submitted = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=handlers,
        max_steps=8,                  # allow a few tool calls
        model="claude-3-5-haiku-latest",
        verbose=False,
    )

    # 4) Validate the submitted artifact path and grade
    if not (isinstance(submitted, str) and os.path.exists(submitted)):
        print("✗ No valid file path submitted; reward = 0.0")
        return 0.0, False

    try:
        df_clean = pd.read_csv(submitted, encoding="ISO-8859-1")
    except Exception as e:
        print(f"✗ Could not load submitted file: {e}; reward = 0.0")
        return 0.0, False

    reward = grading(df_clean, ref_df)
    #success = reward >= 0.6  # 5/8 checks
    #success = reward >= 0.875   #8/8 checks - tighten the grader tolerance
    #print(f"{'✓' if success else '✗'} Trial {trial_idx} → Reward={reward}")
    SUCCESS_THRESHOLD = 0.875
    success = reward >= SUCCESS_THRESHOLD
    print(f"{'✓' if success else '✗'} Trial {trial_idx} → Reward={reward} (threshold={SUCCESS_THRESHOLD})")
    return reward, success


async def main(num_trials: int = 10):
    print(describe_task())
    rewards = []
    successes = 0

    for i in range(1, num_trials + 1):
        r, ok = await run_one_trial(i)
        rewards.append(r)
        successes += int(ok)

    avg_reward = round(sum(rewards) / len(rewards), 2)
    pass_rate = round(100 * successes / num_trials, 1)

    print("\n" + "=" * 60)
    print(f"Trials: {num_trials}")
    print(f"Passed: {successes}/{num_trials}")
    print(f"Average Reward: {avg_reward}")
    print(f"Pass Rate (reward>=0.6): {pass_rate}%")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
