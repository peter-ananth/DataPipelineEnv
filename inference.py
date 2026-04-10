"""
inference.py — Baseline agent script (REQUIRED in root by OpenEnv spec)

Uses OpenAI client pointed at API_BASE_URL env var.
Reads MODEL_NAME and HF_TOKEN from environment.
Runs all 3 tasks and prints final scores.

Usage:
    export API_BASE_URL=<LLM API endpoint>
    export MODEL_NAME=meta-llama/Llama-3-70b-instruct
    export HF_TOKEN=<your HF token>
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import httpx
from openai import OpenAI

# ─────────────────── Config from environment variables ───────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Validate required env vars
_CRITICAL = ["API_BASE_URL", "HF_TOKEN"]
_MISSING = [k for k in _CRITICAL if not os.environ.get(k)]

if _MISSING:
    print(f"[ERROR] Missing critical environment variables: {', '.join(_MISSING)}", file=sys.stderr)
    print(f"[INFO] MODEL_NAME is set to: {MODEL_NAME}", file=sys.stderr)
    # Note: We exit if API_BASE_URL or HF_TOKEN are missing as the agent cannot function.
    sys.exit(1)

if not os.environ.get("MODEL_NAME"):
    print(f"[WARN] MODEL_NAME not found in environment. Defaulting to: {MODEL_NAME}", file=sys.stderr)

# OpenAI client pointed at HF Inference endpoint
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# HTTP client for the environment API
http = httpx.Client(base_url=ENV_BASE_URL, timeout=60.0)


# ─────────────────── LLM agent call ──────────────────────────────────────────

def call_llm(system_prompt: str, user_message: str) -> str:
    """Call the LLM and return response text. Never raises."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"  [LLM error] {e}", file=sys.stderr)
        return ""


def extract_action(llm_output: str, expected_type: str) -> dict[str, Any] | None:
    """
    Extract the raw payload from a specific markdown code block.
    Builds the final environment JSON action automatically.
    """
    if "```" in llm_output:
        parts = llm_output.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # inside code block
                payload = part.strip()
                # Remove leading language tags if present
                if payload.lower().startswith("sql"):
                    payload = payload[3:].strip()
                elif payload.lower().startswith("csv"):
                    payload = payload[3:].strip()
                elif payload.lower().startswith("json"):
                    payload = payload[4:].strip()
                
                if payload:
                    return {"type": expected_type, "payload": payload}

    # Fallback: Just take the raw output if no code blocks are found
    stripped = llm_output.strip()
    if stripped:
        return {"type": expected_type, "payload": stripped}

    return None


# ─────────────────── System prompts per task ─────────────────────────────────

SYSTEM_PROMPTS = {
    "csv_cleaning": """You are an expert data engineer. You will receive:
1. A task description listing EXACTLY what cleaning steps to perform (follow them precisely).
2. A preview of the dirty CSV data.

Critical rules:
- Read and follow the task description exactly — it tells you whether to drop or fill nulls, which casing to use, etc.
- Price values may contain commas or quotes like "1,234.56" — strip them before converting to float.
- quantity must be an integer (use Int64 / int). Invalid values (empty/"invalid") should be handled per the instructions.
- country casing must match exactly what the task says: UPPERCASE, lowercase, or Title Case.
- Keep all original columns: order_id, customer_name, country, product, price, quantity, order_date.
- Do NOT add or remove columns.

Respond ONLY with a single markdown code block containing your cleaned CSV. No JSON wrapper, no explanation.
Example format:
```csv
order_id,customer_name,country,product,price,quantity,order_date
1001,Alice,UNITED STATES,Laptop,1200.5,2,2024-03-01
```""",

    "sql_fix": """You are a SQL expert. You will be given a broken SQLite query and its error message.
Fix ONLY the specific error described. Do not restructure or rewrite the rest of the query.
Respond ONLY with a single markdown code block containing the corrected SQL. No explanation.
Example format:
```sql
SELECT ...
```""",

    "query_reverse": """You are a SQLite expert. You will receive a database schema and an expected output table.
Write a single SQL query that produces EXACTLY that output — same columns, same column names, same row order.
Use aggregations, window functions, CTEs, or CASE expressions as needed.
Respond ONLY with a single markdown code block containing valid SQL. No explanation.
Example format:
```sql
SELECT ...
```""",
}

ACTION_TYPES = {
    "csv_cleaning": "clean_csv",
    "sql_fix": "submit_query",
    "query_reverse": "submit_query",
}


# ─────────────────── Episode runner ──────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error is not None else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_task(task_id: str, seed: int = 42) -> float:
    """Run one full episode for a task. Returns best reward achieved."""
    # Reset environment
    try:
        reset_resp = http.post("/reset", json={"task_id": task_id, "seed": seed})
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to reset: {e}", file=sys.stderr)
        return 0.01

    session_id = reset_data["session_id"]
    obs = reset_data["observation"]
    best_reward = 0.01
    done = False
    attempt = 0
    rewards_list = []

    log_start(task=task_id, env="DataPipelineEnv", model=MODEL_NAME)

    system_prompt = SYSTEM_PROMPTS.get(task_id, SYSTEM_PROMPTS["sql_fix"])
    action_type = ACTION_TYPES.get(task_id, "submit_query")

    while not done:
        attempt += 1

        # Build user message from observation
        user_msg = obs.get("task_description", "")
        # Append dirty data preview so LLM can see actual values
        if obs.get("data_preview"):
            user_msg += f"\n\n**Dirty CSV preview:**\n```csv\n{obs['data_preview']}\n```"
        if obs.get("error_message"):
            user_msg += f"\n\n**Previous attempt error:** {obs['error_message']}"

        # Call LLM
        llm_output = call_llm(system_prompt, user_msg)
        action = extract_action(llm_output, action_type)

        if not action:
            # Submit empty to consume attempt
            action = {"type": action_type, "payload": ""}

        # Submit action
        error_msg = None
        try:
            step_resp = http.post(
                f"/step/{session_id}",
                json={"type": action["type"], "payload": action["payload"]},
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Step failed: {e}", file=sys.stderr)
            break

        reward = step_data["reward"]["value"]
        done = step_data["done"]
        obs = step_data["observation"]
        best_reward = max(best_reward, reward)
        rewards_list.append(reward)

        log_step(step=attempt, action=repr(action["payload"][:50]), reward=reward, done=done, error=error_msg)

        if done:
            break

        time.sleep(0.5)  # rate limit courtesy

    success = best_reward > 0.01
    log_end(success=success, steps=attempt, score=best_reward, rewards=rewards_list)

    # Clean up session
    try:
        http.delete(f"/session/{session_id}")
    except Exception:
        pass

    return best_reward


# ─────────────────── Main ────────────────────────────────────────────────────

def main() -> None:
    # Verify env is reachable
    try:
        health = http.get("/health")
        health.raise_for_status()
    except Exception as e:
        print(f"[FATAL] Environment not reachable at {ENV_BASE_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    tasks = ["csv_cleaning", "sql_fix", "query_reverse"]

    for task_id in tasks:
        try:
            run_task(task_id, seed=42)
        except Exception as e:
            print(f"[ERROR] Task {task_id} crashed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
