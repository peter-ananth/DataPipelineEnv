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
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Validate required env vars
_MISSING = [k for k, v in {
    "API_BASE_URL": API_BASE_URL,
    "MODEL_NAME": MODEL_NAME,
    "HF_TOKEN": HF_TOKEN,
}.items() if not v]

if _MISSING:
    print(f"[ERROR] Missing required environment variables: {', '.join(_MISSING)}", file=sys.stderr)
    sys.exit(1)

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
    "csv_cleaning": """You are a data engineer expert. You will receive a dirty CSV string.
Clean it (remove duplicates, drop null customer_name rows, convert price to float, convert quantity to int, normalize country to Title Case).
Respond ONLY with a single markdown code block containing your cleaned CSV data. Do not use JSON wrappers.
Example format:
```csv
order_id,customer_name,item,price,quantity,country
1001,Alice,Laptop,1200.5,1,France
...
```""",

    "sql_fix": """You are a SQL expert. You will be given a broken SQL query and its error.
Fix the query. Respond ONLY with a single markdown code block containing valid SQL. Do not use JSON wrappers.
Example format:
```sql
SELECT ...
```""",

    "query_reverse": """You are a SQL expert. You will be given a database schema and an expected output table.
Write a SQL query that produces exactly that output.
Respond ONLY with a single markdown code block containing valid SQL. Do not use JSON wrappers.
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

def log_start(session_id: str, task_id: str):
    print(json.dumps({"type": "START", "session_id": session_id, "task_id": task_id}), flush=True)

def log_step(session_id: str, attempt: int, reward: float, done: bool):
    print(json.dumps({"type": "STEP", "session_id": session_id, "step": attempt, "reward": float(reward), "done": done}), flush=True)

def log_end(session_id: str, best_reward: float):
    print(json.dumps({"type": "END", "session_id": session_id, "reward": float(best_reward)}), flush=True)


def run_task(task_id: str, seed: int = 42) -> float:
    """Run one full episode for a task. Returns best reward achieved."""
    # Reset environment
    try:
        reset_resp = http.post("/reset", json={"task_id": task_id, "seed": seed})
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to reset: {e}", file=sys.stderr)
        return 0.0

    session_id = reset_data["session_id"]
    obs = reset_data["observation"]
    best_reward = 0.0
    done = False
    attempt = 0

    log_start(session_id, task_id)

    system_prompt = SYSTEM_PROMPTS.get(task_id, SYSTEM_PROMPTS["sql_fix"])
    action_type = ACTION_TYPES.get(task_id, "submit_query")

    while not done:
        attempt += 1

        # Build user message from observation
        user_msg = obs.get("task_description", "")
        if obs.get("error_message"):
            user_msg += f"\n\nPrevious error: {obs['error_message']}"

        # Call LLM
        llm_output = call_llm(system_prompt, user_msg)
        action = extract_action(llm_output, action_type)

        if not action:
            # Submit empty to consume attempt
            action = {"type": action_type, "payload": ""}

        # Submit action
        try:
            step_resp = http.post(
                f"/step/{session_id}",
                json={"type": action["type"], "payload": action["payload"]},
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as e:
            print(f"[ERROR] Step failed: {e}", file=sys.stderr)
            break

        reward = step_data["reward"]["value"]
        done = step_data["done"]
        obs = step_data["observation"]
        best_reward = max(best_reward, reward)

        log_step(session_id, attempt, reward, done)

        if done:
            break

        time.sleep(0.5)  # rate limit courtesy

    log_end(session_id, best_reward)

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
