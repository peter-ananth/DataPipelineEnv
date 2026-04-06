---
title: Data Pipeline Cloud IDE
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Cloud IDE benchmark for AI on SQL & CSV tasks.
tags:
  - openenv
  - data-engineering
  - sql
  - csv
  - benchmark
---

# 🚀 DataPipelineEnv: The Cloud IDE Benchmark

> **Meta AI × Hugging Face OpenEnv Hackathon Submission**
> A rigorously-structured, deterministic benchmark environment testing LLMs against highly dynamic data engineering tasks.

---

## 🧠 What is this?
**DataPipelineEnv** is an OpenAI-gym style testing environment that evaluates your agent's ability to act as a **Data Engineer**. Instead of relying on static prompts or "one way" solutions, it maps to a randomized seed-generator, opening up a rigorous matrix of **32 distinctly challenging problem permutations**.

But checking endpoints over `curl` or REST isn't enough to stand out. 
We went a major step further: **We turned the backend into a fully-fledged Commercial Cloud IDE.**

---

## 💎 The Cloud IDE
When you visit the root URL, humans and inspecting agents are greeted with:
1. **VS Code Engine:** Native Microsoft Monaco Editor embedded natively via CDN, granting deep syntax highlighting and block generation in the browser.
2. **Glassmorphic Datagrids:** Raw CSV data bounds are intercepted algorithmically and converted into beautiful, responsive HTML layout matrices in real-time.
3. **The Live SQL Sandbox:** A natively integrated Endpoint `/sandbox` allows you to explore the in-memory SQLite schema utilizing raw `SELECT` commands *prior* to finalizing a submission. It creates a deeply interactive dual-state analytical workflow matching premium coding platforms like LeetCode or DataCamp.

---

## 📋 The 32-Variant Benchmark Matrix

We abandoned static tasks. Every new session locks to a random mathematical seed, drastically altering the constraints and correct answers natively.

### 🟢 Stage 1: Dynamic CSV Cleaning (Easy) - *5 Rule-Set Combinations*
Instead of static parsing tasks, the environment will randomly assign instructions. In some instances, it instructs the agent to fully `UPPERCASE` countries. In others, it strictly requests filtering rows based on price points. The Pandas determinator automatically calculates grading metrics relative to the assigned seed matrix.

### 🟡 Stage 2: SQLite Query Debugging (Medium) - *15 Systemic Architectural Bugs*
Agents face 15 deep-layer algorithmic SQL constraints ranging from:
* `HAVING` vs `WHERE` logic collapsing.
* Inverted Aggregations (`Missing GROUP BY arrays`).
* Hard-fail Operator Precedence conflicts (`Missing parenthesizes dictating AND/OR`).
* Null-pointer matching (`= NULL` vs `IS NULL`).

### 🔴 Stage 3: Reverse Engineering (Hard) - *12 Complex Model Replications*
Agents must view a randomized target tabular payload, and figure out exactly how to map `sales` schemas contextually. Spans over:
* YOY Case-Pivot arrays.
* Embedded Subqueries evaluating structural averages.
* Analytical Window Functions (`ROW_NUMBER() OVER(PARTITION BY...)`).

---

## 🎯 Native OpenEnv Adherence
Despite the vastly expanded scope and premium overlay, the `DataPipelineEnv` perfectly respects standard REST metrics mandated by the HuggingFace rules parameters:

### Action Dispatch Mapping
```json
{
  "type": "clean_csv", "payload": "<csv string>"
}
// OR
{
  "type": "submit_query", "payload": "SELECT..."
}
```

### Observation View Model
```json
{
  "task_id": "csv_cleaning",
  "task_description": "...",
  "data_preview": "...",
  "schema": "...",
  "attempt": 1,
  "max_attempts": 5
}
```

---

## 🚀 Easy Local Boot deployment

The architecture bounds simply over a single Docker container via FastAPI.

```bash
# Clone
git clone ...
cd data-pipeline-env

# Fast Boot
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Load `http://localhost:7860` in the browser to view the IDE natively, or test programmatic AI Agents over the standard `/reset` & `/step` parameters!

---
*Built for the OpenEnv Hackathon. Featuring pure deterministic Pandas & SQLite evaluation scoring matrix algorithms. Zero LLM API judge costs.*
