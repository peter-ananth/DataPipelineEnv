"""
DataPipelineEnv — Core OpenEnv Environment
Implements step() / reset() / state() compliant with OpenEnv spec.
"""

from __future__ import annotations

import io
import sqlite3
from typing import Any

import pandas as pd

from app import grader
from app.tasks import easy, medium, hard


class DataPipelineEnv:
    """
    OpenEnv-compliant environment for data pipeline debugging tasks.

    Episode lifecycle:
        reset(task_id) → observation
        step(action)   → (observation, reward, done, info)  [repeatable]
        state()        → full current state snapshot
    """

    SUPPORTED_TASKS = [easy.TASK_ID, medium.TASK_ID, hard.TASK_ID]

    def __init__(self) -> None:
        self._task_id: str | None = None
        self._attempt: int = 0
        self._max_attempts: int = 1
        self._done: bool = True
        self._current_observation: dict[str, Any] = {}
        self._last_reward: float = 0.0
        self._episode_rewards: list[float] = []
        self._info: dict[str, Any] = {}

        # Task-specific state
        self._easy_ref_df: pd.DataFrame | None = None
        self._medium_conn: sqlite3.Connection | None = None
        self._medium_bug: dict | None = None
        self._medium_expected_df: pd.DataFrame | None = None
        self._hard_conn: sqlite3.Connection | None = None
        self._hard_target: dict | None = None
        self._hard_expected_df: pd.DataFrame | None = None
        self._episode_seed: int | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str = easy.TASK_ID,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """
        Start a new episode for the given task.

        Args:
            task_id: One of 'csv_cleaning', 'sql_fix', 'query_reverse'
            seed: Optional seed for deterministic bug/target selection

        Returns:
            Initial observation dict.
        """
        if task_id not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Choose from: {self.SUPPORTED_TASKS}"
            )

        self._task_id = task_id
        self._attempt = 1
        self._done = False
        self._last_reward = 0.0
        self._episode_rewards = []
        self._episode_seed = seed
        self._info = {}

        if task_id == easy.TASK_ID:
            self._max_attempts = easy.MAX_ATTEMPTS
            self._easy_ref_df = easy.get_reference_df(seed=seed)
            obs = easy.get_initial_observation(seed=seed)

        elif task_id == medium.TASK_ID:
            self._max_attempts = medium.MAX_ATTEMPTS
            self._medium_conn = medium.setup_database(seed=seed)
            self._medium_bug = medium.get_random_bug(seed=seed)
            self._medium_expected_df = medium.get_expected_df(
                self._medium_bug["correct_query"], self._medium_conn
            )
            obs = medium.get_initial_observation(self._medium_bug, self._medium_conn)

        else:  # hard
            self._max_attempts = hard.MAX_ATTEMPTS
            self._hard_conn = hard.setup_database(seed=seed)
            self._hard_target = hard.get_target(seed=seed)
            self._hard_expected_df = hard.get_expected_df(
                self._hard_target, self._hard_conn
            )
            obs = hard.get_initial_observation(self._hard_target, self._hard_conn)

        obs["attempt"] = self._attempt
        self._current_observation = obs
        return obs

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict]:
        """
        Submit an action and receive feedback.

        Args:
            action: {"type": "clean_csv" | "submit_query", "payload": str}

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            obs = self._current_observation or {}
            return obs, 0.01, True, {"error": "Episode is done. Call reset() first."}
        if not self._task_id:
            return {}, 0.01, True, {"error": "No active task. Call reset() first."}

        # Validate action structure
        action_type = action.get("type", "")
        payload = action.get("payload", "")
        error_msg: str | None = None
        reward = 0.0

        try:
            if self._task_id == easy.TASK_ID:
                if action_type != "clean_csv":
                    error_msg = f"Task requires action type 'clean_csv', got '{action_type}'"
                    reward = 0.0
                else:
                    submitted_df = easy.parse_submission(payload)
                    reward = grader.grade_csv_clean(submitted_df, self._easy_ref_df)

            elif self._task_id == medium.TASK_ID:
                if action_type != "submit_query":
                    error_msg = f"Task requires action type 'submit_query', got '{action_type}'"
                    reward = 0.0
                else:
                    reward = grader.grade_sql_fix(
                        payload, self._medium_conn, self._medium_expected_df
                    )
                    if reward < 1.0 and self._medium_conn:
                        # Provide updated error context for next attempt
                        try:
                            result = pd.read_sql_query(payload, self._medium_conn)
                            error_msg = f"Query ran but produced {len(result)} rows vs expected {len(self._medium_expected_df)}"
                        except Exception as e:
                            error_msg = str(e)

            else:  # hard
                if action_type != "submit_query":
                    error_msg = f"Task requires action type 'submit_query', got '{action_type}'"
                    reward = 0.0
                else:
                    reward = grader.grade_query_reverse(
                        payload, self._hard_conn, self._hard_expected_df
                    )
                    if reward < 1.0 and self._hard_conn:
                        try:
                            result = pd.read_sql_query(payload, self._hard_conn)
                            error_msg = (
                                f"Your query returned {len(result)} rows with columns "
                                f"{list(result.columns)}. Expected {len(self._hard_expected_df)} rows."
                            )
                        except Exception as e:
                            error_msg = str(e)

        except Exception as e:
            error_msg = str(e)
            reward = 0.0

        reward = float(max(0.01, min(0.99, reward)))
        self._last_reward = reward
        self._episode_rewards.append(reward)
        self._attempt += 1

        done = reward >= 0.99 or self._attempt > self._max_attempts
        self._done = done

        obs = dict(self._current_observation)
        obs["attempt"] = self._attempt
        obs["error_message"] = error_msg
        self._current_observation = obs

        info = {
            "reward": reward,
            "partial_score": {
                "current": reward,
                "best_so_far": max(self._episode_rewards),
                "attempts_used": self._attempt - 1,
                "attempts_remaining": max(0, self._max_attempts - self._attempt + 1),
            },
            "done": done,
        }
        if done and reward >= 0.99:
            info["message"] = "🎉 Near-perfect score! Task complete."
        elif done:
            info["message"] = f"Episode over. Best score: {max(self._episode_rewards):.3f}"

        return obs, reward, done, info

    def sandbox_query(self, query: str) -> dict[str, Any]:
        """Execute a read-only query against the current task's database."""
        if not self._task_id:
            return {"error": "No active session task."}
        
        conn = None
        if self._task_id == medium.TASK_ID:
            conn = self._medium_conn
        elif self._task_id == hard.TASK_ID:
            conn = self._hard_conn
            
        if not conn:
            return {"error": f"Sandbox not supported for task {self._task_id}. SQL tasks only."}
            
        try:
            # Simple protection against destructive queries
            q_upper = query.strip().upper()
            if not q_upper.startswith(("SELECT", "WITH")):
                return {"error": "Sandbox only supports read-only SELECT or WITH queries."}
                
            df = pd.read_sql_query(query, conn)
            # Limit to 50 rows to prevent huge payloads crashing the browser
            df_head = df.head(50)
            return {
                "success": True,
                "columns": list(df_head.columns),
                "data": df_head.fillna("NULL").values.tolist(),
                "row_count": len(df)
            }
        except Exception as e:
            return {"error": str(e)}

    def state(self) -> dict[str, Any]:
        """Return full current state snapshot."""
        return {
            "task_id": self._task_id,
            "done": self._done,
            "attempt": self._attempt,
            "max_attempts": self._max_attempts,
            "last_reward": self._last_reward,
            "best_reward": max(self._episode_rewards) if self._episode_rewards else 0.01,
            "episode_rewards": list(self._episode_rewards),
            "current_observation": self._current_observation,
            "seed": self._episode_seed,
        }

    def close(self) -> None:
        """Release SQLite connections."""
        if self._medium_conn:
            self._medium_conn.close()
            self._medium_conn = None
        if self._hard_conn:
            self._hard_conn.close()
            self._hard_conn = None
