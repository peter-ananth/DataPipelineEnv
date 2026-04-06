"""
Tests for env.py — DataPipelineEnv class.
Covers: reset idempotency, step shape, done logic, multi-attempt, error handling.
"""

from __future__ import annotations

import pytest

from app.env import DataPipelineEnv
from app.tasks import easy, medium, hard


@pytest.fixture
def env():
    e = DataPipelineEnv()
    yield e
    e.close()


class TestReset:

    def test_reset_returns_observation_shape(self, env):
        obs = env.reset(task_id="csv_cleaning")
        required_keys = {"task_id", "task_description", "data_preview", "schema", "attempt", "max_attempts"}
        assert required_keys.issubset(obs.keys())

    def test_reset_task_id_set(self, env):
        env.reset(task_id="csv_cleaning")
        assert env.state()["task_id"] == "csv_cleaning"

    def test_reset_attempt_starts_at_1(self, env):
        obs = env.reset(task_id="csv_cleaning")
        assert obs["attempt"] == 1

    def test_reset_is_idempotent(self, env):
        obs1 = env.reset(task_id="csv_cleaning", seed=42)
        obs2 = env.reset(task_id="csv_cleaning", seed=42)
        assert obs1["task_id"] == obs2["task_id"]
        assert obs1["attempt"] == obs2["attempt"]

    def test_reset_all_task_ids(self, env):
        for task_id in ["csv_cleaning", "sql_fix", "query_reverse"]:
            obs = env.reset(task_id=task_id)
            assert obs["task_id"] == task_id

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent_task")

    def test_reset_done_set_false(self, env):
        env.reset(task_id="csv_cleaning")
        assert env.state()["done"] is False


class TestStep:

    def test_step_returns_correct_shape(self, env):
        env.reset(task_id="csv_cleaning")
        obs, reward, done, info = env.step({"type": "clean_csv", "payload": ""})
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_reward_in_range(self, env):
        env.reset(task_id="csv_cleaning")
        _, reward, _, _ = env.step({"type": "clean_csv", "payload": "order_id,price\n1,bad"})
        assert 0.0 <= reward <= 1.0

    def test_step_wrong_action_type_gives_zero(self, env):
        env.reset(task_id="csv_cleaning")
        _, reward, _, info = env.step({"type": "submit_query", "payload": "SELECT 1"})
        assert reward == 0.0

    def test_step_on_done_episode_returns_done(self, env):
        env.reset(task_id="csv_cleaning")
        env._done = True  # force done state
        _, reward, done, info = env.step({"type": "clean_csv", "payload": ""})
        assert done is True
        assert "error" in info

    def test_step_increments_attempt(self, env):
        env.reset(task_id="csv_cleaning")
        env.step({"type": "clean_csv", "payload": ""})
        assert env.state()["attempt"] == 2

    def test_step_done_when_max_attempts_reached(self, env):
        env.reset(task_id="csv_cleaning")
        for _ in range(easy.MAX_ATTEMPTS):
            obs, reward, done, info = env.step({"type": "clean_csv", "payload": ""})
        assert done is True

    def test_step_done_on_perfect_score(self, env):
        """Perfect CSV submission should terminate the episode."""
        from app.tasks.easy import get_reference_df, get_dirty_csv_string
        env.reset(task_id="csv_cleaning")
        ref = get_reference_df()
        _, reward, done, _ = env.step({
            "type": "clean_csv",
            "payload": ref.to_csv(index=False),
        })
        assert reward == pytest.approx(1.0, abs=0.05)
        assert done is True

    def test_step_sql_fix_correct(self, env):
        env.reset(task_id="sql_fix", seed=2)  # syntax_error variant
        # The correct query for this variant fixes the missing comma
        _, reward, _, _ = env.step({
            "type": "submit_query",
            "payload": "SELECT 1",  # won't match, just test it returns float
        })
        assert 0.0 <= reward <= 1.0

    def test_step_no_crash_on_bad_sql(self, env):
        env.reset(task_id="sql_fix", seed=0)
        obs, reward, done, info = env.step({
            "type": "submit_query",
            "payload": "DROP TABLE orders; --",
        })
        assert 0.0 <= reward <= 1.0


class TestState:

    def test_state_returns_full_snapshot(self, env):
        env.reset(task_id="query_reverse", seed=0)
        state = env.state()
        required = {"task_id", "done", "attempt", "max_attempts", "last_reward", "best_reward", "episode_rewards"}
        assert required.issubset(state.keys())

    def test_state_episode_rewards_tracks_history(self, env):
        env.reset(task_id="csv_cleaning")
        env.step({"type": "clean_csv", "payload": ""})
        env.step({"type": "clean_csv", "payload": ""})
        state = env.state()
        assert len(state["episode_rewards"]) == 2

    def test_state_best_reward_is_max(self, env):
        env.reset(task_id="csv_cleaning")
        env.step({"type": "clean_csv", "payload": ""})
        state = env.state()
        assert state["best_reward"] == max(state["episode_rewards"])
