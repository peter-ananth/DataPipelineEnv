"""
FastAPI application — DataPipelineEnv REST API
Wraps DataPipelineEnv with session management (UUID-based).
Endpoints: POST /reset, POST /step, GET /state/{session_id}, GET /tasks, GET /health
"""

from __future__ import annotations

import uuid
import pathlib
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.env import DataPipelineEnv
from app.tasks import easy, medium, hard


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Models — OpenEnv-compliant typed shapes
# ──────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    task_description: str
    data_preview: str
    schema_: str = Field(alias="schema")
    error_message: Optional[str] = None
    attempt: int
    max_attempts: int

    model_config = {"populate_by_name": True}


class Action(BaseModel):
    type: str = Field(
        ...,
        pattern="^(clean_csv|submit_query)$",
        description="Action type: 'clean_csv' for Task 1, 'submit_query' for Tasks 2 & 3",
    )
    payload: str = Field(..., description="CSV string or SQL query string")


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    partial_score: dict[str, Any]


class StepResult(BaseModel):
    observation: dict[str, Any]
    reward: Reward
    done: bool
    info: dict[str, Any]


class SandboxRequest(BaseModel):
    query: str


class SandboxResponse(BaseModel):
    success: bool = False
    error: Optional[str] = None
    columns: Optional[list[str]] = None
    data: Optional[list[list[Any]]] = None
    row_count: Optional[int] = None


class ResetRequest(BaseModel):
    task_id: str = Field(
        default=easy.TASK_ID,
        description="One of: csv_cleaning, sql_fix, query_reverse",
    )
    seed: Optional[int] = Field(default=None, description="Optional seed for reproducibility")


class ResetResponse(BaseModel):
    session_id: str
    observation: dict[str, Any]


class StateResponse(BaseModel):
    session_id: str
    state: dict[str, Any]


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    max_attempts: int
    action_type: str
    reward_range: list[float]


class HealthResponse(BaseModel):
    status: str
    version: str
    sessions_active: int


# ──────────────────────────────────────────────────────────────────────────────
# Session Store (in-memory)
# ──────────────────────────────────────────────────────────────────────────────

_sessions: dict[str, DataPipelineEnv] = {}

TASK_CATALOG: list[TaskInfo] = [
    TaskInfo(
        id=easy.TASK_ID,
        name=easy.TASK_NAME,
        difficulty=easy.DIFFICULTY,
        description="Fix a dirty e-commerce orders CSV: remove duplicates, handle nulls, fix data types, normalize country casing.",
        max_attempts=easy.MAX_ATTEMPTS,
        action_type="clean_csv",
        reward_range=[0.0, 1.0],
    ),
    TaskInfo(
        id=medium.TASK_ID,
        name=medium.TASK_NAME,
        difficulty=medium.DIFFICULTY,
        description="Fix a broken SQL query (syntax error, wrong JOIN, missing WHERE, or wrong column) to match expected results.",
        max_attempts=medium.MAX_ATTEMPTS,
        action_type="submit_query",
        reward_range=[0.0, 1.0],
    ),
    TaskInfo(
        id=hard.TASK_ID,
        name=hard.TASK_NAME,
        difficulty=hard.DIFFICULTY,
        description="Write a SQL query from scratch that produces the shown expected output table. Exact match required for full score.",
        max_attempts=hard.MAX_ATTEMPTS,
        action_type="submit_query",
        reward_range=[0.0, 1.0],
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean up sessions on shutdown."""
    yield
    for env in _sessions.values():
        env.close()
    _sessions.clear()


app = FastAPI(
    title="DataPipelineEnv",
    description=(
        "OpenEnv environment for debugging data pipelines. "
        "Agents clean messy CSVs and fix broken SQL queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    """Serve the Vanilla HTML frontend."""
    html_path = pathlib.Path(__file__).parent / "frontend.html"
    if not html_path.exists():
        return HTMLResponse(content="<h1>Frontend mapping error. frontend.html not found.</h1>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Docker health check + liveness probe."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        sessions_active=len(_sessions),
    )


@app.get("/tasks", response_model=list[TaskInfo], tags=["Environment"])
async def list_tasks() -> list[TaskInfo]:
    """List all available tasks with descriptions and metadata."""
    return TASK_CATALOG


@app.post("/reset", response_model=ResetResponse, status_code=status.HTTP_201_CREATED, tags=["Environment"])
async def reset(request: Optional[ResetRequest] = None) -> ResetResponse:
    """
    Start a new episode. Returns a session_id and initial observation.
    Each call creates a fresh environment instance.
    """
    if request is None:
        request = ResetRequest()
        
    valid_ids = {t.id for t in TASK_CATALOG}
    if request.task_id not in valid_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid task_id '{request.task_id}'. Choose from: {sorted(valid_ids)}",
        )

    session_id = str(uuid.uuid4())
    env = DataPipelineEnv()
    try:
        observation = env.reset(task_id=request.task_id, seed=request.seed)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize environment: {e}",
        )

    _sessions[session_id] = env
    
    # Bounded cache to prevent OOM
    if len(_sessions) > 500:
        oldest_id = next(iter(_sessions))
        oldest_env = _sessions.pop(oldest_id)
        oldest_env.close()

    return ResetResponse(session_id=session_id, observation=observation)


@app.post("/step/{session_id}", response_model=StepResult, tags=["Environment"])
async def step(session_id: str, action: Action) -> StepResult:
    """
    Submit an action for the active episode.
    Returns observation, reward (0.0–1.0), done flag, and partial score info.
    """
    env = _get_session(session_id)

    obs, reward_val, done, info = env.step(
        {"type": action.type, "payload": action.payload}
    )

    return StepResult(
        observation=obs,
        reward=Reward(
            value=round(reward_val, 4),
            partial_score=info.get("partial_score", {}),
        ),
        done=done,
        info={k: v for k, v in info.items() if k not in ("partial_score",)},
    )


@app.post("/sandbox/{session_id}", response_model=SandboxResponse, tags=["Environment"])
async def run_sandbox(session_id: str, req: SandboxRequest) -> SandboxResponse:
    """Run a scratchpad SELECT query against the live database."""
    env = _get_session(session_id)
    result = env.sandbox_query(req.query)
    
    if "error" in result:
        return SandboxResponse(success=False, error=result["error"])
        
    return SandboxResponse(
        success=True,
        columns=result["columns"],
        data=result["data"],
        row_count=result["row_count"]
    )


@app.get("/state/{session_id}", response_model=StateResponse, tags=["Environment"])
async def get_state(session_id: str) -> StateResponse:
    """Return the full current state snapshot for a session."""
    env = _get_session(session_id)
    return StateResponse(session_id=session_id, state=env.state())


@app.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["System"])
async def close_session(session_id: str) -> None:
    """Explicitly close and delete a session to free resources."""
    env = _get_session(session_id)
    env.close()
    del _sessions[session_id]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> DataPipelineEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Call POST /reset first.",
        )
    return env
