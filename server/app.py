"""
server/app.py — OpenEnv multi-mode deployment entry point.
Required by openenv validate for server discovery.
Re-exports the FastAPI app from the main module.
"""
from app.main import app

__all__ = ["app"]
