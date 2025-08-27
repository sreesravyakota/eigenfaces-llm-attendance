# backend/app/core/config.py
import os
from pathlib import Path

# Resolve a default SQLite path at project root (adjust if you prefer)
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # -> points to repo root
DEFAULT_DB = PROJECT_ROOT / "app.db"

def get_database_url() -> str:
    # Prefer env var, else fallback to local sqlite file
    return os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB}")
