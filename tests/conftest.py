# tests/conftest.py
import sys
from pathlib import Path

# Ensure project `src` directory is on sys.path so tests importing
# using the legacy `src.*` namespace resolve correctly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pytest
from sqlalchemy import text
from fidzulu.db import oracle_engine


@pytest.fixture(scope="session")
def db_conn():
    engine = oracle_engine()
    with engine.connect() as conn:
        yield conn