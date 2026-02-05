# tests/conftest.py
import pytest
from sqlalchemy import text
from fidzulu.db import oracle_engine
from pathlib import Path
from dotenv import load_dotenv

# Force loading .env from project root
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@pytest.fixture(scope="session")
def db_conn():
    engine = oracle_engine()
    with engine.connect() as conn:
        yield conn