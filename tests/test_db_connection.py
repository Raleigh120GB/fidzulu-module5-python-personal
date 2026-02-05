# from fidzulu.db import oracle_engine
# from fidzulu.config import load_db_config
# from sqlalchemy import text
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

# def main():
#     cfg = load_db_config()
#     print("Loaded config:", cfg)
#     engine = oracle_engine()
#     with engine.connect() as conn:
#         result = conn.execute(text("SELECT 1 FROM dual"))
#         print("DB connection test result:", result.scalar())

# if __name__ == "__main__":
#     main()

# tests/test_db_connection.py
from sqlalchemy import text


# tests/test_db_connection.py
from sqlalchemy import text

def test_dual(db_conn):
    assert db_conn.execute(text("SELECT 1 FROM dual")).scalar() == 1