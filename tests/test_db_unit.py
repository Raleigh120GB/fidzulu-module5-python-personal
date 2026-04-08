import types
from fidzulu import db


class DummyCfg:
    user = "testuser"
    host = "db.example.com"
    port = 1521
    service_name = "ORCL"
    password = "secret"


def test_oracle_engine_builds_dsn_and_calls_create_engine(monkeypatch):
    captured = {}

    def fake_create_engine(url, pool_pre_ping=False):
        captured["url"] = url
        captured["pool_pre_ping"] = pool_pre_ping
        return "dummy_engine"

    # Patch config loader and create_engine used inside fidzulu.db
    monkeypatch.setattr(db, "load_db_config", lambda: DummyCfg())
    monkeypatch.setattr(db, "create_engine", fake_create_engine)

    # Replace logger so test output is clean
    class DummyLogger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

    monkeypatch.setattr(db, "logger", DummyLogger())

    engine = db.oracle_engine()

    assert engine == "dummy_engine"

    # Verify URL contains real credentials and DSN
    assert captured["url"].startswith(
        f"oracle+oracledb://{DummyCfg.user}:{DummyCfg.password}@/?dsn="
    )

    expected_dsn = (
        f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={DummyCfg.host})"
        f"(PORT={DummyCfg.port}))(CONNECT_DATA=(SERVICE_NAME={DummyCfg.service_name})))"
    )
    assert expected_dsn in captured["url"]
    assert captured["pool_pre_ping"] is True
