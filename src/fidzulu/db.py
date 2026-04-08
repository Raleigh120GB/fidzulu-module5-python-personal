from sqlalchemy import create_engine
from fidzulu.config import load_db_config
from fidzulu.utils.logging import get_logger, log_anomaly
from fidzulu.exceptions import EngineCreationError

logger = get_logger(__name__)


def oracle_engine():
    cfg = load_db_config()
    # Log non-sensitive connection metadata only
    logger.info({
        "event": "engine_create_attempt",
        "db_user": cfg.user,
        "host": cfg.host,
        "port": cfg.port,
        "service_name": cfg.service_name,
    })

    # Build DSN for thin driver (password never logged)
    try:
        dsn = f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={cfg.host})(PORT={cfg.port}))(CONNECT_DATA=(SERVICE_NAME={cfg.service_name})))"
        url = f"oracle+oracledb://{cfg.user}:{cfg.password}@/?dsn={dsn}"
        logger.info("Creating Oracle engine with DSN (password redacted from log)")
        engine = create_engine(url, pool_pre_ping=True)
        logger.info("Oracle engine created successfully")
        return engine
    except Exception as exc:
        # Log detailed internal error, then raise a safe error for callers
        log_anomaly(logger, "failed to create oracle engine", {"error": str(exc)})
        raise EngineCreationError("Unable to create database engine; check configuration and credentials.")