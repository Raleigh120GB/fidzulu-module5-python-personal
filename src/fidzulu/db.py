from sqlalchemy import create_engine
from fidzulu.config import load_db_config
from src.fidzulu.utils.logging import get_logger

logger = get_logger(__name__)


def oracle_engine():
    cfg = load_db_config()
    logger.info(f"Creating Oracle engine for {cfg.user}@{cfg.host}:{cfg.port}/{cfg.service_name}")
    # Build DSN for thin driver
    dsn = f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={cfg.host})(PORT={cfg.port}))(CONNECT_DATA=(SERVICE_NAME={cfg.service_name})))"
    url = f"oracle+oracledb://{cfg.user}:{cfg.password}@/?dsn={dsn}"
    logger.debug(f"Oracle engine URL: {url}")  
    logger.info("Creating Oracle engine with DSN and user credentials")
    engine = create_engine(url, pool_pre_ping=True)
    logger.info("Oracle engine created successfully")   
    return engine