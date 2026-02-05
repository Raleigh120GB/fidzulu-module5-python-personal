# src/fidzulu/config.py
import os
from dataclasses import dataclass
from src.fidzulu.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class DBConfig:
    host: str
    port: int
    service_name: str
    user: str
    password: str

def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        logger.error(f"Missing required environment variable: {name}")
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

def load_db_config() -> DBConfig:
    host = os.getenv("FIDZULU_DB_HOST", "localhost")
    port_str = os.getenv("FIDZULU_DB_PORT")
    if not port_str:
        # Make the default explicit for port only
        port = 1521
    else:
        port = int(port_str)

    # Require the pluggable service and credentials — no XE fallback
    service = _require("FIDZULU_DB_SERVICE")
    user = _require("FIDZULU_DB_USER")
    password = _require("FIDZULU_DB_PASSWORD")

    cfg = DBConfig(host=host, port=port, service_name=service, user=user, password=password)
    logger.info(f"Loaded DBConfig: host={cfg.host}, port={cfg.port}, service={cfg.service_name}, user={cfg.user}")
    return cfg