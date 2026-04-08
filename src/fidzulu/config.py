# src/fidzulu/config.py
import os
import json
from dataclasses import dataclass
from fidzulu.utils.logging import get_logger

logger = get_logger(__name__)


def _load_secret_file(path: str) -> dict:
    """Load secrets from a file path. Supports JSON or simple KEY=VALUE lines.

    This allows mounting secrets from a secrets manager (Kubernetes, Docker, etc.)
    as files and keeps credentials out of source code.
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
            try:
                return json.loads(text)
            except Exception:
                # fallback: parse KEY=VALUE lines
                result = {}
                for line in text.splitlines():
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    result[k.strip()] = v.strip()
                return result
    except Exception:
        return {}

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
    # First try loading secrets from a mounted secret file (recommended)
    secret_file = os.getenv("FIDZULU_DB_SECRET_FILE")
    secret_vals = _load_secret_file(secret_file) if secret_file else {}

    host = secret_vals.get("FIDZULU_DB_HOST") or os.getenv("FIDZULU_DB_HOST", "localhost")
    port_str = secret_vals.get("FIDZULU_DB_PORT") or os.getenv("FIDZULU_DB_PORT")
    if not port_str:
        port = 1521
    else:
        port = int(port_str)

    # Require the pluggable service and credentials — no XE fallback
    service = secret_vals.get("FIDZULU_DB_SERVICE") or _require("FIDZULU_DB_SERVICE")
    user = secret_vals.get("FIDZULU_DB_USER") or _require("FIDZULU_DB_USER")
    password = secret_vals.get("FIDZULU_DB_PASSWORD") or _require("FIDZULU_DB_PASSWORD")

    cfg = DBConfig(host=host, port=port, service_name=service, user=user, password=password)
    # Log only non-sensitive metadata. Do NOT log passwords or full connection strings.
    logger.info(f"Loaded DBConfig: host={cfg.host}, port={cfg.port}, service={cfg.service_name}")
    return cfg