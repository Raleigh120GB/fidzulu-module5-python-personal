# src/fidzulu/utils/logging.py
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(ch)
    return logger


def _mask_params(params: dict) -> dict:
    if not isinstance(params, dict):
        return params
    masked = {}
    for k, v in params.items():
        key = str(k).lower()
        if "pass" in key or "password" in key or "secret" in key:
            masked[k] = "[REDACTED]"
        else:
            masked[k] = v
    return masked


def log_query_attempt(logger: logging.Logger, operation: str, query_name: str, params=None):
    logger.info({
        "event": "query_attempt",
        "operation": operation,
        "query": query_name,
        "params": _mask_params(params) if params is not None else None,
    })


def log_query_failure(logger: logging.Logger, operation: str, query_name: str, error: Exception, params=None):
    logger.error({
        "event": "query_failure",
        "operation": operation,
        "query": query_name,
        "params": _mask_params(params) if params is not None else None,
        "error": str(error),
    })


def log_empty_result(logger: logging.Logger, operation: str, query_name: str, params=None):
    logger.warning({
        "event": "empty_result",
        "operation": operation,
        "query": query_name,
        "params": _mask_params(params) if params is not None else None,
    })


def log_anomaly(logger: logging.Logger, message: str, details: dict = None):
    logger.warning({
        "event": "anomaly",
        "message": message,
        "details": details or {},
    })