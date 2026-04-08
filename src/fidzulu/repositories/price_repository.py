# src/fidzulu/repositories/price_repository.py

from typing import Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine
from fidzulu.utils.logging import get_logger, log_query_attempt, log_query_failure, log_empty_result
from fidzulu.exceptions import RepositoryError

logger = get_logger(__name__)

class PriceRepository:
    def __init__(self, engine):
        self.engine = engine

    def get_prices_by_category(self, cat_id: int) -> dict:
        query = text("""
            SELECT p.prod_id, pr.pri_baseprice, pr.pri_startdate, pr.pri_enddate
            FROM FIDZULU.prices pr
            JOIN FIDZULU.products p ON pr.prod_id = p.prod_id
            WHERE p.cat_id = :cat_id
            ORDER BY p.prod_id, pr.pri_startdate
        """)
        operation = "get_prices_by_category"
        params = {"cat_id": cat_id}
        log_query_attempt(logger, operation, "prices_by_category", params)
        try:
            with self.engine.connect() as conn:
                results = conn.execute(query, params).fetchall()
        except Exception as exc:
            log_query_failure(logger, operation, "prices_by_category", exc, params)
            raise RepositoryError("Failed to fetch prices. Check database connectivity and credentials.")

        if not results:
            log_empty_result(logger, operation, "prices_by_category", params)

        dataset = {"CategoryID": cat_id}

        for row in results:
            prod_id, base_price, start_date, end_date = row

            if prod_id not in dataset:
                dataset[prod_id] = {
                    "prices": [],
                    "start_dates": [],
                    "end_dates": []
                }

            dataset[prod_id]["prices"].append(base_price)
            dataset[prod_id]["start_dates"].append(start_date)
            dataset[prod_id]["end_dates"].append(end_date)

        return dataset

