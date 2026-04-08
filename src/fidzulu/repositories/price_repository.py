# src/fidzulu/repositories/price_repository.py

from typing import Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine
from fidzulu.utils.logging import get_logger, log_query_attempt, log_query_failure, log_empty_result, log_anomaly
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
        skipped = 0

        for row in results:
            prod_id, base_price, start_date, end_date = row

            # Basic sanity checks: non-null numeric price, valid dates, start<=end
            if base_price is None:
                skipped += 1
                continue
            try:
                price_val = float(base_price)
            except Exception:
                skipped += 1
                continue
            if price_val <= 0:
                skipped += 1
                continue

            if start_date is None or end_date is None:
                skipped += 1
                continue
            try:
                # rely on DB types; compare if possible
                if end_date < start_date:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            if prod_id not in dataset:
                dataset[prod_id] = {
                    "prices": [],
                    "start_dates": [],
                    "end_dates": []
                }

            dataset[prod_id]["prices"].append(price_val)
            dataset[prod_id]["start_dates"].append(start_date)
            dataset[prod_id]["end_dates"].append(end_date)

        # Remove any products that ended up with no valid rows
        empty_pids = [p for p in list(dataset.keys()) if p != "CategoryID" and not dataset[p]["prices"]]
        for p in empty_pids:
            del dataset[p]

        if skipped > 0:
            log_anomaly(logger, "filtered_invalid_price_rows", {"category": cat_id, "skipped_rows": skipped})

        if len(dataset) == 1:
            # Only CategoryID present => no valid product rows
            log_anomaly(logger, "no_valid_price_data", {"category": cat_id})
            raise RepositoryError("No valid price data returned for category")

        return dataset

