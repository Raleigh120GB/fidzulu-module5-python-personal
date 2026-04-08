# src/fidzulu/repositories/product_repository.py
from sqlalchemy.engine import Engine
from sqlalchemy import text
from fidzulu.utils.logging import get_logger, log_query_attempt, log_query_failure, log_empty_result, log_anomaly
from fidzulu.exceptions import RepositoryError

logger = get_logger(__name__)

class ProductRepository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def get_products_by_category(self, cat_id: int) -> dict:
        sql = """
        SELECT prod_id, prod_name, cat_id, brand_id, prod_createdat
        FROM FIDZULU.products
        WHERE cat_id = :cat_id
        ORDER BY prod_id
        """
        operation = "get_products_by_category"
        params = {"cat_id": cat_id}
        log_query_attempt(logger, operation, "products_by_category", params)
        try:
            with self.engine.connect() as conn:
                results = conn.execute(text(sql), params).fetchall()
        except Exception as exc:
            log_query_failure(logger, operation, "products_by_category", exc, params)
            raise RepositoryError("Failed to fetch products. Check database connectivity and credentials.")

        if not results:
            log_empty_result(logger, operation, "products_by_category", params)

        dataset = {"CategoryID": cat_id}
        skipped = 0

        for row in results:
            prod_id, prod_name, cat_id_val, brand_id, created_at = row

            # Basic sanity checks
            if prod_name is None or str(prod_name).strip() == "":
                skipped += 1
                continue
            if created_at is None:
                skipped += 1
                continue

            if prod_id not in dataset:
                dataset[prod_id] = {
                    "names": [],
                    "brand_ids": [],
                    "created_at": []
                }

            dataset[prod_id]["names"].append(prod_name)
            dataset[prod_id]["brand_ids"].append(brand_id)
            dataset[prod_id]["created_at"].append(created_at)

        # Remove empty pids
        empty_pids = [p for p in list(dataset.keys()) if p != "CategoryID" and not dataset[p]["names"]]
        for p in empty_pids:
            del dataset[p]

        if skipped > 0:
            log_anomaly(logger, "filtered_invalid_product_rows", {"category": cat_id, "skipped_rows": skipped})

        if len(dataset) == 1:
            log_anomaly(logger, "no_valid_product_data", {"category": cat_id})
            raise RepositoryError("No valid product data returned for category")

        return dataset