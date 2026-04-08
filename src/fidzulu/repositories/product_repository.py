# src/fidzulu/repositories/product_repository.py
from sqlalchemy.engine import Engine
from sqlalchemy import text
from fidzulu.utils.logging import get_logger, log_query_attempt, log_query_failure, log_empty_result
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

        for row in results:
            prod_id, prod_name, cat_id_val, brand_id, created_at = row

            if prod_id not in dataset:
                dataset[prod_id] = {
                    "names": [],
                    "brand_ids": [],
                    "created_at": []
                }

            dataset[prod_id]["names"].append(prod_name)
            dataset[prod_id]["brand_ids"].append(brand_id)
            dataset[prod_id]["created_at"].append(created_at)

        return dataset