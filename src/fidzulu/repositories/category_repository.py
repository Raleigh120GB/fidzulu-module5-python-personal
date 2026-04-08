# src/fidzulu/repositories/category_repository.py
from sqlalchemy import text
from fidzulu.utils.logging import get_logger, log_query_attempt, log_query_failure, log_empty_result
from fidzulu.exceptions import RepositoryError

logger = get_logger(__name__)

class CategoryRepository:
    def __init__(self, engine):
        self.engine = engine

    def get_category_id_by_name(self, keyword: str) -> dict:
        query = text("""
            SELECT cat_id
            FROM FIDZULU.categories
            WHERE LOWER(cat_name) = LOWER(:keyword)
        """)
        operation = "get_category_id_by_name"
        params = {"keyword": keyword}
        log_query_attempt(logger, operation, "category_by_name", params)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, params).fetchone()
        except Exception as exc:
            # Log detailed failure internally, then raise a safe error
            log_query_failure(logger, operation, "category_by_name", exc, params)
            raise RepositoryError("Failed to retrieve category id. Check database connectivity and credentials.")

        if not result:
            log_empty_result(logger, operation, "category_by_name", params)
            raise ValueError(f"No category found for keyword: {keyword}")

        return {"id": result[0]}
