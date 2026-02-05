# src/fidzulu/repositories/category_repository.py
from sqlalchemy import text

class CategoryRepository:
    def __init__(self, engine):
        self.engine = engine

    def get_category_id_by_name(self, keyword: str) -> dict:
        query = text("""
            SELECT cat_id
            FROM FIDZULU.categories
            WHERE LOWER(cat_name) = LOWER(:keyword)
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"keyword": keyword}).fetchone()
            if not result:
                raise ValueError(f"No category found for keyword: {keyword}")
            # Return as dictionary
            return {"id": result[0]}
