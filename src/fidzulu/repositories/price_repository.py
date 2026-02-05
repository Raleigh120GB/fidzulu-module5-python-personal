# src/fidzulu/repositories/price_repository.py
from typing import Optional
#import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

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
        with self.engine.connect() as conn:
            results = conn.execute(query, {"cat_id": cat_id}).fetchall()

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

