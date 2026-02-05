# tests/test_repositories.py
from src.fidzulu.db import oracle_engine
from src.fidzulu.repositories.price_repository import PriceRepository

def test_prices_by_category_returns_rows():
    engine = oracle_engine()
    repo = PriceRepository(engine)

    dataset = repo.get_prices_by_category(cat_id=13)  # Vegetables

    # Top-level structure
    assert isinstance(dataset, dict)
    assert "CategoryID" in dataset
    assert dataset["CategoryID"] == 13

    # Ensure at least one product exists
    product_ids = [key for key in dataset.keys() if key != "CategoryID"]
    assert len(product_ids) >= 1

    # Validate structure for the first product
    prod_id = product_ids[0]
    prod_data = dataset[prod_id]

    assert isinstance(prod_data, dict)
    assert {"prices", "start_dates", "end_dates"}.issubset(prod_data.keys())

    # Ensure parallel list lengths
    assert len(prod_data["prices"]) == len(prod_data["start_dates"]) == len(prod_data["end_dates"])
    assert len(prod_data["prices"]) >= 25  # Tomatoes or Carrots should have ~25 entries