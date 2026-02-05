# tests/test_features.py
from src.fidzulu.db import oracle_engine
from src.fidzulu.repositories.price_repository import PriceRepository
from src.fidzulu.repositories.category_repository import CategoryRepository

def test_prices_by_category_returns_dict():
    engine = oracle_engine()
    cat_repo = CategoryRepository(engine)
    price_repo = PriceRepository(engine)

    # Dynamically resolve category id
    cat_id = cat_repo.get_category_id_by_name("Vegetables")["id"]

    dataset = price_repo.get_prices_by_category(cat_id=cat_id)

    # Assertions on top-level structure
    assert isinstance(dataset, dict)
    assert "CategoryID" in dataset
    assert dataset["CategoryID"] == cat_id

    # Ensure at least two products (Tomatoes + Carrots)
    product_ids = [key for key in dataset.keys() if key != "CategoryID"]
    assert len(product_ids) >= 2

    # Check structure for one product
    prod_id = product_ids[0]
    prod_data = dataset[prod_id]

    assert isinstance(prod_data, dict)
    assert {"prices", "start_dates", "end_dates"}.issubset(prod_data.keys())

    # Ensure parallel list lengths
    assert len(prod_data["prices"]) == len(prod_data["start_dates"]) == len(prod_data["end_dates"])
    assert len(prod_data["prices"]) >= 25  # Tomatoes or Carrots should have ~25 entries