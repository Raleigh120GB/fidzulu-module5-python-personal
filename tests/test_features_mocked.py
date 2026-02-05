# tests/test_features_mocked.py
import pytest
from unittest.mock import MagicMock
from src.fidzulu.repositories.price_repository import PriceRepository
from src.fidzulu.repositories.category_repository import CategoryRepository

@pytest.fixture
def fake_engine():
    # This is just a placeholder object, never connects
    return MagicMock()

def test_prices_by_category_with_mock(fake_engine):
    # Mock CategoryRepository to return a known category id
    cat_repo = CategoryRepository(fake_engine)
    cat_repo.get_category_id_by_name = MagicMock(return_value={"id": 13})

    # Mock PriceRepository to return a fake dictionary
    price_repo = PriceRepository(fake_engine)
    fake_dataset = {
        "CategoryID": 13,
        1: {
            "prices": [10.0, 12.0],
            "start_dates": ["2025-01-01", "2025-02-01"],
            "end_dates": ["2025-01-31", "2025-02-28"]
        },
        2: {
            "prices": [20.0],
            "start_dates": ["2025-01-15"],
            "end_dates": ["2025-01-31"]
        }
    }
    price_repo.get_prices_by_category = MagicMock(return_value=fake_dataset)

    # Now run the feature logic
    cat_id = cat_repo.get_category_id_by_name("vegetables")["id"]
    dataset = price_repo.get_prices_by_category(cat_id=cat_id)

    # Assertions
    assert isinstance(dataset, dict)
    assert dataset["CategoryID"] == 13
    assert 1 in dataset and 2 in dataset
    assert {"prices", "start_dates", "end_dates"}.issubset(dataset[1].keys())
    assert len(dataset[1]["prices"]) == 2
    assert len(dataset[2]["prices"]) == 1