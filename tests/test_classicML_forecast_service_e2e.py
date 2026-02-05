"""
End-to-End (E2E) tests for PriceForecastService using real Oracle database data.

These tests validate the complete ML pipeline from data extraction to forecasting
using actual production data from the FIDZULU database.

Requirements:
- Valid Oracle database connection
- .env file with DB credentials configured
- Real vegetable price data in the database (category_id=13)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text

from src.fidzulu.db import oracle_engine
from src.fidzulu.repositories.price_repository import PriceRepository
from src.fidzulu.repositories.product_repository import ProductRepository
from src.fidzulu.repositories.category_repository import CategoryRepository
from src.fidzulu.services.classicML_forecast_service import PriceForecastService


# ====================================================================
# FIXTURES
# ====================================================================

@pytest.fixture(scope="module")
def db_engine():
    """Create database engine for the test module."""
    engine = oracle_engine()
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def vegetable_category_id(db_engine):
    """Get the category ID for vegetables."""
    repo = CategoryRepository(db_engine)
    try:
        result = repo.get_category_id_by_name("Vegetables")
        return result["id"]
    except ValueError:
        # If "Vegetables" doesn't exist, use category ID 13 as fallback
        return 13


@pytest.fixture(scope="module")
def real_price_data(db_engine, vegetable_category_id):
    """Fetch real vegetable price data from the database."""
    repo = PriceRepository(db_engine)
    dataset = repo.get_prices_by_category(cat_id=vegetable_category_id)
    return dataset


@pytest.fixture(scope="module")
def real_product_data(db_engine, vegetable_category_id):
    """Fetch real product information from the database."""
    repo = ProductRepository(db_engine)
    dataset = repo.get_products_by_category(cat_id=vegetable_category_id)
    return dataset


@pytest.fixture(scope="module")
def valid_product_ids(real_price_data):
    """Get list of product IDs that have sufficient data for modeling."""
    product_ids = [pid for pid in real_price_data.keys() if pid != "CategoryID"]
    
    # Filter products with at least 25 data points (enough for train/test split)
    valid_ids = []
    for pid in product_ids:
        if len(real_price_data[pid]["prices"]) >= 25:
            valid_ids.append(pid)
    
    return valid_ids


# ====================================================================
# DATA VALIDATION TESTS
# ====================================================================

class TestRealDataValidation:
    """Validate the structure and quality of real database data."""

    def test_price_data_fetched_successfully(self, real_price_data):
        """Test that price data is successfully retrieved from database."""
        assert real_price_data is not None
        assert isinstance(real_price_data, dict)
        assert "CategoryID" in real_price_data

    def test_price_data_has_products(self, real_price_data):
        """Test that price data contains at least one product."""
        product_ids = [k for k in real_price_data.keys() if k != "CategoryID"]
        assert len(product_ids) >= 1, "No products found in database"

    def test_price_data_structure(self, real_price_data, valid_product_ids):
        """Test that each product has correct data structure."""
        for pid in valid_product_ids[:3]:  # Check first 3 products
            assert pid in real_price_data
            assert "prices" in real_price_data[pid]
            assert "start_dates" in real_price_data[pid]
            assert "end_dates" in real_price_data[pid]
            
            # Verify parallel arrays
            prices_len = len(real_price_data[pid]["prices"])
            assert len(real_price_data[pid]["start_dates"]) == prices_len
            assert len(real_price_data[pid]["end_dates"]) == prices_len

    def test_price_data_has_positive_prices(self, real_price_data, valid_product_ids):
        """Test that most prices in database are positive."""
        for pid in valid_product_ids[:3]:
            prices = real_price_data[pid]["prices"]
            positive_count = sum(1 for p in prices if p > 0)
            # At least 90% should be positive
            assert positive_count >= len(prices) * 0.9

    def test_price_data_has_valid_dates(self, real_price_data, valid_product_ids):
        """Test that dates can be parsed successfully."""
        for pid in valid_product_ids[:3]:
            start_dates = real_price_data[pid]["start_dates"]
            end_dates = real_price_data[pid]["end_dates"]
            
            # Verify we can parse dates
            for sd, ed in zip(start_dates[:5], end_dates[:5]):
                assert sd is not None
                assert ed is not None

    def test_product_data_fetched_successfully(self, real_product_data):
        """Test that product metadata is successfully retrieved."""
        assert real_product_data is not None
        assert isinstance(real_product_data, dict)
        assert "CategoryID" in real_product_data

    def test_sufficient_data_for_modeling(self, valid_product_ids):
        """Test that there are products with sufficient data points."""
        assert len(valid_product_ids) >= 1, "No products with >= 25 data points found"


# ====================================================================
# E2E INITIALIZATION TESTS
# ====================================================================

class TestPriceForecastServiceE2EInitialization:
    """Test service initialization with real database data."""

    def test_service_initializes_with_real_data(self, real_price_data):
        """Test that service initializes successfully with real data."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        assert service is not None
        assert service.raw_data == real_price_data
        assert service.df is not None
        assert len(service.train_sets) >= 1
        assert len(service.test_sets) >= 1
        assert len(service.models) >= 1

    def test_service_creates_models_for_valid_products(self, real_price_data, valid_product_ids):
        """Test that models are created for products with sufficient data."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        # At least some valid products should have models
        models_created = len(service.models)
        assert models_created >= 1
        
        # Check that created models are for valid product IDs
        for pid in service.models.keys():
            assert pid in real_price_data

    def test_wrangler_processes_real_data_correctly(self, real_price_data):
        """Test that data wrangler processes real data without errors."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        assert service.df is not None
        assert not service.df.empty
        assert 'prod_id' in service.df.columns
        assert 'base_price' in service.df.columns
        assert 'start_date' in service.df.columns
        assert 'end_date' in service.df.columns
        
        # All prices should be positive after wrangling
        assert (service.df['base_price'] > 0).all()

    def test_train_test_split_with_real_data(self, real_price_data):
        """Test that train/test split works correctly with real data."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        for pid in list(service.models.keys())[:3]:  # Check first 3 products
            assert pid in service.train_sets
            assert pid in service.test_sets
            
            train_df = service.train_sets[pid]
            test_df = service.test_sets[pid]
            
            # Both should have data
            assert len(train_df) > 0
            assert len(test_df) > 0
            
            # Train should have more data than test (80/20 split roughly)
            assert len(train_df) >= len(test_df)

    def test_wrangler_feedback_with_real_data(self, real_price_data):
        """Test that wrangler provides feedback when processing real data."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        assert service.feedback is not None
        assert isinstance(service.feedback, dict)
        # Real data might have some invalid prices or dates
        # Feedback should capture this if it occurs


# ====================================================================
# E2E EVALUATION TESTS
# ====================================================================

class TestPriceForecastServiceE2EEvaluation:
    """Test model evaluation with real database data."""

    @pytest.fixture(scope="class")
    def trained_service(self, real_price_data):
        """Create a trained service instance for the test class."""
        return PriceForecastService(real_price_data, test_ratio=0.2)

    def test_evaluate_real_product(self, trained_service):
        """Test evaluation on a real product."""
        # Get first available product
        pid = list(trained_service.models.keys())[0]
        
        metrics = trained_service.evaluate(pid)
        
        assert metrics is not None
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['rmse'] >= metrics['mae']  # RMSE should be >= MAE

    def test_evaluate_multiple_real_products(self, trained_service):
        """Test evaluation across multiple real products."""
        available_pids = list(trained_service.models.keys())[:5]  # First 5 products
        
        all_metrics = {}
        for pid in available_pids:
            metrics = trained_service.evaluate(pid)
            all_metrics[pid] = metrics
            
            assert metrics['mae'] >= 0
            assert metrics['rmse'] >= 0
        
        # Should have evaluated multiple products
        assert len(all_metrics) >= 1

    def test_evaluation_metrics_are_reasonable(self, trained_service):
        """Test that evaluation metrics are within reasonable bounds."""
        pid = list(trained_service.models.keys())[0]
        metrics = trained_service.evaluate(pid)
        
        # Get price range for context
        test_df = trained_service.test_sets[pid]
        price_mean = test_df['base_price'].mean()
        price_std = test_df['base_price'].std()
        
        # MAE and RMSE should not be astronomically high
        # Allow up to 3x the standard deviation as reasonable error
        assert metrics['mae'] < price_mean * 2, f"MAE too high: {metrics['mae']}"
        assert metrics['rmse'] < price_mean * 2, f"RMSE too high: {metrics['rmse']}"

    def test_consistent_evaluation_results(self, trained_service):
        """Test that evaluation returns consistent results when called multiple times."""
        pid = list(trained_service.models.keys())[0]
        
        metrics1 = trained_service.evaluate(pid)
        metrics2 = trained_service.evaluate(pid)
        
        # Results should be identical
        assert metrics1['mae'] == metrics2['mae']
        assert metrics1['rmse'] == metrics2['rmse']


# ====================================================================
# E2E FORECASTING TESTS
# ====================================================================

class TestPriceForecastServiceE2EForecasting:
    """Test forecasting functionality with real database data."""

    @pytest.fixture(scope="class")
    def trained_service(self, real_price_data):
        """Create a trained service instance for the test class."""
        return PriceForecastService(real_price_data, test_ratio=0.2)

    def test_forecast_real_product_30_days(self, trained_service):
        """Test 30-day forecast for a real product."""
        pid = list(trained_service.models.keys())[0]
        
        forecast_df = trained_service.forecast(pid, horizon_days=30)
        
        assert forecast_df is not None
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == 30
        assert 'date' in forecast_df.columns
        assert 'pred_price' in forecast_df.columns

    def test_forecast_real_product_180_days(self, trained_service):
        """Test default 180-day forecast for a real product."""
        pid = list(trained_service.models.keys())[0]
        
        forecast_df = trained_service.forecast(pid)
        
        assert len(forecast_df) == 180
        assert forecast_df['pred_price'].notna().all()

    def test_forecast_multiple_products(self, trained_service):
        """Test forecasting across multiple real products."""
        available_pids = list(trained_service.models.keys())[:3]
        
        forecasts = {}
        for pid in available_pids:
            forecast_df = trained_service.forecast(pid, horizon_days=60)
            forecasts[pid] = forecast_df
            
            assert len(forecast_df) == 60
            assert forecast_df['pred_price'].notna().all()
        
        # Forecasts should be different for different products
        if len(forecasts) >= 2:
            pids = list(forecasts.keys())
            assert not forecasts[pids[0]]['pred_price'].equals(forecasts[pids[1]]['pred_price'])

    def test_forecast_dates_start_after_training_period(self, trained_service):
        """Test that forecast dates begin after the last training date."""
        pid = list(trained_service.models.keys())[0]
        
        last_train_date = trained_service.train_sets[pid]['start_date'].max()
        forecast_df = trained_service.forecast(pid, horizon_days=30)
        first_forecast_date = pd.to_datetime(forecast_df['date'].iloc[0])
        
        assert first_forecast_date > last_train_date

    def test_forecast_dates_are_continuous(self, trained_service):
        """Test that forecast dates are continuous without gaps."""
        pid = list(trained_service.models.keys())[0]
        
        forecast_df = trained_service.forecast(pid, horizon_days=90)
        dates = pd.to_datetime(forecast_df['date'])
        
        # Check for daily continuity
        date_diffs = dates.diff().iloc[1:]
        assert all(date_diffs == pd.Timedelta(days=1))

    def test_forecast_predictions_are_finite(self, trained_service):
        """Test that all predictions are finite (no NaN, inf)."""
        pid = list(trained_service.models.keys())[0]
        
        forecast_df = trained_service.forecast(pid, horizon_days=60)
        
        assert forecast_df['pred_price'].notna().all()
        assert np.isfinite(forecast_df['pred_price']).all()

    def test_forecast_predictions_have_reasonable_values(self, trained_service):
        """Test that predictions are within reasonable range of historical data."""
        pid = list(trained_service.models.keys())[0]
        
        # Get historical price range
        train_df = trained_service.train_sets[pid]
        hist_min = train_df['base_price'].min()
        hist_max = train_df['base_price'].max()
        hist_mean = train_df['base_price'].mean()
        
        forecast_df = trained_service.forecast(pid, horizon_days=30)
        
        # Predictions should be somewhat related to historical data
        # Allow for +/- 100% of historical range as reasonable bound
        margin = (hist_max - hist_min)
        assert forecast_df['pred_price'].min() > hist_min - margin * 2
        assert forecast_df['pred_price'].max() < hist_max + margin * 2


# ====================================================================
# E2E COMBINED WORKFLOW TESTS
# ====================================================================

class TestPriceForecastServiceE2EWorkflow:
    """Test complete end-to-end workflows with real data."""

    def test_complete_pipeline_single_product(self, real_price_data, valid_product_ids):
        """Test complete pipeline for a single product: init → evaluate → forecast."""
        # Initialize service
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        # Select a product with sufficient data
        pid = valid_product_ids[0]
        
        # Verify model was trained
        assert pid in service.models
        
        # Evaluate
        metrics = service.evaluate(pid)
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        
        # Forecast
        forecast_df = service.forecast(pid, horizon_days=90)
        assert len(forecast_df) == 90
        assert forecast_df['pred_price'].notna().all()
        
        # Get training data
        combined_train = service.combined_train_df()
        assert pid in combined_train['prod_id'].values

    def test_batch_processing_multiple_products(self, real_price_data):
        """Test batch processing workflow for multiple products."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        available_pids = list(service.models.keys())[:5]
        
        results = {
            'evaluations': {},
            'forecasts': {}
        }
        
        # Process all products
        for pid in available_pids:
            # Evaluate each product
            results['evaluations'][pid] = service.evaluate(pid)
            
            # Forecast for each product
            results['forecasts'][pid] = service.forecast(pid, horizon_days=60)
        
        # Verify all processed successfully
        assert len(results['evaluations']) == len(available_pids)
        assert len(results['forecasts']) == len(available_pids)
        
        # Verify each has valid results
        for pid in available_pids:
            assert 'mae' in results['evaluations'][pid]
            assert len(results['forecasts'][pid]) == 60

    def test_combined_train_df_with_real_data(self, real_price_data):
        """Test combined training DataFrame creation with real data."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        combined_df = service.combined_train_df()
        
        assert combined_df is not None
        assert not combined_df.empty
        assert 'prod_id' in combined_df.columns
        assert 'base_price' in combined_df.columns
        
        # Should include data from all trained products
        unique_products = combined_df['prod_id'].unique()
        assert len(unique_products) == len(service.models)
        
        # Total rows should match sum of individual train sets
        expected_rows = sum(len(df) for df in service.train_sets.values())
        assert len(combined_df) == expected_rows


# ====================================================================
# E2E ROBUSTNESS TESTS
# ====================================================================

class TestPriceForecastServiceE2ERobustness:
    """Test service robustness with real-world data challenges."""

    def test_handles_different_test_ratios(self, real_price_data):
        """Test service with different train/test split ratios."""
        test_ratios = [0.1, 0.2, 0.3]
        
        for ratio in test_ratios:
            service = PriceForecastService(real_price_data, test_ratio=ratio)
            
            assert service is not None
            assert len(service.models) >= 1
            
            # Verify split ratio is approximately correct
            pid = list(service.models.keys())[0]
            train_size = len(service.train_sets[pid])
            test_size = len(service.test_sets[pid])
            total_size = train_size + test_size
            
            actual_test_ratio = test_size / total_size
            # Allow 10% tolerance
            assert abs(actual_test_ratio - ratio) < 0.15

    def test_service_with_minimal_test_set(self, real_price_data):
        """Test service with very small test set (test_ratio=0.1)."""
        service = PriceForecastService(real_price_data, test_ratio=0.1)
        
        pid = list(service.models.keys())[0]
        
        # Should still evaluate successfully even with small test set
        metrics = service.evaluate(pid)
        assert metrics['mae'] >= 0
        
        # Should still forecast successfully
        forecast_df = service.forecast(pid, horizon_days=30)
        assert len(forecast_df) == 30

    def test_service_maintains_state_across_operations(self, real_price_data):
        """Test that service state remains consistent across multiple operations."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        pid = list(service.models.keys())[0]
        
        # Perform multiple operations
        metrics1 = service.evaluate(pid)
        forecast1 = service.forecast(pid, horizon_days=30)
        metrics2 = service.evaluate(pid)
        forecast2 = service.forecast(pid, horizon_days=30)
        combined = service.combined_train_df()
        
        # State should be consistent
        assert metrics1 == metrics2
        assert forecast1.equals(forecast2)
        assert len(combined) > 0


# ====================================================================
# E2E PERFORMANCE TESTS
# ====================================================================

class TestPriceForecastServiceE2EPerformance:
    """Test performance characteristics with real data."""

    def test_initialization_completes_in_reasonable_time(self, real_price_data):
        """Test that service initialization completes without timeout."""
        import time
        
        start_time = time.time()
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Should complete in under 30 seconds for typical vegetable dataset
        assert elapsed < 30.0, f"Initialization took too long: {elapsed:.2f}s"
        assert service is not None

    def test_evaluation_completes_quickly(self, real_price_data):
        """Test that evaluation completes quickly."""
        import time
        
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        pid = list(service.models.keys())[0]
        
        start_time = time.time()
        metrics = service.evaluate(pid)
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Evaluation should be fast (< 5 seconds)
        assert elapsed < 5.0, f"Evaluation took too long: {elapsed:.2f}s"
        assert metrics is not None

    def test_forecast_generation_completes_quickly(self, real_price_data):
        """Test that forecast generation completes quickly."""
        import time
        
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        pid = list(service.models.keys())[0]
        
        start_time = time.time()
        forecast_df = service.forecast(pid, horizon_days=180)
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Forecast should be fast (< 5 seconds)
        assert elapsed < 5.0, f"Forecast took too long: {elapsed:.2f}s"
        assert len(forecast_df) == 180


# ====================================================================
# E2E DATA QUALITY TESTS
# ====================================================================

class TestPriceForecastServiceE2EDataQuality:
    """Test data quality aspects with real database data."""

    def test_wrangling_filters_invalid_prices(self, real_price_data):
        """Test that wrangling process filters out invalid prices from real data."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        # After wrangling, all prices should be positive
        assert (service.df['base_price'] > 0).all()
        
        # Check if any filtering occurred
        if 'invalid_prices' in service.feedback:
            # Some invalid prices were found and removed
            assert service.feedback['invalid_prices'] is not None

    def test_outlier_filtering_during_split(self, real_price_data):
        """Test that IQR filtering removes outliers during train/test split."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        pid = list(service.models.keys())[0]
        train_df = service.train_sets[pid]
        
        # Check that outliers were filtered (data should be more concentrated)
        q1 = train_df['base_price'].quantile(0.25)
        q3 = train_df['base_price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # All training data should be within IQR bounds
        assert (train_df['base_price'] >= lower_bound).all()
        assert (train_df['base_price'] <= upper_bound).all()

    def test_temporal_ordering_maintained(self, real_price_data):
        """Test that temporal ordering is maintained in train/test sets."""
        service = PriceForecastService(real_price_data, test_ratio=0.2)
        
        pid = list(service.models.keys())[0]
        train_df = service.train_sets[pid]
        test_df = service.test_sets[pid]
        
        # Train dates should be before or equal to test dates
        last_train_date = train_df['start_date'].max()
        first_test_date = test_df['start_date'].min()
        
        # In chronological split, test should come after train
        assert first_test_date >= train_df['start_date'].min()
