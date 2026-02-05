# tests/test_oracle_engine.py
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.engine import Engine
from fidzulu.db import oracle_engine
from fidzulu.config import DBConfig


class TestOracleEngine:
    """Test suite for the oracle_engine() function."""

    @patch('fidzulu.db.create_engine')
    @patch('fidzulu.db.load_db_config')
    def test_oracle_engine_creates_engine_successfully(self, mock_load_config, mock_create_engine):
        """
        Positive test: Verify that oracle_engine() creates a SQLAlchemy engine
        with the correct configuration.
        """
        # Arrange
        mock_config = DBConfig(
            host="test-host",
            port=1521,
            service_name="test_service",
            user="test_user",
            password="test_password"
        )
        mock_load_config.return_value = mock_config
        
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine
        
        # Act
        result = oracle_engine()
        
        # Assert
        assert result == mock_engine
        mock_load_config.assert_called_once()
        
        # Verify the DSN and URL construction
        expected_dsn = "(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=test-host)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=test_service)))"
        expected_url = f"oracle+oracledb://test_user:test_password@/?dsn={expected_dsn}"
        mock_create_engine.assert_called_once_with(expected_url, pool_pre_ping=True)

    @patch('fidzulu.db.load_db_config')
    def test_oracle_engine_fails_when_config_missing(self, mock_load_config):
        """
        Negative test: Verify that oracle_engine() raises an exception
        when required configuration is missing.
        """
        # Arrange
        mock_load_config.side_effect = RuntimeError("Missing required environment variable: FIDZULU_DB_SERVICE")
        
        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            oracle_engine()
        
        assert "Missing required environment variable" in str(exc_info.value)
        mock_load_config.assert_called_once()
