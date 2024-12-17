"""Unit tests for sleep quality analysis functionality."""

import os

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

class TestSleepQualityAnalysis:
    """Test suite for sleep quality analysis methods and data processing."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'Person ID': range(1, 6),
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Age': [25, 30, 35, 40, 45],
            'Sleep Duration': [7.0, 6.5, 8.0, 7.5, 6.0],
            'Quality of Sleep': [7, 6, 8, 7, 5],
            'Physical Activity Level': [60, 45, 70, 55, 50],
            'Stress Level': [4, 6, 3, 5, 7],
            'Heart Rate': [70, 75, 68, 72, 80],
            'Daily Steps': [8000, 7000, 9000, 7500, 6500]
        })

    def test_data_loading(self):
        """Test if data can be loaded correctly"""
        data_path = os.path.join('sleep_health_analysis', 'data', 'Sleep_health_and_lifestyle_dataset.csv')
        assert os.path.exists(data_path), "Dataset file should exist"
        df = pd.read_csv(data_path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        required_columns = ['Person ID', 'Sleep Duration', 'Quality of Sleep']
        assert all(col in df.columns for col in required_columns)

    def test_data_preprocessing(self, sample_data):
        """Test data preprocessing steps"""
        # Check for missing values
        assert not sample_data.isnull().any().any(), "There should be no missing values"
        
        # Check data types
        assert sample_data['Sleep Duration'].dtype == float
        assert sample_data['Quality of Sleep'].dtype == int
        assert sample_data['Physical Activity Level'].dtype == int

    def test_performance_metrics(self):
        """Test performance metric calculations"""
        y_true = np.array([7, 6, 8, 7, 5])
        y_pred = np.array([7, 6, 7, 7, 6])
        
        # Test MSE calculation
        mse = mean_squared_error(y_true, y_pred)
        assert isinstance(mse, float)
        assert mse >= 0
        
        # Test R2 score calculation
        r2 = r2_score(y_true, y_pred)
        assert isinstance(r2, float)
        assert r2 <= 1.0
        
        # Test accuracy calculation
        accuracy = accuracy_score(y_true, y_pred)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_value_ranges(self, sample_data):
        """Test if values are within expected ranges"""
        assert all(0 <= x <= 24 for x in sample_data['Sleep Duration'])
        assert all(1 <= x <= 10 for x in sample_data['Quality of Sleep'])
        assert all(0 <= x <= 100 for x in sample_data['Physical Activity Level'])
        assert all(1 <= x <= 10 for x in sample_data['Stress Level']) 