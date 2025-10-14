"""
Machine Learning Prediction Module for Stock Analysis
Implements multiple ML models for price prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime


class StockPredictor:
    """
    Multi-model stock price prediction system
    """
    
    def __init__(self, prediction_horizon=5):
        """
        Initialize predictor
        
        Args:
            prediction_horizon: Number of periods ahead to predict (default: 5 = 25 minutes)
        """
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_60',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI'
        ]
        
    def prepare_features(self, df):
        """
        Prepare features for ML models
        
        Args:
            df: DataFrame with OHLCV data and technical indicators
            
        Returns:
            X, y, feature_names, prepared_df
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Add lag features (previous prices)
        df['Close_lag_1'] = df['Close'].shift(1)
        df['Close_lag_2'] = df['Close'].shift(2)
        df['Close_lag_3'] = df['Close'].shift(3)
        
        # Add rolling statistics
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_MA'] = df['Price_Range'].rolling(window=5).mean()
        
        # Add momentum indicators
        df['Momentum_5'] = df['Close'].diff(5)
        df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        
        # Target: Future price change percentage
        df['future_price'] = df['Close'].shift(-self.prediction_horizon)
        df['price_change_pct'] = (df['future_price'] - df['Close']) / df['Close'] * 100
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Extended feature list
        extended_features = self.feature_columns + [
            'Close_lag_1', 'Close_lag_2', 'Close_lag_3',
            'Volume_MA_5', 'Price_Range', 'Price_Range_MA',
            'Momentum_5', 'ROC_5'
        ]
        
        # Filter features that exist in dataframe
        available_features = [f for f in extended_features if f in df.columns]
        
        X = df[available_features]
        y = df['price_change_pct']
        
        return X, y, available_features, df
    
    def train_models(self, X_train, y_train):
        """
        Train multiple ML models
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print("🤖 Training ML models...")
        
        # Model 1: Linear Regression
        print("   Training Linear Regression...")
        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_train, y_train)
        
        # Model 2: Random Forest
        print("   Training Random Forest...")
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=4
        )
        self.models['Random Forest'].fit(X_train, y_train)
        
        # Model 3: LightGBM
        print("   Training LightGBM...")
        self.models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        self.models['LightGBM'].fit(X_train, y_train)
        
        # Model 4: XGBoost
        print("   Training XGBoost...")
        self.models['XGBoost'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.models['XGBoost'].fit(X_train, y_train)
        
        print("✅ All models trained successfully")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            evaluation_results[model_name] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
        
        return evaluation_results
    
    def predict(self, X_latest):
        """
        Make predictions using all trained models
        
        Args:
            X_latest: Latest feature values (single row)
            
        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}
        confidence_scores = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(X_latest)[0]
            predictions[model_name] = pred
            
            # Calculate confidence based on model type
            if model_name == 'Random Forest':
                # Use standard deviation of tree predictions as uncertainty
                tree_predictions = [tree.predict(X_latest)[0] for tree in model.estimators_]
                std = np.std(tree_predictions)
                confidence_scores[model_name] = max(0, 100 - (std * 10))
            else:
                # Simple confidence based on prediction magnitude
                confidence_scores[model_name] = max(0, min(100, 100 - abs(pred) * 2))
        
        return predictions, confidence_scores
    
    def get_ensemble_prediction(self, predictions, confidence_scores):
        """
        Create ensemble prediction weighted by confidence
        
        Args:
            predictions: Dictionary of model predictions
            confidence_scores: Dictionary of confidence scores
            
        Returns:
            Weighted ensemble prediction
        """
        total_confidence = sum(confidence_scores.values())
        
        if total_confidence == 0:
            return np.mean(list(predictions.values()))
        
        weighted_sum = sum(
            pred * confidence_scores[model] 
            for model, pred in predictions.items()
        )
        
        return weighted_sum / total_confidence


def train_and_predict_stock(df, prediction_horizon=5):
    """
    Main function to train models and make predictions
    
    Args:
        df: DataFrame with OHLCV data and indicators
        prediction_horizon: Number of periods to predict ahead
        
    Returns:
        Dictionary with predictions and metadata
    """
    try:
        predictor = StockPredictor(prediction_horizon=prediction_horizon)
        
        # Prepare features
        X, y, feature_names, prepared_df = predictor.prepare_features(df)
        
        # Check if we have enough data
        if len(X) < 50:
            return {
                'error': 'Insufficient data for reliable predictions',
                'min_required': 50,
                'available': len(X)
            }
        
        # Split data (time series - no shuffle)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Train models
        predictor.train_models(X_train, y_train)
        
        # Evaluate models
        evaluation = predictor.evaluate_models(X_test, y_test)
        
        # Make predictions on latest data
        X_latest = X.tail(1)
        predictions, confidence_scores = predictor.predict(X_latest)
        
        # Get ensemble prediction
        ensemble_pred = predictor.get_ensemble_prediction(predictions, confidence_scores)
        
        # Determine prediction direction
        direction = "Bullish" if ensemble_pred > 0 else "Bearish" if ensemble_pred < 0 else "Neutral"
        
        return {
            'success': True,
            'predictions': {
                model: f"{pred:+.2f}%" for model, pred in predictions.items()
            },
            'ensemble_prediction': f"{ensemble_pred:+.2f}%",
            'confidence_scores': {
                model: f"{score:.1f}%" for model, score in confidence_scores.items()
            },
            'prediction_direction': direction,
            'prediction_period': f'next {prediction_horizon * 5} minutes',
            'evaluation_metrics': {
                model: {
                    'RMSE': f"{metrics['RMSE']:.3f}",
                    'MAE': f"{metrics['MAE']:.3f}",
                    'R2': f"{metrics['R2']:.3f}"
                }
                for model, metrics in evaluation.items()
            },
            'feature_count': len(feature_names),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
    except Exception as e:
        return {
            'error': f'Prediction error: {str(e)}',
            'success': False
        }


# Feature importance analysis
def get_feature_importance(predictor, feature_names):
    """
    Get feature importance from tree-based models
    
    Args:
        predictor: Trained StockPredictor instance
        feature_names: List of feature names
        
    Returns:
        Dictionary with feature importance for each model
    """
    importance_dict = {}
    
    for model_name in ['Random Forest', 'LightGBM', 'XGBoost']:
        if model_name in predictor.models:
            model = predictor.models[model_name]
            importances = model.feature_importances_
            
            importance_dict[model_name] = dict(
                sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10 features
            )
    
    return importance_dict