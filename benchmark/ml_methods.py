"""
Traditional Machine Learning Methods for Time Series Forecasting

Supported algorithms:
- Random Forest
- XGBoost
- LightGBM
- Linear Regression
- Ridge Regression
- Seasonal Naive (baseline)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import lightgbm as lgb
import xgboost as xgb
from gluonts.model.forecast import SampleForecast
from common import eval, eval_time
from common import get_prediction_length
warnings.filterwarnings('ignore')


def create_lag_features(data, n_lags=10):
    """Create lag features for supervised learning."""
    features = []
    for i in range(n_lags, len(data)):
        features.append(data[i-n_lags:i])
    return np.array(features)


def create_model(model_name, n_jobs=-1, **kwargs):
    """Create a model instance based on name.
    
    Args:
        model_name: Name of the model
        n_jobs: Number of CPU threads (-1 for all cores, 1 for single thread)
        **kwargs: Additional model parameters
    """
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=10,     # Drastically reduced for speed
            max_depth=5,         # Reduced
            min_samples_split=20,  # Increased
            min_samples_leaf=10,   # Increased
            max_features='sqrt',   # Limit features
            n_jobs=n_jobs,
            random_state=42,
            **kwargs
        )
    elif model_name == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=10,     # Drastically reduced for speed
            max_depth=3,         # Reduced
            learning_rate=0.3,   # Increased for faster convergence
            subsample=0.7,       # Reduced
            colsample_bytree=0.7,  # Reduced
            tree_method='hist',  # Faster algorithm
            n_jobs=n_jobs,
            random_state=42,
            **kwargs
        )
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=10,     # Drastically reduced for speed
            max_depth=3,         # Reduced
            learning_rate=0.3,   # Increased for faster convergence
            num_leaves=7,        # Reduced (2^3 - 1)
            subsample=0.7,       # Reduced
            colsample_bytree=0.7,  # Reduced
            n_jobs=n_jobs,
            random_state=42,
            verbose=-1,
            force_row_wise=True,  # Faster for small datasets
            **kwargs
        )
    elif model_name == "linear_regression":
        return LinearRegression(**kwargs)
    elif model_name == "ridge":
        return Ridge(alpha=1.0, random_state=42, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def seasonal_naive_forecast(history, prediction_length, season_length):
    """Simple seasonal naive forecast - repeat last season."""
    if season_length is None or season_length < 1:
        season_length = 1
    
    forecast = []
    for i in range(prediction_length):
        idx = len(history) - season_length + (i % season_length)
        if idx >= 0 and idx < len(history):
            forecast.append(history[idx])
        else:
            forecast.append(history[-1])
    
    return np.array(forecast)


def ml_forecast(history, prediction_length, model_name, season_length=None, n_lags=None, n_jobs=-1):
    """
    Forecast using traditional ML methods.
    
    Args:
        history: Historical time series data
        prediction_length: Number of steps to forecast
        model_name: Name of the model to use
        season_length: Seasonal period (for feature engineering)
        n_lags: Number of lags to use as features
    """
    # Handle NaN values by forward filling then backward filling
    if np.any(np.isnan(history)):
        history = pd.Series(history).ffill().bfill().values
        # If still has NaN (all NaN case), fill with 0
        if np.any(np.isnan(history)):
            history = np.nan_to_num(history, nan=0.0)
    
    if model_name == "seasonal_naive":
        return seasonal_naive_forecast(history, prediction_length, season_length)
    
    # Determine number of lags
    if n_lags is None:
        n_lags = min(max(prediction_length * 2, 10), len(history) // 4)
        n_lags = max(n_lags, prediction_length + 1)
    
    # Check if we have enough data
    if len(history) < n_lags + prediction_length:
        # Fall back to seasonal naive
        return seasonal_naive_forecast(history, prediction_length, season_length)
    
    # Create training data
    X_train = create_lag_features(history[:-prediction_length], n_lags)
    y_train = history[n_lags:-prediction_length]
    
    if len(X_train) == 0 or len(y_train) == 0:
        return seasonal_naive_forecast(history, prediction_length, season_length)
    
    # Train model
    model = create_model(model_name, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    
    # Iterative forecasting
    forecast = []
    current_window = history[-n_lags:].copy()
    
    for _ in range(prediction_length):
        # Predict next step
        X_pred = current_window.reshape(1, -1)
        y_pred = model.predict(X_pred)[0]
        forecast.append(y_pred)
        
        # Update window
        current_window = np.append(current_window[1:], y_pred)
    
    return np.array(forecast)


class MLPredictor:
    """Predictor wrapper for ML methods that works with GluonTS evaluate_model."""
    
    def __init__(self, model_name, prediction_length, season_length, n_jobs=-1):
        self.model_name = model_name
        self.prediction_length = prediction_length
        self.season_length = season_length
        self.n_jobs = n_jobs
        self.count = 0
    
    def predict(self, dataset, num_samples=100):
        """Generate forecasts for a dataset."""
        for item in dataset:
            self.count += 1
            if self.count % 1000 == 0:
                print(f"  Processed {self.count} time series")
            
            try:
                # Handle both dict and tuple formats
                if isinstance(item, tuple):
                    # Item is (input_dict, label) tuple
                    item_dict = item[0]
                else:
                    # Item is just a dict
                    item_dict = item
                
                history = item_dict['target']
                start = item_dict['start']
                
                # Handle multivariate data
                if len(history.shape) > 1:
                    # Process each dimension separately
                    num_dims = history.shape[1]
                    all_dim_forecasts = []
                    
                    for dim in range(num_dims):
                        forecast_1d = ml_forecast(
                            history[:, dim],
                            self.prediction_length,
                            self.model_name,
                            self.season_length,
                            n_jobs=self.n_jobs
                        )
                        all_dim_forecasts.append(forecast_1d)
                    
                    forecast_mean = np.stack(all_dim_forecasts, axis=-1)
                else:
                    # Univariate
                    forecast_mean = ml_forecast(
                        history,
                        self.prediction_length,
                        self.model_name,
                        self.season_length,
                        n_jobs=self.n_jobs
                    )
                
                # Create samples (repeat mean for all samples)
                if len(forecast_mean.shape) > 1:
                    samples = np.tile(forecast_mean[np.newaxis, :, :], (num_samples, 1, 1))
                else:
                    samples = np.tile(forecast_mean[np.newaxis, :], (num_samples, 1))
                
                # Create forecast object
                forecast_start_date = start + len(history)
                yield SampleForecast(
                    samples=samples,
                    start_date=forecast_start_date,
                    item_id=item_dict.get('item_id', str(self.count)),
                )
                
            except Exception as e:
                print(f"  Warning: Error on series {self.count}: {e}")
                # Generate a fallback forecast (repeat last value)
                try:
                    if len(history.shape) > 1:
                        fallback = np.tile(history[-1][np.newaxis, :], (self.prediction_length, 1))
                        samples = np.tile(fallback[np.newaxis, :, :], (num_samples, 1, 1))
                    else:
                        fallback = np.full(self.prediction_length, history[-1])
                        samples = np.tile(fallback[np.newaxis, :], (num_samples, 1))
                    
                    forecast_start_date = start + len(history)
                    yield SampleForecast(
                        samples=samples,
                        start_date=forecast_start_date,
                        item_id=item_dict.get('item_id', str(self.count)),
                    )
                except:
                    continue


def main():
    """Main entry point for ML methods evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ML methods for time series forecasting')
    parser.add_argument('--model', type=str, default='random_forest',
                      choices=['random_forest', 'xgboost', 'lightgbm', 
                               'linear_regression', 'ridge', 'seasonal_naive'],
                      help='Model to use')
    parser.add_argument('--n-jobs', type=int, default=16,
                      help='Number of CPU threads to use (-1 for all cores, 1 for single thread)')
    
    args = parser.parse_args()
    
    model_name = args.model
    n_jobs = args.n_jobs
    print(f"Evaluating model: {model_name}")
    print(f"Using {n_jobs if n_jobs > 0 else 'all available'} CPU threads")
    print("Processing all time series (no sample limit)")
    
    def predictor_factory(dataset):
        """Factory function to create predictor for each dataset."""
        from gluonts.time_feature import get_seasonality
        season_length = get_seasonality(dataset.freq)
        
        return MLPredictor(
            model_name=model_name,
            prediction_length=dataset.prediction_length,
            season_length=season_length,
            n_jobs=n_jobs
        )
    
    # Use common.eval() which handles everything
    # eval(
    #     model_name=model_name,
    #     model_path=f"sklearn/{model_name}",  # Descriptive path
    #     predictor_factory=predictor_factory,
    #     batch_size=1024
    # )

    eval_time(
        model_name=model_name,
        model_path=f"sklearn/{model_name}",  # Descriptive path
        predictor_factory=predictor_factory,
        estimation_samples=10
    )



if __name__ == "__main__":
    main()
