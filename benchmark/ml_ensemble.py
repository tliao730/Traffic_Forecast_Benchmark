"""
Ensemble Methods for Time Series Forecasting

Combines multiple ML methods for better predictions.

Supported ensemble strategies:
- Simple Average (默认)
- Weighted Average
- Median
"""

import warnings
import numpy as np
import pandas as pd
from gluonts.model.forecast import SampleForecast
from common import eval

# Import the ML forecasting function from ml_methods
from ml_methods import ml_forecast

warnings.filterwarnings('ignore')


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple ML methods.
    
    Args:
        models: List of model names to ensemble
        weights: Optional weights for weighted average (must sum to 1)
        strategy: 'mean', 'weighted', or 'median'
        prediction_length: Number of steps to forecast
        season_length: Seasonal period
    """
    
    def __init__(self, models, prediction_length, season_length, 
                 weights=None, strategy='mean', n_jobs=-1):
        self.models = models
        self.prediction_length = prediction_length
        self.season_length = season_length
        self.weights = weights
        self.strategy = strategy
        self.n_jobs = n_jobs
        self.count = 0
        
        # Validate inputs
        if strategy == 'weighted':
            if weights is None:
                raise ValueError("weights must be provided for weighted strategy")
            if len(weights) != len(models):
                raise ValueError("weights length must match models length")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("weights must sum to 1.0")
        
        print(f"Ensemble configuration:")
        print(f"  Models: {', '.join(models)}")
        print(f"  Strategy: {strategy}")
        if weights:
            print(f"  Weights: {dict(zip(models, weights))}")
    
    def predict(self, dataset, num_samples=100):
        """Generate ensemble forecasts for a dataset."""
        for item in dataset:
            self.count += 1
            if self.count % 1000 == 0:
                print(f"  Processed {self.count} time series")
            
            try:
                history = item['target']
                start = item['start']
                
                # Handle multivariate data
                if len(history.shape) > 1:
                    # Process each dimension separately
                    num_dims = history.shape[1]
                    all_dim_forecasts = []
                    
                    for dim in range(num_dims):
                        forecast_1d = self._ensemble_forecast(
                            history[:, dim],
                            self.prediction_length,
                            self.season_length
                        )
                        all_dim_forecasts.append(forecast_1d)
                    
                    forecast_mean = np.stack(all_dim_forecasts, axis=-1)
                else:
                    # Univariate
                    forecast_mean = self._ensemble_forecast(
                        history,
                        self.prediction_length,
                        self.season_length
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
                    item_id=item.get('item_id', str(self.count)),
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
                        item_id=item.get('item_id', str(self.count)),
                    )
                except:
                    continue
    
    def _ensemble_forecast(self, history, prediction_length, season_length):
        """
        Generate ensemble forecast for a single univariate series.
        
        Returns:
            Ensemble forecast combining multiple models
        """
        # Generate forecasts from all models
        forecasts = []
        for model_name in self.models:
            try:
                forecast = ml_forecast(
                    history,
                    prediction_length,
                    model_name,
                    season_length,
                    n_jobs=self.n_jobs
                )
                forecasts.append(forecast)
            except Exception as e:
                print(f"    Warning: {model_name} failed, skipping: {e}")
                continue
        
        if len(forecasts) == 0:
            # Fallback: return last value
            return np.full(prediction_length, history[-1])
        
        # Stack forecasts: shape (num_models, prediction_length)
        forecasts = np.array(forecasts)
        
        # Combine forecasts based on strategy
        if self.strategy == 'mean':
            # Simple average
            ensemble_forecast = np.mean(forecasts, axis=0)
        elif self.strategy == 'weighted':
            # Weighted average
            # Only use weights for successful models
            active_weights = self.weights[:len(forecasts)]
            active_weights = np.array(active_weights) / sum(active_weights)  # Renormalize
            ensemble_forecast = np.average(forecasts, axis=0, weights=active_weights)
        elif self.strategy == 'median':
            # Median (more robust to outliers)
            ensemble_forecast = np.median(forecasts, axis=0)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return ensemble_forecast


def main():
    """Main entry point for ensemble evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ensemble methods for time series forecasting')
    parser.add_argument('--strategy', type=str, default='mean',
                      choices=['mean', 'weighted', 'median'],
                      help='Ensemble strategy')
    parser.add_argument('--models', type=str, nargs='+',
                      default=['random_forest', 'lightgbm', 'xgboost', 'ridge'],
                      help='Models to ensemble')
    parser.add_argument('--weights', type=float, nargs='+',
                      default=None,
                      help='Weights for weighted average (must sum to 1)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                      help='Number of CPU threads to use (-1 for all cores, 1 for single thread)')
    
    args = parser.parse_args()
    
    # Validate weights if provided
    if args.weights:
        if len(args.weights) != len(args.models):
            raise ValueError("Number of weights must match number of models")
        if not np.isclose(sum(args.weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
    
    # Create model name
    if args.strategy == 'weighted' and args.weights:
        model_name = f"ensemble_{args.strategy}_{'_'.join(args.models)}"
        weight_str = '_'.join([f"{w:.2f}" for w in args.weights])
        model_name = f"{model_name}_w{weight_str}"
    else:
        model_name = f"ensemble_{args.strategy}_{'_'.join(args.models)}"
    
    print(f"Evaluating ensemble model: {model_name}")
    print(f"Using {args.n_jobs if args.n_jobs > 0 else 'all available'} CPU threads")
    print("Processing all time series (no sample limit)")
    
    def predictor_factory(dataset):
        """Factory function to create predictor for each dataset."""
        from gluonts.time_feature import get_seasonality
        season_length = get_seasonality(dataset.freq)
        
        return EnsemblePredictor(
            models=args.models,
            prediction_length=dataset.prediction_length,
            season_length=season_length,
            weights=args.weights,
            strategy=args.strategy,
            n_jobs=args.n_jobs
        )
    
    # Use common.eval() which handles everything
    eval(
        model_name=model_name,
        model_path=f"ensemble/{args.strategy}",
        predictor_factory=predictor_factory,
        batch_size=1024
    )


if __name__ == "__main__":
    main()
