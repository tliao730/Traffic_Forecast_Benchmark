"""
ARIMA Model for Time Series Forecasting

Uses pmdarima's auto_arima for automatic parameter selection.
"""

import argparse
import warnings
import signal
from contextlib import contextmanager

import numpy as np
import pandas as pd
from gluonts.model.forecast import SampleForecast
from pmdarima import auto_arima
from common import eval, eval_time

warnings.filterwarnings('ignore')


# Timeout context manager
class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ARIMAPredictor:
    """Predictor wrapper for ARIMA that works with GluonTS evaluate_model."""
    
    def __init__(self, prediction_length, season_length, timeout=30):
        self.prediction_length = prediction_length
        self.season_length = season_length
        self.timeout = timeout
        self.count = 0
        self.success_count = 0
        self.timeout_count = 0
        self.error_count = 0
    
    def predict(self, dataset, num_samples=100):
        """Generate forecasts for a dataset."""
        for item in dataset:
            self.count += 1
            if self.count % 100 == 0:
                print(f"  Processed {self.count} (success: {self.success_count}, "
                      f"timeout: {self.timeout_count}, error: {self.error_count})")
            
            try:
                # Set a timeout for each time series
                with time_limit(self.timeout):
                    history = item['target']
                    start = item['start']
                    
                    # Handle multivariate data
                    if len(history.shape) > 1:
                        # Process each dimension separately
                        num_dims = history.shape[1]
                        all_dim_forecasts = []
                        
                        for dim in range(num_dims):
                            history_1d = history[:, dim]
                            
                            # Handle NaN values
                            if np.any(np.isnan(history_1d)):
                                history_1d = pd.Series(history_1d).ffill().bfill().values
                                if np.any(np.isnan(history_1d)):
                                    history_1d = np.nan_to_num(history_1d, nan=0.0)
                            
                            if len(history_1d) < 10:
                                # Use last value
                                forecast_1d = np.full(self.prediction_length, history_1d[-1])
                            else:
                                # Use auto_arima
                                model = auto_arima(
                                    history_1d,
                                    seasonal=False,  # Disable seasonal for speed
                                    max_p=2,
                                    max_q=2,
                                    max_d=1,
                                    start_p=1,
                                    start_q=1,
                                    trace=False,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True,
                                    n_jobs=1,
                                    maxiter=50,
                                )
                                forecast_1d = model.predict(n_periods=self.prediction_length)
                            
                            all_dim_forecasts.append(forecast_1d)
                        
                        forecast_mean = np.stack(all_dim_forecasts, axis=-1)
                    else:
                        # Univariate
                        # Handle NaN values
                        if np.any(np.isnan(history)):
                            history = pd.Series(history).ffill().bfill().values
                            if np.any(np.isnan(history)):
                                history = np.nan_to_num(history, nan=0.0)
                        
                        if len(history) < 10:
                            # Use last value
                            forecast_mean = np.full(self.prediction_length, history[-1])
                        else:
                            # Use auto_arima
                            model = auto_arima(
                                history,
                                seasonal=False,  # Disable seasonal for speed
                                max_p=2,
                                max_q=2,
                                max_d=1,
                                start_p=1,
                                start_q=1,
                                trace=False,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True,
                                n_jobs=1,
                                maxiter=50,
                            )
                            forecast_mean = model.predict(n_periods=self.prediction_length)
                    
                    # Create samples (repeat mean for all samples)
                    if len(forecast_mean.shape) > 1:
                        samples = np.tile(forecast_mean[np.newaxis, :, :], (num_samples, 1, 1))
                    else:
                        samples = np.tile(forecast_mean[np.newaxis, :], (num_samples, 1))
                    
                    # Create forecast object
                    forecast_start_date = start + len(item['target'])
                    
                    self.success_count += 1
                    
                    yield SampleForecast(
                        samples=samples,
                        start_date=forecast_start_date,
                        item_id=item.get('item_id', str(self.count)),
                    )
                    
            except TimeoutException:
                self.timeout_count += 1
                # Generate fallback forecast (repeat last value)
                try:
                    history = item['target']
                    start = item['start']
                    
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
                    self.error_count += 1
                    continue
                    
            except Exception as e:
                self.error_count += 1
                # Generate fallback forecast
                try:
                    history = item['target']
                    start = item['start']
                    
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


def main():
    """Main entry point for ARIMA evaluation."""
    model_name = "arima"
    print(f"Evaluating model: {model_name}")
    print("Processing all time series (no sample limit)")
    print("Using simplified ARIMA (seasonal=False) for speed")
    print("Timeout: 30 seconds per time series")
    
    def predictor_factory(dataset):
        """Factory function to create predictor for each dataset."""
        from gluonts.time_feature import get_seasonality
        season_length = get_seasonality(dataset.freq)
        
        return ARIMAPredictor(
            prediction_length=dataset.prediction_length,
            season_length=season_length,
            timeout=30
        )
    
    parser = argparse.ArgumentParser(description="ARIMA evaluation or time estimation")
    parser.add_argument("--eval-time", action="store_true", help="Run eval_time (time estimation) only")
    args = parser.parse_args()

    model_path = "pmdarima/auto_arima"
    if args.eval_time:
        eval_time(
            model_name=model_name,
            model_path=model_path,
            predictor_factory=predictor_factory,
        )
    else:
        eval(
            model_name=model_name,
            model_path=model_path,
            predictor_factory=predictor_factory,
            batch_size=1024
        )


if __name__ == "__main__":
    main()
