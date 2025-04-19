import logging
import operator
from functools import reduce
from typing import Dict, Optional
import joblib
import os
from datetime import datetime
import requests

import numpy as np
import pandas as pd
from sympy import use
import talib.abstract as ta
from technical.qtpylib import crossed

from pandas import DataFrame
from technical import qtpylib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.fftpack import fft
from scipy.stats import zscore

from freqtrade import data
from freqtrade.exchange.exchange_utils import *
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.optimize.analysis import lookahead
from freqtrade.strategy import IStrategy, IntParameter, RealParameter, CategoricalParameter
from freqtrade.persistence import Trade
from freqtrade.enums import RunMode
from freqtrade.vendor.qtpylib.indicators import crossed_above, crossed_below

logger = logging.getLogger(__name__)


class LSTMStrategy_v34(IStrategy):
    """
    This is an example strategy that uses the LSTMRegressor model to predict the target score.
    Use at your own risk.
    This is a simple example strategy and should be used for educational purposes only.
    """

    plot_config = {
        "main_plot": {
        },
        "subplots": {
            "predictions": {
                "True Label": {"color": "blue", "plot_type": "line"},  # Rename T to "True Label"
                "Prediction": {"color": "red", "plot_type": "line"},  # Rename "&-s_target" to "Prediction"
                "Avg Prediction": {"color": "green", "plot_type": "line"},  # Rename "&-s_target_mean" to "Avg Prediction"
                "long_threshold": {"color": "green", "plot_type": "line"},
                "short_threshold": {"color": "red", "plot_type": "line"},
            },
            "Confidence": {
                "prediction_confidence": {"color": "orange", "plot_type": "scatter"},  # Plot prediction confidence
                "confidence_threshold" : {"color": "brown", "plot_type": "scatter"},
            },
            "Indicators": {
                "atr_scaled": {"color": "green", "plot_type": "line"},
            },
            "Thresholds": {
                "rolling_trend_scaled": {"color": "blue", "plot_type": "line"},
                "rolling_trend_threshold": {"color": "green", "plot_type": "line"},
            },
        },
    }

    # ROI table:
    minimal_roi = {
        "0": 1  # we let the model decide when to exit
    }

    # Stoploss:
    stoploss = -1  # Were letting the model decide when to sell

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.0139
    trailing_only_offset_is_reached = True

    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True
    use_custom_stoploss = True

    startup_candle_count = 300
                                                
    prediction_metrics_storage = []  # Class-level storage for all pairs

    # hyperopt categorical parameters switch
    hyperopt_categorical = True

    # use leverage
    use_leverage = False

    # ✅ Entry hyperopt parameters
    dynamic_long_threshold_multiplier = RealParameter(0.5, 1.2, default=1.0, space="buy", load=True, optimize=True)
    dynamic_short_threshold_multiplier = RealParameter(0.5, 1.2, default=1.0, space="buy", load=True, optimize=True)
    vol_rank_threshold = RealParameter(0.05, 0.5, default=0.25, space="buy", load=True, optimize=True)
    high_confidence_threshold = RealParameter(0.7, 0.95, default=0.85, space="buy", load=True, optimize=True)
    use_confidence_filter_entry = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    confidence_threshold_multiplier = RealParameter(0.1, 1.0, default=1.0, space="buy", load=True, optimize=True)
    use_trend_filter = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    rolling_trend_threshold_multiplier = RealParameter(0.2, 2.0, default=1.0, space="buy", load=True, optimize=True)
    vol_window = IntParameter(5, 100, default=24, space="buy", load=True, optimize=True)  # Window for volatility calculation
    trend_window = IntParameter(5, 100, default=48, space="buy", load=True, optimize=True)  # Window for trend calculation

    # Exit trend hyperopt parameters
    use_target_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)    
    dynamic_long_exit_threshold_multiplier = RealParameter(0.5, 1.5, default=1.0, space="sell", load=True, optimize=True)
    dynamic_short_exit_threshold_multiplier = RealParameter(0.5, 1.5, default=1.0, space="sell", load=True, optimize=True)
    use_trend_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    trend_exit_threshold_multiplier = RealParameter(0.2, 2.0, default=1.0, space="sell", load=True, optimize=True)
    use_timed_exit = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    timed_exit_long_threshold = IntParameter(10, 100, default=20, space="sell", load=True, optimize=True)
    timed_exit_short_threshold = IntParameter(10, 100, default=20, space="sell", load=True, optimize=True)

    # ✅ Stoploss Hyperopt Parameters
    soft_stoploss_pct = RealParameter(-0.25, -0.03, default=-0.05, space="sell", load=True, optimize=True)
    min_profit_for_trailing = RealParameter(0.001, 0.05, default=0.005, space="sell", load=True, optimize=True)
    atr_stoploss_multiplier = RealParameter(1.0, 10.0, default=3.0, space="sell", load=True, optimize=True)
    historical_volatility_factor = RealParameter(0.3, 1.2, default=0.5, space="sell", load=True, optimize=True) # Used in ATR-based calculation scaling
    prediction_confidence_factor = RealParameter(0.2, 1.0, default=0.6, space="sell", load=True, optimize=True) # Used in ATR-based calculation scaling
    max_loss_floor = RealParameter(0.01, 0.05, default=0.03, space="sell", load=True, optimize=True) # Min % for max loss cap
    max_loss_ceiling = RealParameter(0.05, 0.25, default=0.08, space="sell", load=True, optimize=True) # Max % for max loss cap
    max_loss_vol_multiplier = RealParameter(0.5, 5.0, default=2.0, space="sell", load=True, optimize=True) # Volatility sensitivity for max loss cap
    initial_stop_duration_candles = IntParameter(1, 10, default=3, space="sell", load=True, optimize=True) # Duration in candles for initial stop

    # ✅ Stake amount hyperopt parameters
    stake_scaling_factor = RealParameter(0.1, 2.0, default=1.0, space="buy", load=True, optimize=True)

    # ✅ Leverage Hyperopt Parameters
    leverage_range_end = IntParameter(1, 10, default=3, space="buy", load=True, optimize=use_leverage)
    volatility_influence = RealParameter(0.0, 1.0, default=0.2, space="buy", load=True, optimize=use_leverage)
    confidence_influence = RealParameter(0.0, 1.0, default=0.2, space="buy", load=True, optimize=use_leverage)    

    if use_leverage:
        # Buy hyperspace params:
        buy_params = {
            "confidence_influence": 0.69972,
            "confidence_threshold_multiplier": 0.92167,
            "dynamic_long_threshold_multiplier": 0.94825,
            "dynamic_short_threshold_multiplier": 1.1908,
            "high_confidence_threshold": 0.77177,
            "leverage_range_end": 4,
            "rolling_trend_threshold_multiplier": 1.53269,
            "stake_scaling_factor": 1.28344,
            "trend_window": 63,
            "use_confidence_filter_entry": True,
            "use_trend_filter": True,
            "vol_rank_threshold": 0.25956,
            "vol_window": 65,
            "volatility_influence": 0.15197,
        }

        # Sell hyperspace params:
        sell_params = {
            "atr_stoploss_multiplier": 1.75911,
            "dynamic_long_exit_threshold_multiplier": 1.43217,
            "dynamic_short_exit_threshold_multiplier": 1.22422,
            "historical_volatility_factor": 0.80103,
            "initial_stop_duration_candles": 1,
            "max_loss_ceiling": 0.14319,
            "max_loss_floor": 0.02042,
            "max_loss_vol_multiplier": 1.91933,
            "min_profit_for_trailing": 0.00397,
            "prediction_confidence_factor": 0.94111,
            "soft_stoploss_pct": -0.22662,
            "timed_exit_long_threshold": 45,
            "timed_exit_short_threshold": 79,
            "trend_exit_threshold_multiplier": 1.13459,
            "use_target_exit_filter": True,
            "use_timed_exit": True,
            "use_trend_exit_filter": True,
        }
    else:
        # Buy hyperspace params:
        buy_params = {
            "confidence_threshold_multiplier": 0.96214,
            "dynamic_long_threshold_multiplier": 0.78657,
            "dynamic_short_threshold_multiplier": 0.74133,
            "high_confidence_threshold": 0.78351,
            "rolling_trend_threshold_multiplier": 0.62609,
            "stake_scaling_factor": 1.765,
            "trend_window": 90,
            "use_confidence_filter_entry": True,
            "use_trend_filter": False,
            "vol_rank_threshold": 0.15384,
            "vol_window": 76,
            "confidence_influence": 0.2,  # value loaded from strategy
            "leverage_range_end": 3,  # value loaded from strategy
            "volatility_influence": 0.2,  # value loaded from strategy
        }

        # Sell hyperspace params:
        sell_params = {
            "atr_stoploss_multiplier": 1.08672,
            "dynamic_long_exit_threshold_multiplier": 1.34974,
            "dynamic_short_exit_threshold_multiplier": 1.2927,
            "historical_volatility_factor": 0.84423,
            "initial_stop_duration_candles": 1,
            "max_loss_ceiling": 0.17438,
            "max_loss_floor": 0.03156,
            "max_loss_vol_multiplier": 0.65719,
            "min_profit_for_trailing": 0.0028,
            "prediction_confidence_factor": 0.96542,
            "soft_stoploss_pct": -0.24775,
            "timed_exit_long_threshold": 72,
            "timed_exit_short_threshold": 71,
            "trend_exit_threshold_multiplier": 1.58606,
            "use_target_exit_filter": True,
            "use_timed_exit": False,
            "use_trend_exit_filter": True,
        }

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.trades: Dict[str, datetime] = {}  # Initialize self.trades in the constructor
        # ✅ Load historical Fear & Greed Index data if available
        self.historical_fng_data = self.load_historical_fng_data()

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, metadata: Dict, **kwargs):
        """
        Expands all features for FreqAI while keeping feature count optimized.
        Added Bollinger Band Width Percentage.
        """

        # ✅ Key Technical Indicators (Retained)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=14)  # Momentum Strength
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=5)  # Trend Direction

        # ✅ Bollinger Bands (Ensuring Calculation Before Use)
        if "bb_upperband-period" not in dataframe or "bb_lowerband-period" not in dataframe:
            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(dataframe), window=period, stds=2.2
            )
            dataframe["bb_lowerband-period"] = bollinger["lower"]
            dataframe["bb_middleband-period"] = bollinger["mid"]
            dataframe["bb_upperband-period"] = bollinger["upper"]

        # ❌ REMOVED old bb_width calculation
        # dataframe["%-bb_width-period"] = (
        #     dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        # ) / dataframe["bb_middleband-period"]

        # ✅ ADDED Bollinger Band Width Percentage
        dataframe["%-bb_width_pct-period"] = (
             (dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]) / dataframe["bb_middleband-period"]
        )

        dataframe['%-volume_roc-period'] = ta.ROC(dataframe['volume'], timeperiod=5)
        dataframe['%-volume_rsi-period'] = ta.RSI(dataframe['volume'], timeperiod=14)
        dataframe['%-obv-period'] = ta.OBV(dataframe) # On Balance Volume

        # ✅ Temporarily Remove Lower-Impact Indicators (Can Reintroduce if Needed)
        # drop_columns = [
        #     "%-cci-period", "%-momentum-period", "%-macd-period",
        #     "%-macdsignal-period", "%-macdhist-period"
        # ]
        # dataframe.drop(columns=[col for col in drop_columns if col in dataframe.columns], inplace=True, errors="ignore")

        # ✅ Fix NaNs
        dataframe.fillna(0, inplace=True)

        # ✅ Apply Z-Score Normalization to **volatile features only**
        # Added %-bb_width_pct-period to zscore
        zscore_columns = ["%-bb_width_pct-period", "%-rsi-period", "%-roc-period"]
        for col in zscore_columns:
            if col in dataframe.columns: # Check if column exists before applying zscore
                 dataframe.loc[:, f"{col}-zscore"] = pd.Series(zscore(dataframe[col]), index=dataframe.index).fillna(0)

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        # ✅ **Optimized Lag-Based Features**
        # lag_amount = 6
        # lag_features = ["close", "volume"]

        # lagged_data = {f"{feature}_lag{lag}": dataframe[feature].shift(lag) for feature in lag_features for lag in range(1, lag_amount + 1)}
        # dataframe = pd.concat([dataframe, pd.DataFrame(lagged_data, index=dataframe.index)], axis=1)
        # dataframe.loc[:, dataframe.columns.str.contains("_lag")] = dataframe.loc[:, dataframe.columns.str.contains("_lag")].bfill()

        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, metadata: Dict, **kwargs):
        """
        Defines features that should remain in their original timeframe.
        Added ATR Percentage.
        """

        # Ensure ATR exist first
        if "atr" not in dataframe.columns:
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=14).bfill()

        # ✅ ADDED ATR Percentage
        dataframe["%-atr_pct"] = (dataframe["atr"] / dataframe["close"]) * 100

        # ✅ Keep existing time-based features
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe.loc[:, "%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe.loc[:, "%-hour_of_day"] = dataframe["date"].dt.hour

        # ✅ Rolling Features (Fixed NaNs)
        dataframe.loc[:, "%-rolling_volatility"] = dataframe["close"].rolling(window=24).std().bfill()
        dataframe.loc[:, "%-rolling_mean"] = dataframe["close"].rolling(window=24).mean().bfill()

        # ✅ Replaced Rolling Mean with EMA
        dataframe.loc[:, "%-ema_trend"] = ta.EMA(dataframe, timeperiod=24).bfill()

        # ✅ CUSUM (Trend Break Detector - Should NOT be expanded)
        def get_cusum(series):
            series_mean = series.mean()
            return (series - series_mean).cumsum()

        dataframe.loc[:, "%-cusum_close"] = get_cusum(dataframe["close"]).fillna(0)

        # ✅ Optimized Hurst Exponent (Trend Strength - Smoothed)
        def hurst_exponent(ts, max_lag=20):
            if len(ts) < max_lag:
                return np.nan
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0]

        dataframe.loc[:, "%-hurst"] = dataframe["close"].rolling(window=72).apply(hurst_exponent, raw=True)
        dataframe.loc[:, "%-hurst_smooth"] = dataframe["%-hurst"].rolling(window=10).mean().bfill()

        # ✅ Fourier Transform (Fixed NaNs & Normalized)
        def compute_fourier(series, n_components=3):
            if len(series) < 72:
                return np.nan
            fft_vals = fft(series)
            return np.abs(fft_vals[:n_components]).sum()

        dataframe.loc[:, "%-fourier_price"] = dataframe["close"].rolling(window=72).apply(compute_fourier, raw=True)
        dataframe.loc[:, "%-fourier_price"] = dataframe["%-fourier_price"].fillna(dataframe["%-fourier_price"].median())

        # ✅ Normalize Fourier Features using ATR
        dataframe.loc[:, "%-fourier_price_norm"] = dataframe["%-fourier_price"] / (dataframe["atr"] + 1e-6)

        # ✅ Apply Z-Score Normalization to **volatile features only**
        # Added %-atr_pct to zscore
        zscore_columns = ["%-rolling_volatility", "%-rolling_mean", "%-fourier_price_norm", "%-atr_pct"]
        for col in zscore_columns:
             if col in dataframe.columns: # Check if column exists before applying zscore
                 dataframe.loc[:, f"{col}-zscore"] = pd.Series(zscore(dataframe[col]), index=dataframe.index).fillna(0)

        # ✅ Incorporate order flow features
        dataframe = self.get_order_flow_features(dataframe, metadata)

        # ✅ Fetch Fear & Greed Index
        current_date = dataframe['date'].iloc[-1] if 'date' in dataframe else None
        fear_greed_value, fear_greed_classification = self.get_fear_and_greed_index(current_date)
        dataframe['fear_greed_index'] = fear_greed_value
        dataframe['fear_greed_index'] = dataframe['fear_greed_index'].ffill()

        logger.info(f"🔍 Total features before model training: {len(dataframe.columns)}")

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        # ✅ Assign `&-s_target` for FreqAI
        dataframe['&-s_target'] = self.create_target_T(dataframe)

        return dataframe

    def create_target_T(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Creates a new target (T) based on normalized future price change using ATR.
        Removed tanh() transformation.
        Simplified normalization to use ATR only.
        """

        window = 24
        
        dataframe["ATR"] = ta.ATR(dataframe, timeperiod=window).bfill()  # ATR-based normalization
        dataframe["close"] = dataframe["close"].replace(0, np.nan).bfill()  # Prevent division by zero

        # ✅ Compute dynamic lookahead (ensuring valid values)
        dataframe["lookahead_dynamic"] = np.clip((dataframe["ATR"] / dataframe["close"]) * 100, 5, 24).fillna(10).astype(int)

        # ✅ Compute Future Price Change dynamically using a loop
        future_change = []
        for i in range(len(dataframe)):
            try:
                lookahead = int(dataframe["lookahead_dynamic"].iloc[i])
                future_index = i + lookahead
                if future_index >= len(dataframe):
                    future_change.append(np.nan) # Handle cases where future index is out of bounds
                else:
                    future_close = dataframe["close"].iloc[future_index]
                    future_change.append(future_close - dataframe["close"].iloc[i])
            except (KeyError, IndexError) as e:
                print(f"Error calculating future change: {e}")
                future_change.append(np.nan)  # Handle cases where row.name is not a valid index

        dataframe["future_change"] = future_change

        # ✅ Compute Trend Strength Using Future Price Change
        dataframe["TS"] = dataframe["future_change"].rolling(window).mean()

        # ✅ Normalize Trend Strength Using ATR ONLY
        dataframe["T"] = dataframe["TS"] / (dataframe["ATR"] + 1e-6) # Simplified Normalization

        # ❌ REMOVED: Apply `tanh()` to Limit Extreme Values
        dataframe["T"] = np.tanh(dataframe["T"])

        # ✅ Fill NaNs (no longer inplace)
        dataframe["T"] = dataframe["T"].fillna(0)

        return dataframe["T"]
    
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates indicators used for trade entry and exit signals.
        Optimized for the 1-hour timeframe.
        Minimizes arbitrary parameters, maximizing dynamic calculations.
        Adds safeguards against NaN/inf/None. Uses modern pandas fill methods.
        """

        self.freqai_info = self.config["freqai"]

        # --- Initial Data Cleaning ---
        # Use .bfill() instead of fillna(method='bfill')
        df['close'] = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill().bfill()
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').replace(0, 1e-9).ffill().fillna(1e-9) # Keep fillna(value)

        # --- Standard Indicators ---
        # ✅ 1. ATR (Volatility)
        volatility_window = self.vol_window.value
        pct_change_std = df["close"].pct_change().rolling(volatility_window).std().fillna(0)
        atr_timeperiod = int(10 + 20 * (pct_change_std.iloc[-1] / 0.05))
        atr_timeperiod = max(5, min(atr_timeperiod, 30))
        # Use .bfill() instead of fillna(method='bfill')
        df["atr"] = ta.ATR(df, timeperiod=atr_timeperiod).ffill().bfill().fillna(1e-9) # Keep final fillna(value)
        df["atr_normalized"] = df["atr"] / df["close"].clip(lower=1e-9)
        df["atr_normalized"] = df["atr_normalized"].ffill().fillna(0)

        # ✅ 2. Volume Rank (Volume)
        # Use .bfill() instead of fillna(method='bfill')
        atr_mean = df["atr"].rolling(24).mean().bfill().fillna(1e-9) # Keep final fillna(value)
        vol_rank_window = int(12 + 24 * (atr_mean.iloc[-1] / (df["close"].iloc[-1] * 0.05)))
        vol_rank_window = max(6, min(vol_rank_window, 48))
        df["vol_rank"] = df["volume"].rolling(vol_rank_window).rank(pct=True).fillna(0)

        # ✅ 3. Rolling Trend (Trend)
        atr_scaling_factor = atr_mean.iloc[-1] / (df["close"].iloc[-1] * 0.05)
        pct_change_period = int(12 + 24 * atr_scaling_factor)
        pct_change_period = max(6, min(pct_change_period, 48))
        rolling_window = int(3 + 6 * atr_scaling_factor)
        rolling_window = max(2, min(rolling_window, 12))
        df["rolling_trend"] = df["close"].pct_change(pct_change_period).rolling(rolling_window).mean().fillna(0)

        # ✅ 4. ATR Scaling (Volatility Scaling)
        atr_scaling_window = int(24 + 48 * (pct_change_std.iloc[-1] / 0.05))
        atr_scaling_window = max(12, min(atr_scaling_window, 72))
        atr_quantile = 0.90
        # Use .bfill() instead of fillna(method='bfill')
        quantile_val = df["atr_normalized"].rolling(atr_scaling_window).quantile(atr_quantile).ffill().bfill().fillna(1e-9) # Keep final fillna(value)
        df["atr_scaled"] = df["atr_normalized"] / quantile_val.clip(lower=1e-9)
        df["atr_scaled"] = df["atr_scaled"].ffill().fillna(0)

        # ✅ 5. MinMax Scaling ATR Scaled
        atr_scaled_min = df["atr_scaled"].min()
        atr_scaled_max = df["atr_scaled"].max()
        atr_scaled_range = atr_scaled_max - atr_scaled_min + 1e-9
        df["atr_scaled"] = (df["atr_scaled"] - atr_scaled_min) / atr_scaled_range
        df["atr_scaled"] = df["atr_scaled"].clip(0, 1).fillna(0)

        # ✅ 6. Improved Rolling Trend Scaling
        trend_window = self.trend_window.value
        # Use .bfill() instead of fillna(method='bfill')
        adaptive_window_atr_mean = df["atr"].rolling(50).mean().bfill().fillna(1e-9) # Keep final fillna(value)
        adaptive_window = int(min(trend_window, max(20, adaptive_window_atr_mean.iloc[-1] * 10)))
        mean_trend = df["rolling_trend"].rolling(adaptive_window).mean().ffill().fillna(0)
        std_trend = df["rolling_trend"].rolling(adaptive_window).std().ffill().fillna(1e-9)
        df["rolling_trend_scaled"] = (df["rolling_trend"] - mean_trend) / std_trend.clip(lower=1e-9)
        df["rolling_trend_scaled"] = df["rolling_trend_scaled"].fillna(0)

        # ✅ Call FreqAI for ML Predictions
        df = self.freqai.start(df, metadata, self)

        # --- Process FreqAI Outputs (Except Confidence-Related) ---
        freqai_cols_pre_metrics = ["&-s_target", "&-s_target_mean", "&-s_target_std", "do_predict"]
        for col in freqai_cols_pre_metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').ffill().fillna(0)
            else:
                logger.warning(f"FreqAI column '{col}' missing after freqai.start(). Assigning 0.")
                df[col] = 0.0

        # ✅ Store Threshold Base Values (Not Dependent on Confidence)
        df["dynamic_long_threshold_base"] = df["&-s_target_mean"] + df["&-s_target_std"] * df["atr_scaled"]
        df["dynamic_short_threshold_base"] = df["&-s_target_mean"] - df["&-s_target_std"] * df["atr_scaled"]
        df["rolling_trend_threshold_base"] = df["rolling_trend_scaled"].rolling(100, min_periods=10).median().ffill().fillna(0)
        df["dynamic_exit_threshold_base"] = (
            df["&-s_target"].ewm(span=50).mean().ffill().fillna(0) +
            df["atr_scaled"] * df["&-s_target_std"] * (0.6 + df["vol_rank"] * 0.3)
        )
        df["exit_trend_threshold_base"] = df["rolling_trend_scaled"].rolling(50).median().ffill().fillna(0)

        # ✅ Final Threshold Calculations (Not Dependent on Confidence) - Using direct .value access
        df["long_threshold"] = (df["dynamic_long_threshold_base"] * getattr(self.dynamic_long_threshold_multiplier, 'value', 1.0) * df["atr_scaled"]).fillna(0)
        df["short_threshold"] = (df["dynamic_short_threshold_base"] * getattr(self.dynamic_short_threshold_multiplier, 'value', 1.0) * df["atr_scaled"]).fillna(0)
        df["long_exit_threshold"] = (df['dynamic_exit_threshold_base'] * getattr(self.dynamic_long_exit_threshold_multiplier, 'value', 1.0) * df["atr_scaled"]).fillna(0)
        df["short_exit_threshold"] = (df['dynamic_exit_threshold_base'] * getattr(self.dynamic_short_exit_threshold_multiplier, 'value', 1.0) * df["atr_scaled"]).fillna(0)
        df["rolling_trend_threshold"] = (df["rolling_trend_threshold_base"] * getattr(self.rolling_trend_threshold_multiplier, 'value', 1.0)).fillna(0)
        df["rolling_trend_exit_threshold"] = (df["exit_trend_threshold_base"] * getattr(self.trend_exit_threshold_multiplier, 'value', 1.0)).fillna(0)

        # ✅ Keeping "T" for Plotting Purposes Only
        df["T"] = self.create_target_T(df)
        df["Prediction"] = df["&-s_target"]
        df["Avg Prediction"] = df["&-s_target_mean"]
        df["True Label"] = df["T"]

        # --- Compute Prediction Metrics (This calculates prediction_confidence) ---
        log_metrics = (
            not hasattr(self, "dp")
            or getattr(self.dp, "runmode", None) in [RunMode.BACKTEST, RunMode.HYPEROPT, RunMode.PLOT]
        )
        self.compute_prediction_metrics(df, metadata, log_metrics=log_metrics)

        # --- Process Confidence and Calculate Confidence Thresholds (AFTER compute_prediction_metrics) ---
        if "prediction_confidence" in df.columns:
            df["prediction_confidence"] = pd.to_numeric(df["prediction_confidence"], errors='coerce').ffill().fillna(0)
        else:
            logger.warning(f"FreqAI column 'prediction_confidence' missing after compute_prediction_metrics. Assigning 0.")
            df["prediction_confidence"] = 0.0

        # Calculate confidence thresholds using the now available prediction_confidence - Using direct .value access
        df["confidence_threshold_base"] = df["prediction_confidence"].rolling(100).quantile(0.5).ffill().fillna(0)
        df["confidence_threshold"] = (df["confidence_threshold_base"] * getattr(self.confidence_threshold_multiplier, 'value', 1.0)).fillna(0)

        # --- Calculate Trade Duration ---
        if not hasattr(self, 'trades'): self.trades = {}
        if not hasattr(self, 'timeframe_minutes'): self.timeframe_minutes = timeframe_to_minutes(self.timeframe)

        if metadata['pair'] not in self.trades:
            df['trade_duration'] = 0
        else:
            if 'open_since' in self.trades[metadata['pair']]:
                open_since = self.trades[metadata['pair']]['open_since']
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    df['trade_duration'] = (df.index - open_since).total_seconds() / 60 / self.timeframe_minutes
                else:
                    logger.warning(f"Dataframe index is not datetime type for {metadata['pair']}. Cannot calculate trade_duration.")
                    df['trade_duration'] = 0
            else:
                logger.warning(f"'open_since' not found in self.trades for {metadata['pair']}. Setting trade_duration to 0.")
                df['trade_duration'] = 0

        return df
    
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines the entry signal for both long and short trades using crossed_above/below.
        Leverages indicators calculated in populate_indicators().
        Handles potential NaN values before crossing checks.
        """

        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None  # Initialize enter_tag

        # --- Fill NaN values and Ensure Numeric Type before comparison ---
        # List of columns used in comparisons or conditions
        cols_to_process = [
            "&-s_target", "long_threshold", "short_threshold",
            "rolling_trend_scaled", "rolling_trend_threshold", "prediction_confidence",
            "confidence_threshold", "vol_rank", "atr_scaled",
            "do_predict" # Though likely int, process just in case
        ]

        for col in cols_to_process:
            if col in df.columns:
                try:
                    # 1. Forward fill, then fill remaining NaNs with 0
                    filled_series = df[col].ffill().fillna(0) # Uses ffill() and fillna(value)
                    # 2. Explicitly convert to float to handle potential None/object types
                    df[col] = filled_series.astype(float)
                except Exception as e:
                    logger.error(f"Error processing column '{col}' for {metadata['pair']}: {e}. Assigning default 0.")
                    df[col] = 0.0
            else:
                logger.warning(f"Column '{col}' not found in dataframe for {metadata['pair']}. Entry/Exit signals might be incorrect. Assigning default 0.")
                # Assign a default series of 0s if the column is missing entirely
                df[col] = 0.0


        # ✅ Combined Entry Condition (Direction-Aware) - Uses direct column names and hyperparams
        def combined_entry_condition(df, side: str):
            # Base condition check
            base_condition = (df["do_predict"] == 1) # Comparison should be safe after astype(float)
            vol_rank_thresh_val = getattr(self.vol_rank_threshold, 'value', None)
            if vol_rank_thresh_val is not None:
                base_condition &= (df["vol_rank"] > vol_rank_thresh_val)
            else:
                logger.warning(f"vol_rank_threshold.value is None for {metadata['pair']}. Skipping vol_rank check.")

            condition = base_condition

            # Ensure confidence_threshold value is available
            conf_thresh_mult_val = getattr(self.confidence_threshold_multiplier, 'value', None)
            if self.use_confidence_filter_entry.value and conf_thresh_mult_val is not None:
                condition &= (df["prediction_confidence"] > df["confidence_threshold"]) # Comparison safe after astype(float)
            elif self.use_confidence_filter_entry.value:
                    logger.warning(f"confidence_threshold_multiplier.value is None for {metadata['pair']}. Skipping confidence filter.")

            # Ensure trend_threshold value is available
            trend_thresh_mult_val = getattr(self.rolling_trend_threshold_multiplier, 'value', None)
            if self.use_trend_filter.value and trend_thresh_mult_val is not None:
                if side == "long":
                    condition &= (df["rolling_trend_scaled"] > df["rolling_trend_threshold"]) # Comparison safe after astype(float)
                elif side == "short":
                    condition &= (df["rolling_trend_scaled"] < df["rolling_trend_threshold"]) # Comparison safe after astype(float)
            elif self.use_trend_filter.value:
                    logger.warning(f"rolling_trend_threshold_multiplier.value is None for {metadata['pair']}. Skipping trend filter.")

            return condition

        # --- Perform Crossing Checks (Should now be safe) ---
        try:
            # ─── Long Entry Logic ───────────────────────────
            long_cross = crossed_above(df["&-s_target"], df["long_threshold"])
            df.loc[
                long_cross & combined_entry_condition(df, "long"),
                ["enter_long", "enter_tag"]
            ] = (1, "long")

            # ─── Short Entry Logic ──────────────────────────
            short_cross = crossed_below(df["&-s_target"], df["short_threshold"])
            df.loc[
                short_cross & combined_entry_condition(df, "short"),
                ["enter_short", "enter_tag"]
            ] = (1, "short")

            # ─── Fallback Entry ─────────────────────────────
            # Use getattr for safer access to hyperparameter value
            high_conf_val = getattr(self.high_confidence_threshold, 'value', None)
            if high_conf_val is not None:
                high_confidence = df["prediction_confidence"] > high_conf_val

                fallback_long_cross = crossed_above(df["rolling_trend_scaled"], df["rolling_trend_threshold"])
                df.loc[
                    (df["do_predict"] == 1) & (df["&-s_target"] > 0.01 * df["atr_scaled"]) & high_confidence & (df["enter_long"] == 0) &
                    fallback_long_cross,
                    ["enter_long", "enter_tag"]
                ] = (1, "long_fallback")

                fallback_short_cross = crossed_below(df["rolling_trend_scaled"], df["rolling_trend_threshold"])
                df.loc[
                    (df["do_predict"] == 1) & (df["&-s_target"] < -0.01 * df["atr_scaled"]) & high_confidence & (df["enter_short"] == 0) &
                    fallback_short_cross,
                    ["enter_short", "enter_tag"]
                ] = (1, "short_fallback")
            else:
                    logger.warning(f"high_confidence_threshold.value is None for {metadata['pair']}. Skipping fallback entries.")


        except TypeError as e:
            # Keep critical logging in case the error reappears under different conditions
            logger.critical(f"CRITICAL: TypeError occurred during crossed_above/below for {metadata['pair']}: {e}")
            # Log relevant column dtypes directly
            logger.critical(f"Data types:\nTarget: {df['&-s_target'].dtype}\nLong Thresh: {df['long_threshold'].dtype}\nShort Thresh: {df['short_threshold'].dtype}\nTrend: {df['rolling_trend_scaled'].dtype}\nTrend Thresh: {df['rolling_trend_threshold'].dtype}")

        return df
    
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines the exit signal for both long and short trades using crossed_above/below.
        Leverages indicators calculated in populate_indicators().
        """

        df['exit_long'] = False
        df['exit_short'] = False

        # ✅ Combined Exit Condition
        def combined_exit_condition(df):
            return (df['prediction_confidence'] > df["confidence_threshold"]) & (df['vol_rank'] > self.vol_rank_threshold.value)

        # 1. Target Reversal Exit (Primary)
        if self.use_target_exit_filter.value:
            df.loc[
                crossed_below(df['&-s_target'], df["long_exit_threshold"]) &
                combined_exit_condition(df),
                ['exit_long', 'exit_tag']
            ] = (True, 'exit_long')
            df.loc[
                crossed_above(df['&-s_target'], df["short_exit_threshold"]) &
                combined_exit_condition(df),
                ['exit_short', 'exit_tag']
            ] = (True, 'exit_short')

        # 2. Trend Reversal Exit (Secondary - if target exit not triggered)
        if self.use_trend_exit_filter.value:
            df.loc[
                crossed_below(df['rolling_trend_scaled'], df["rolling_trend_exit_threshold"]) &
                (df['exit_long'] == False) & combined_exit_condition(df),  # Only if target exit not triggered
                ['exit_long', 'exit_tag']
            ] = (True, 'exit_long_trend')
            df.loc[
                crossed_above(df['rolling_trend_scaled'], df["rolling_trend_exit_threshold"]) &
                (df['exit_short'] == False) & combined_exit_condition(df),  # Only if target exit not triggered
                ['exit_short', 'exit_tag']
            ] = (True, 'exit_short_trend')

        # 3. Timed Exit (Tertiary - if other exits not triggered)
        # Timed exit doesn't lend itself well to crossed_above/below, so we keep it as is.
        if self.use_timed_exit.value:
            df.loc[
                (df['trade_duration'] > self.timed_exit_long_threshold.value) &
                (df['exit_long'] == False) & (df['prediction_confidence'] < df['confidence_threshold']),  # Only if other exits not triggered and low confidence
                ['exit_long', 'exit_tag']
            ] = (True, 'exit_long_timed')
            df.loc[
                (df['trade_duration'] > self.timed_exit_short_threshold.value) &
                (df['exit_short'] == False) & (df['prediction_confidence'] < df['confidence_threshold']),  # Only if other exits not triggered and low confidence
                ['exit_short', 'exit_tag']
            ] = (True, 'exit_short_timed')

        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic returning negative percentages relative to current_rate
        to achieve trailing stop behavior, similar to v32's implicit functionality.
        Returns:
            float: Negative value.
                    -1: Keep current stoploss price.
                    Negative percentage (e.g., -0.05): Set stop 5% below current_rate (long) or 5% above current_rate (short).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            # logger.warning(f"Dataframe unavailable for {pair}. Keeping existing stoploss.")
            return -1 # Keep existing stoploss

        # Ensure trade open rate is valid (needed for main calculation's percentage conversion)
        if not trade or not trade.open_rate:
                logger.warning(f"Trade or open_rate unavailable for pair {pair}. Keeping existing stoploss.")
                return -1 # Keep existing stoploss

        # --- Get Data & Parameters ---
        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        if atr <= 0:
                logger.warning(f"ATR is zero or negative for pair {pair}. Keeping existing stoploss.")
                return -1 # Keep existing stoploss

        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        prediction_confidence = last_candle.get("prediction_confidence", 0.5)

        # Calculate trade duration in candles using timeframe_to_minutes
        tf_minutes = timeframe_to_minutes(self.timeframe)
        trade_duration_candles = (current_time - trade.open_date_utc).total_seconds() / (tf_minutes * 60)

        # Get hyperopt parameters (percentages are expected to be negative for stoploss)
        soft_stoploss_pct = self.soft_stoploss_pct.value # Negative value
        initial_duration = self.initial_stop_duration_candles.value
        min_profit_for_trailing = self.min_profit_for_trailing.value # Profit threshold
        atr_multiplier = self.atr_stoploss_multiplier.value
        historical_volatility_factor = self.historical_volatility_factor.value
        prediction_confidence_factor = self.prediction_confidence_factor.value
        max_loss_floor_val = self.max_loss_floor.value # Positive value (e.g., 0.03 for 3%)
        max_loss_ceiling_val = self.max_loss_ceiling.value # Positive value (e.g., 0.08 for 8%)
        max_loss_vol_multiplier_val = self.max_loss_vol_multiplier.value

        # --- Determine Stoploss Percentage ---

        # --- Initial Stoploss Phase ---
        # Return soft_stoploss_pct directly. Freqtrade interprets this negative percentage
        # relative to current_rate, creating a trailing stop.
        if trade_duration_candles < initial_duration:
            # logger.info(f"[Stoploss Initial TRAILING] Pair: {pair} | Duration: {trade_duration_candles:.1f} < {initial_duration} | Using Soft Stop Pct: {soft_stoploss_pct:.4f}")
            return soft_stoploss_pct

        # --- Soft Stoploss Phase (After Initial Duration, Before Profit Target) ---
        # Return soft_stoploss_pct directly. Freqtrade interprets this relative to current_rate.
        if current_profit < min_profit_for_trailing:
            # logger.info(f"[Stoploss Soft TRAILING] Pair: {pair} | Profit: {current_profit:.4f} < {min_profit_for_trailing:.4f} | Using Soft Stop Pct: {soft_stoploss_pct:.4f}")
            return soft_stoploss_pct

        # --- Main Stoploss Calculation (After Initial Duration & Profit Target Met) ---
        # Calculate the desired stoploss percentage based on ATR, volatility, confidence.
        dynamic_volatility_factor = 1 + historical_volatility * historical_volatility_factor
        confidence_factor = 1 - (prediction_confidence * prediction_confidence_factor)
        stoploss_buffer_abs = atr * atr_multiplier * dynamic_volatility_factor * confidence_factor

        # Convert absolute buffer to percentage relative to OPEN rate (for consistent scaling)
        # We still use open_rate here just to get a comparable percentage scale,
        # but the final returned percentage will be applied to current_rate by Freqtrade.
        stoploss_pct_calculated = stoploss_buffer_abs / trade.open_rate # Positive value

        # Calculate max loss percentage (dynamic based on volatility)
        max_loss_abs_pct = min(max_loss_floor_val + historical_volatility * max_loss_vol_multiplier_val, max_loss_ceiling_val) # Positive value

        # Determine the final percentage: the LARGER loss (closer to zero) between calculated and max_loss cap.
        # Negate the positive calculated values to get the required negative percentage.
        final_stoploss_pct = max(-stoploss_pct_calculated, -max_loss_abs_pct) # Result is negative

        # Ensure the calculated percentage is actually negative, fallback to soft stop if error
        if final_stoploss_pct >= 0:
            # logger.info(f"[Stoploss Error TRAILING] Pair: {pair} | Calculated positive stoploss {final_stoploss_pct:.4f}. Using soft stoploss {soft_stoploss_pct:.4f} instead.")
            stoploss_pct_to_use = soft_stoploss_pct
        else:
            stoploss_pct_to_use = final_stoploss_pct
            # logger.info(
            #     f"[Stoploss Main TRAILING] Pair: {pair} | DVF: {dynamic_volatility_factor:.4f} | CF: {confidence_factor:.4f} | "
            #     f"Buffer Abs: {stoploss_buffer_abs:.4f} | Calc SL %: {-stoploss_pct_calculated:.4f} | "
            #     f"Max Loss %: {-max_loss_abs_pct:.4f} | Final SL Pct: {stoploss_pct_to_use:.4f} | "
            #     f"Profit: {current_profit:.2f} | Duration Candles: {trade_duration_candles:.1f}"
            # )

        # Return the final negative percentage. Freqtrade will apply this relative to current_rate.
        return stoploss_pct_to_use

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float,
                            min_stake: float | None, max_stake: float, leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            logger.warning(f"Dataframe unavailable for {pair}. Using proposed stake: {proposed_stake}")
            return proposed_stake

        last_candle = dataframe.iloc[-1]
        prediction_confidence = last_candle.get("prediction_confidence", 0.5)
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01

        # Cap volatility for stability in calculation
        scaled_volatility = min(historical_volatility, 0.05)

        # --- Adjust proposed_stake based on confidence and volatility ---

        # Confidence Factor: Scales stake based on prediction confidence.
        # Centered around 1.0. Higher confidence -> higher factor. Uses confidence_influence hyperparam.
        conf_influence = self.confidence_influence.value
        # Scale influence range: (prediction_confidence - 0.5) is [-0.5, 0.5]. Multiply by 2*influence to get range [-influence, +influence] around 1.0
        confidence_factor = 1.0 + (prediction_confidence - 0.5) * conf_influence * 2

        # Volatility Factor: Scales stake inversely based on volatility.
        # Centered around 1.0. Higher volatility -> lower factor. Uses volatility_influence hyperparam.
        vol_influence = self.volatility_influence.value
        # Use (1 - vol * influence * scale_factor) to decrease stake with volatility.
        # Scale vol_influence impact (e.g., by 10) as scaled_volatility is small. Ensure factor > 0.
        volatility_factor = max(0.1, 1.0 - (scaled_volatility * vol_influence * 10)) # Ensure factor doesn't go below 0.1

        # Start with proposed stake and apply adjustments
        stake_amount = proposed_stake * confidence_factor * volatility_factor

        # Apply the general stake scaling factor
        stake_amount *= self.stake_scaling_factor.value

        # --- Apply Constraints ---
        # Ensure stake is within min/max bounds provided by Freqtrade
        stake_amount = min(stake_amount, max_stake) # Cap at exchange/balance max

        # Ensure stake meets minimum requirement
        if min_stake and stake_amount < min_stake:
            # logger.info(f"[STAKE MIN] Pair: {pair} | Calculated: {stake_amount:.2f} < Min: {min_stake:.2f}. Using Min.")
            stake_amount = min_stake
        # Re-check max_stake cap after potential min_stake adjustment
        stake_amount = min(stake_amount, max_stake)


        # logger.info(
        #     f"[STAKE ADJUSTED] Pair: {pair} | Proposed: {proposed_stake:.2f} | HV: {historical_volatility:.4f} | Conf: {prediction_confidence:.2f} | "
        #     f"Conf Factor: {confidence_factor:.3f} | Vol Factor: {volatility_factor:.3f} | Scaling: {self.stake_scaling_factor.value:.2f} | "
        #     f"Final: {stake_amount:.2f} (Min: {min_stake}, Max: {max_stake})"
        # )

        return stake_amount

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade based on market risk and prediction confidence.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            logger.warning(f"dataframe empty!")
            return 1.0  # Default leverage

        last_candle = dataframe.iloc[-1]
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        prediction_confidence = last_candle.get("prediction_confidence", 0.5)

        # ✅ Calculate leverage based on volatility and confidence
        volatility_factor = 1 - historical_volatility  # Lower volatility -> higher leverage
        confidence_factor = prediction_confidence  # Higher confidence -> higher leverage

        # ✅ Define the leverage range
        min_leverage = 1
        strategy_max_leverage = self.leverage_range_end.value

        # ✅ Calculate the base leverage (midpoint of the range)
        base_leverage = (min_leverage + strategy_max_leverage) / 2

        # ✅ Adjust leverage based on volatility and confidence
        leverage_adjustment = (
            (volatility_factor - 0.5) * self.volatility_influence.value +
            (confidence_factor - 0.5) * self.confidence_influence.value
        ) * (max_leverage - min_leverage)

        # ✅ Apply the adjustment to the base leverage
        leverage_value = base_leverage + leverage_adjustment

        # ✅ Clip leverage to be within the allowed range
        leverage_value = int(min(max(min_leverage, leverage_value), strategy_max_leverage, max_leverage))

        # logger.info(f"[LEVERAGE] Pair: {pair} | Side: {side} | Confidence: {prediction_confidence:.2f} | Leverage: {leverage_value:.2f}")
        if self.use_leverage:
            return leverage_value
        else: 
            return 1.0
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, 
                            current_time, entry_tag, side: str, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1]

        confidence = last_candle.get("prediction_confidence", 0.5)
        min_trade_size = amount * 0.5
        max_trade_size = amount * 1.5

        adjusted_size = max(min_trade_size, min(amount * confidence, max_trade_size))

        return super().confirm_trade_entry(pair, order_type, adjusted_size, rate, time_in_force, 
                                            current_time, entry_tag, side, **kwargs)


    def compute_prediction_metrics(self, dataframe: pd.DataFrame, metadata: dict, label_col: str= "T", prediction_col: str = "&-s_target", log_metrics: bool = True) -> pd.DataFrame: 
        """
        Computes and stores prediction accuracy metrics for all trading pairs.
        Saves the results to a CSV file after backtesting.
        """
        prediction_mean = prediction_col + "_mean"
        prediction_std = prediction_col + "_std"

        if log_metrics:
            logger.info(f"🔍 {label_col} | mean: {dataframe[label_col].mean()} , min: {dataframe[label_col].min()}, max: {dataframe[label_col].max()}")
            logger.info(f"🔍 {prediction_col} | mean: {dataframe[prediction_col].mean()} , min: {dataframe[prediction_col].min()}, max: {dataframe[prediction_col].max()}")
            logger.info(f"🔍 {prediction_mean} | mean: {dataframe[prediction_mean].mean()} , min: {dataframe[prediction_mean].min()}, max: {dataframe[prediction_mean].max()}")
            logger.info(f"🔍 {prediction_std} | mean: {dataframe[prediction_std].mean()} , min: {dataframe[prediction_std].min()}, max: {dataframe[prediction_std].max()}")

        # Ensure required columns exist
        if prediction_col not in dataframe.columns:
            logger.warning(f"❌ Column '{prediction_col}' not found in dataframe. Skipping prediction metrics.")
            return dataframe

        # ✅ Step 1: Directional Accuracy (Sign Match)
        dataframe["prediction_correct"] = (np.sign(dataframe[label_col]) == np.sign(dataframe[prediction_col])).astype(int)

        # ✅ Step 2: Rolling Accuracy (Last 50 candles)
        dataframe["rolling_accuracy"] = dataframe["prediction_correct"].rolling(50, min_periods=1).mean()

        # ✅ Step 3: Mean Absolute Error (MAE)
        dataframe["mae"] = np.abs(dataframe[label_col] - dataframe[prediction_col]).rolling(100, min_periods=1).mean()

        # ✅ Step 4: Prediction Confidence (Normalized by Standard Deviation)
        std_col = prediction_std
        if std_col in dataframe.columns:
            dataframe["prediction_confidence"] = (np.abs(dataframe[prediction_col]) / (dataframe[std_col] + 1e-6)).clip(0, 1)

            # Confidence score is only counted for correct predictions
            dataframe["confidence_correct"] = np.where(
                dataframe["prediction_correct"] == 1, dataframe["prediction_confidence"], 0
            )

            # Normalize avg confidence over correct predictions
            correct_preds = dataframe["prediction_correct"].rolling(100, min_periods=1).sum()
            dataframe["avg_confidence_correct"] = dataframe["confidence_correct"].rolling(100, min_periods=1).sum() / (correct_preds + 1e-6)
        else:
            logger.warning(f"⚠️ Column '{std_col}' not found. Skipping confidence tracking.")
            dataframe["avg_confidence_correct"] = np.nan

        # ✅ Step 5: Calculate Fraction of Predicted Targets
        total_predictions = (dataframe["do_predict"] == 1).sum()
        if log_metrics:
            logger.info(f"🔍 `do_predict=1` Count: {total_predictions}, `do_predict=-1` Count: {(dataframe['do_predict'] == -1).sum()}")
        total_targets_available = dataframe[label_col].notna().sum()
        fraction_predicted = total_predictions / total_targets_available if total_targets_available > 0 else 0

        # ✅ Step 6: Store Metrics in Class-Level List
        pair = metadata["pair"]
        metrics = {
            "pair": pair,
            "total_predictions": total_predictions,
            "fraction_predicted": fraction_predicted,
            "rolling_accuracy": dataframe["rolling_accuracy"].iloc[-1],
            "mae": dataframe["mae"].iloc[-1],
            "avg_confidence_correct": dataframe["avg_confidence_correct"].iloc[-1] if "avg_confidence_correct" in dataframe.columns else np.nan,
            "correlation": dataframe[prediction_col].corr(dataframe[label_col])  # ✅ Step 8: Correlation between Target and Predictions
        }
        self.prediction_metrics_storage.append(metrics)

        # ✅ Step 7: Log Key Statistics
        if log_metrics:
            logger.info(
                "🔍 Prediction Metrics | Pair: %s | Total Predictions: %s | Fraction Predicted: %.4f | Rolling Accuracy: %.4f | MAE: %.6f | Avg Confidence: %.4f | Correlation: %.4f",
                pair, total_predictions, fraction_predicted, metrics["rolling_accuracy"], metrics["mae"], metrics["avg_confidence_correct"], metrics["correlation"]
            )

        return dataframe

    def save_prediction_metrics(self, filename="prediction_metrics.csv"):
        """
        Saves the accumulated prediction metrics to a CSV file after backtesting.
        """
        if not self.prediction_metrics_storage:
            logger.warning("⚠️ No prediction metrics found to save.")
            return

        df = pd.DataFrame(self.prediction_metrics_storage)
        output_path = os.path.join(self.config["user_data_dir"], filename)
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Prediction metrics saved to {output_path}")

    def get_order_flow_features(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Calculates basic order flow features from trade data, using Freqtrade's built-in columns.
        Handles potential errors gracefully.
        """

        try:
            # ✅ Check if the required columns exist
            if not all(col in dataframe.columns for col in ['trades_count', 'volume_weighted_average_price', 'buy_ratio']):
                logger.warning(f"Required order flow columns not found in dataframe for {metadata['pair']}. Skipping order flow features.")
                dataframe['order_flow_volume'] = 0
                dataframe['order_flow_buy_ratio'] = 0
                return dataframe

            # ✅ Calculate order flow volume (using volume_weighted_average_price * trades_count as a proxy)
            # This is a simplified calculation; adjust as needed based on your data
            dataframe['order_flow_volume'] = dataframe['volume_weighted_average_price'] * dataframe['trades_count']

            # ✅ Use the built-in buy_ratio column
            dataframe['order_flow_buy_ratio'] = dataframe['buy_ratio']

            logger.info(f"✅ Successfully calculated order flow features for {metadata['pair']}.")

        except Exception as e:
            logger.exception(f"❌ Error calculating order flow features for {metadata['pair']}: {e}")
            dataframe['order_flow_volume'] = 0
            dataframe['order_flow_buy_ratio'] = 0

        return dataframe
    def load_historical_fng_data(self) -> pd.DataFrame | None:
        """
        Loads historical Fear & Greed Index data from a CSV file.
        """
        fng_data_path = self.config.get('fng_data_path')
        if fng_data_path and os.path.exists(fng_data_path):
            try:
                fng_data = pd.read_csv(fng_data_path, index_col='timestamp', parse_dates=True)
                print(f"✅ Loaded historical Fear & Greed Index data from {fng_data_path}")
                return fng_data
            except Exception as e:
                print(f"❌ Error loading historical Fear & Greed Index data: {e}")
                return None
        else:
            if fng_data_path:
                print(f"❌ Historical Fear & Greed Index data file not found at {fng_data_path}")
            else:
                print("ℹ️ No historical Fear & Greed Index data path specified in config.")
            return None

    def get_fear_and_greed_index(self, current_date=None):
        """
        Fetches the Crypto Fear & Greed Index from Alternative.me or historical data.
        """
        if self.dp and self.dp.runmode in (RunMode.BACKTEST, RunMode.HYPEROPT) and self.historical_fng_data is not None:
            # ✅ Use historical data during backtesting
            if current_date in self.historical_fng_data.index:
                row = self.historical_fng_data.loc[current_date]
                return int(row['value']), row['value_classification']
            else:
                # Handle missing data in historical dataset
                return None, None
        else:
            # ✅ Use API for live trading
            try:
                response = requests.get("https://api.alternative.me/fng/?limit=0")
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()
                if data and data['data']:
                    # Assuming you want the latest value
                    latest_data = data['data'][0]
                    # logger.info(f"fng: {latest_data}")
                    return int(latest_data['value']), latest_data['value_classification']
                else:
                    return None, None
            except requests.exceptions.RequestException as e:
                print(f"Error fetching Fear & Greed Index: {e}")
                return None, None