import logging
import operator
from functools import reduce
from typing import Dict, Optional
import joblib
import os
from datetime import datetime
# from regex import F
import requests

import numpy as np
import pandas as pd
from sympy import false, use
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
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.persistence import Trade
from freqtrade.enums import RunMode
from freqtrade.strategy.parameters import DecimalParameter
from freqtrade.vendor.qtpylib.indicators import crossed_above, crossed_below

logger = logging.getLogger(__name__)


class LSTMStrategy_v50(IStrategy):
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
                "Prediction": {"color": "purple", "plot_type": "line"},  # Rename "&-s_target" to "Prediction"
                "Avg Prediction": {"color": "brown", "plot_type": "line"},  # Rename "&-s_target_mean" to "Avg Prediction"
                "long_threshold": {"color": "green", "plot_type": "line"},
                "short_threshold": {"color": "red", "plot_type": "line"},
            },
            "Confidence": {
                "prediction_confidence": {"color": "orange", "plot_type": "scatter"},  # Plot prediction confidence
                "confidence_threshold" : {"color": "brown", "plot_type": "scatter"},
                "do_predict": {"color": "purple", "plot_type": "scatter"},  # Plot do_predict
            },
            "Indicators": {
                "atr_scaled": {"color": "green", "plot_type": "line"},
                "vol_rank": {"color": "blue", "plot_type": "line"},
            },
            "Thresholds": {
                "rolling_trend_scaled": {"color": "blue", "plot_type": "line"},
                "rolling_trend_threshold": {"color": "green", "plot_type": "line"},
            },
        },
    }

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
    dynamic_long_threshold_multiplier = DecimalParameter(0.5, 2.0, default=1.0, decimals=2, space="buy", load=True, optimize=True)
    dynamic_short_threshold_multiplier = DecimalParameter(0.5, 2.0, default=1.0, decimals=2, space="buy", load=True, optimize=True)
    use_crossed_entry = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    use_vol_filter = CategoricalParameter([True, False], default=False, space="buy", load=True, optimize=hyperopt_categorical)
    vol_rank_threshold = DecimalParameter(0.05, 0.9, default=0.25, decimals=2, space="buy", load=True, optimize=True)
    high_confidence_threshold = DecimalParameter(0.7, 0.95, default=0.85, decimals=2, space="buy", load=True, optimize=True)
    use_confidence_filter_entry = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    confidence_threshold_multiplier = DecimalParameter(0.1, 1.0, default=1.0, decimals=2, space="buy", load=True, optimize=True)
    use_trend_filter = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    rolling_trend_threshold_multiplier = DecimalParameter(0.2, 2.0, default=1.0, decimals=2, space="buy", load=True, optimize=True)
    vol_window = IntParameter(5, 500, default=24, space="buy", load=True, optimize=True)  # Window for volatility calculation
    trend_window = IntParameter(5, 500, default=48, space="buy", load=True, optimize=True)  # Window for trend calculation

    # Exit trend hyperopt parameters
    use_target_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)    
    use_trend_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    use_timed_exit = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    timed_exit_long_threshold = IntParameter(10, 300, default=20, space="sell", load=True, optimize=True)
    timed_exit_short_threshold = IntParameter(10, 300, default=20, space="sell", load=True, optimize=True)

    # ✅ Stoploss Hyperopt Parameters
    soft_stoploss_pct = DecimalParameter(-0.25, -0.03, default=-0.05, decimals=2, space="sell", load=True, optimize=True)
    min_profit_for_trailing = DecimalParameter(0.001, 0.05, default=0.005, decimals=2, space="sell", load=True, optimize=True)
    atr_stoploss_multiplier = DecimalParameter(1.0, 10.0, default=3.0, decimals=2, space="sell", load=True, optimize=True)
    historical_volatility_factor = DecimalParameter(0.2, 2.0, default=0.5, decimals=2, space="sell", load=True, optimize=True) # Used in ATR-based calculation scaling
    prediction_confidence_factor = DecimalParameter(0.2, 2.0, default=0.6, decimals=2, space="sell", load=True, optimize=True) # Used in ATR-based calculation scaling
    max_loss_floor = DecimalParameter(0.01, 0.05, default=0.03, decimals=2, space="sell", load=True, optimize=True) # Min % for max loss cap
    max_loss_ceiling = DecimalParameter(0.05, 0.25, default=0.08, decimals=2, space="sell", load=True, optimize=True) # Max % for max loss cap
    max_loss_vol_multiplier = DecimalParameter(0.5, 5.0, default=2.0, decimals=2, space="sell", load=True, optimize=True) # Volatility sensitivity for max loss cap
    initial_stop_duration_candles = IntParameter(1, 10, default=3, space="sell", load=True, optimize=True) # Duration in candles for initial stop

    # ✅ Stake amount hyperopt parameters
    stake_scaling_factor = DecimalParameter(0.1, 2.0, default=1.0, decimals=2, space="buy", load=True, optimize=True)

    # ✅ Leverage Hyperopt Parameters
    leverage_range_end = IntParameter(2, 10, default=3, space="buy", load=True, optimize=use_leverage)
    volatility_influence = DecimalParameter(0.0, 1.0, default=0.2, decimals=2, space="buy", load=True, optimize=use_leverage)
    confidence_influence = DecimalParameter(0.0, 1.0, default=0.2, decimals=2, space="buy", load=True, optimize=use_leverage)    

    # # Buy hyperspace params:
    # buy_params = {
    #     "confidence_threshold_multiplier": 0.44037,
    #     "dynamic_long_threshold_multiplier": 0.71711,
    #     "dynamic_short_threshold_multiplier": 1.04467,
    #     "high_confidence_threshold": 0.7768,
    #     "rolling_trend_threshold_multiplier": 1.65927,
    #     "stake_scaling_factor": 1.9934,
    #     "trend_window": 87,
    #     "vol_rank_threshold": 0.48934,
    #     "vol_window": 66,
    #     "confidence_influence": 0.2,  # value loaded from strategy
    #     "leverage_range_end": 3,  # value loaded from strategy
    #     "use_confidence_filter_entry": True,  # value loaded from strategy
    #     "use_crossed_entry": True,  # value loaded from strategy
    #     "use_trend_filter": True,  # value loaded from strategy
    #     "use_vol_filter": False,  # value loaded from strategy
    #     "volatility_influence": 0.2,  # value loaded from strategy
    # }

    # # Sell hyperspace params:
    # sell_params = {
    #     "atr_stoploss_multiplier": 1.12683,
    #     "historical_volatility_factor": 0.94163,
    #     "initial_stop_duration_candles": 5,
    #     "max_loss_ceiling": 0.12431,
    #     "max_loss_floor": 0.02644,
    #     "max_loss_vol_multiplier": 3.8183,
    #     "min_profit_for_trailing": 0.01289,
    #     "prediction_confidence_factor": 0.89872,
    #     "soft_stoploss_pct": -0.13512,
    #     "timed_exit_long_threshold": 13,
    #     "timed_exit_short_threshold": 97,
    #     "use_target_exit_filter": True,  # value loaded from strategy
    #     "use_timed_exit": True,  # value loaded from strategy
    #     "use_trend_exit_filter": True,  # value loaded from strategy
    # }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 1
    }

    # Stoploss:
    stoploss = -1.0  # value loaded from strategy

    # # Trailing stop:
    # trailing_stop = False  # value loaded from strategy
    # trailing_stop_positive = 0.001  # value loaded from strategy
    # trailing_stop_positive_offset = 0.0139  # value loaded from strategy
    # trailing_only_offset_is_reached = True  # value loaded from strategy

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.trades: Dict[str, datetime] = {}  # Initialize self.trades in the constructor
        # ✅ Load historical Fear & Greed Index data if available
        # self.historical_fng_data = self.load_historical_fng_data()

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
        # dataframe = self.get_order_flow_features(dataframe, metadata)

        # ✅ Fetch Fear & Greed Index
        # current_date = dataframe['date'].iloc[-1] if 'date' in dataframe else None
        # fear_greed_value, fear_greed_classification = self.get_fear_and_greed_index(current_date)
        # dataframe['fear_greed_index'] = fear_greed_value
        # dataframe['fear_greed_index'] = dataframe['fear_greed_index'].ffill()

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

        window = 72
        
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
        Adds safeguards against NaN/inf/None values.
        """
        self.freqai_info = self.config["freqai"]

        # --- Initial Data Cleaning ---
        df['close'] = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill().bfill()
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').replace(0, 1e-9).ffill().fillna(1e-9)

        # Simplify window calculations - use consistent windows
        vol_window = self.vol_window.value
        trend_window = self.trend_window.value
        
        # Calculate basic indicators with consistent windows
        df["atr"] = ta.ATR(df, timeperiod=vol_window).ffill().bfill().fillna(1e-9)
        df["vol_rank"] = df["volume"].rolling(vol_window).rank(pct=True).fillna(0)
        df["rolling_trend"] = df["close"].pct_change(trend_window).rolling(trend_window//4).mean().fillna(0)
        
        # Calculate normalized ATR and scaling
        df["atr_normalized"] = df["atr"] / df["close"].clip(lower=1e-9)
        df["atr_scaled"] = (df["atr_normalized"] / df["atr_normalized"].rolling(trend_window).quantile(0.90)
                        .clip(lower=1e-9)).clip(0, 1).fillna(0)
        
        # Calculate rolling trend scaled for visualization and signals
        df["rolling_trend_scaled"] = (df["rolling_trend"] / df["rolling_trend"].rolling(trend_window).std()
                                .clip(lower=1e-9)).clip(-3, 3).fillna(0)
        
        # Call FreqAI for ML Predictions
        df = self.freqai.start(df, metadata, self)
        
        # Process FreqAI outputs - handle missing columns gracefully
        freqai_cols = ["&-s_target", "&-s_target_mean", "&-s_target_std", "do_predict"]
        for col in freqai_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').ffill().fillna(0)
            else:
                logger.warning(f"FreqAI column '{col}' missing after freqai.start(). Assigning 0.")
                df[col] = 0.0
        
        # Calculate prediction trend - NEW
        df["pred_direction_change"] = np.sign(df["&-s_target"]).diff().ne(0).astype(int)
        df["pred_trend"] = df["&-s_target"].rolling(5).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, 
            raw=True
        ).fillna(0)
        
        # Calculate prediction strength - NEW
        df["prediction_strength"] = (np.abs(df["&-s_target"]) / 
                                (df["&-s_target_std"].clip(lower=1e-6))).clip(0, 3)
        
        # Store threshold base values
        df["dynamic_long_threshold_base"] = df["&-s_target_mean"] + df["&-s_target_std"] * df["atr_scaled"]
        df["dynamic_short_threshold_base"] = df["&-s_target_mean"] - df["&-s_target_std"] * df["atr_scaled"]
        df["rolling_trend_threshold_base"] = df["rolling_trend_scaled"].rolling(100, min_periods=10).median().ffill().fillna(0)
        
        # Calculate final thresholds with parameter multipliers
        df["long_threshold"] = (df["dynamic_long_threshold_base"] * 
                            self.dynamic_long_threshold_multiplier.value).fillna(0)
        df["short_threshold"] = (df["dynamic_short_threshold_base"] * 
                            self.dynamic_short_threshold_multiplier.value).fillna(0)
        df["rolling_trend_threshold"] = (df["rolling_trend_threshold_base"] * 
                                    self.rolling_trend_threshold_multiplier.value).fillna(0)
        
        # Store target data for plotting
        df["T"] = self.create_target_T(df)
        df["Prediction"] = df["&-s_target"]
        df["Avg Prediction"] = df["&-s_target_mean"]
        df["True Label"] = df["T"]
        
        # Compute prediction metrics and confidence
        log_metrics = (
            not hasattr(self, "dp")
            or getattr(self.dp, "runmode", None) in [RunMode.BACKTEST, RunMode.HYPEROPT, RunMode.PLOT]
        )
        self.compute_prediction_metrics(df, metadata, log_metrics=log_metrics)
        self.save_prediction_metrics()
        
        # Process confidence data
        if "prediction_confidence" in df.columns:
            df["prediction_confidence"] = pd.to_numeric(df["prediction_confidence"], errors='coerce').ffill().fillna(0)
        else:
            logger.warning("FreqAI column 'prediction_confidence' missing after compute_prediction_metrics. Assigning 0.")
            df["prediction_confidence"] = 0.0
        
        # Calculate confidence thresholds
        df["confidence_threshold_base"] = df["prediction_confidence"].rolling(100).quantile(0.5).ffill().fillna(0)
        df["confidence_threshold"] = (df["confidence_threshold_base"] * 
                                    self.confidence_threshold_multiplier.value).fillna(0)
        
        # Calculate trade duration
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
                    if 'date' in df.columns:
                        df['trade_duration'] = ((pd.to_datetime(df['date']) - open_since).dt.total_seconds() / 60 
                                            / self.timeframe_minutes)
                    else:
                        df['trade_duration'] = 0
            else:
                df['trade_duration'] = 0
        
        return df
    
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines the entry signal for both long and short trades.
        Uses target predictions and strength metrics for clear entry conditions.
        """
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None

        # Base conditions - simplified with explicit step-by-step logic
        base_conditions = df["do_predict"] == 1
        
        if self.use_vol_filter.value:
            base_conditions &= df["vol_rank"] > self.vol_rank_threshold.value
            
        if self.use_confidence_filter_entry.value:
            base_conditions &= df["prediction_confidence"] > df["confidence_threshold"]
        
        # Trend conditions
        long_trend = ~self.use_trend_filter.value | (df["rolling_trend_scaled"] > df["rolling_trend_threshold"])
        short_trend = ~self.use_trend_filter.value | (df["rolling_trend_scaled"] < df["rolling_trend_threshold"])

        # Entry signals using prediction strength
        if self.use_crossed_entry.value:
            long_signal = crossed_above(df["&-s_target"], df["long_threshold"])
            short_signal = crossed_below(df["&-s_target"], df["short_threshold"])
        else:
            # Direct comparison with thresholds
            long_signal = df["&-s_target"] > df["long_threshold"]
            short_signal = df["&-s_target"] < df["short_threshold"]
        
        # Apply entry conditions with strength-based tags
        # Strong signal: high prediction strength (signal to noise ratio)
        strong_signal = df["prediction_strength"] > 2.0
        
        # Long entries
        df.loc[
            long_signal & base_conditions & long_trend & strong_signal,
            ["enter_long", "enter_tag"]
        ] = (1, "long_strong")
        
        df.loc[
            long_signal & base_conditions & long_trend & ~strong_signal & (df["enter_long"] == 0),
            ["enter_long", "enter_tag"]
        ] = (1, "long")
        
        # Short entries
        df.loc[
            short_signal & base_conditions & short_trend & strong_signal,
            ["enter_short", "enter_tag"]
        ] = (1, "short_strong")
        
        df.loc[
            short_signal & base_conditions & short_trend & ~strong_signal & (df["enter_short"] == 0),
            ["enter_short", "enter_tag"]
        ] = (1, "short")
        
        # High confidence trend-based entry (fallback)
        high_confidence = df["prediction_confidence"] > self.high_confidence_threshold.value
        
        # Only trigger fallback entries if we have high prediction confidence
        if self.use_trend_filter.value:
            # Long fallback on trend crossing with positive prediction
            df.loc[
                (df["enter_long"] == 0) & high_confidence & 
                (df["&-s_target"] > 0) & 
                crossed_above(df["rolling_trend_scaled"], df["rolling_trend_threshold"]) &
                base_conditions,
                ["enter_long", "enter_tag"]
            ] = (1, "long_trend")
            
            # Short fallback on trend crossing with negative prediction
            df.loc[
                (df["enter_short"] == 0) & high_confidence & 
                (df["&-s_target"] < 0) &
                crossed_below(df["rolling_trend_scaled"], df["rolling_trend_threshold"]) &
                base_conditions,
                ["enter_short", "enter_tag"]
            ] = (1, "short_trend")

        return df 
    
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines exit signals for both long and short trades.
        Uses prediction trends and threshold crossings for clear exit conditions.
        """
        df['exit_long'] = False
        df['exit_short'] = False
        df['exit_tag'] = None

        # Base exit condition - quality data filter
        base_exit_condition = (
            (df['prediction_confidence'] > df["confidence_threshold"]) & 
            (df["do_predict"] == 1)
        )
        
        # Add volume filter if enabled
        if self.use_vol_filter.value:
            base_exit_condition &= df['vol_rank'] > self.vol_rank_threshold.value

        # 1. Target-based exits (Primary)
        if self.use_target_exit_filter.value:
            # Exit long positions when:
            # - Target crosses below threshold, OR
            # - Direction changes with negative trend
            df.loc[
                (crossed_below(df['&-s_target'], df["long_threshold"]) | 
                ((df["pred_direction_change"] == 1) & (df["pred_trend"] < 0))) &
                base_exit_condition,
                ['exit_long', 'exit_tag']
            ] = (True, 'target_reversal')
            
            # Exit short positions when:
            # - Target crosses above threshold, OR
            # - Direction changes with positive trend
            df.loc[
                (crossed_above(df['&-s_target'], df["short_threshold"]) | 
                ((df["pred_direction_change"] == 1) & (df["pred_trend"] > 0))) &
                base_exit_condition,
                ['exit_short', 'exit_tag']
            ] = (True, 'target_reversal')

        # 2. Trend-based exits (Secondary)
        if self.use_trend_exit_filter.value:
            # Only apply if target exit hasn't triggered
            df.loc[
                (df['exit_long'] == False) &
                crossed_below(df['rolling_trend_scaled'], df["rolling_trend_threshold"]) &
                base_exit_condition,
                ['exit_long', 'exit_tag']
            ] = (True, 'trend_reversal')
            
            df.loc[
                (df['exit_short'] == False) &
                crossed_above(df['rolling_trend_scaled'], df["rolling_trend_threshold"]) &
                base_exit_condition,
                ['exit_short', 'exit_tag']
            ] = (True, 'trend_reversal')
        
        # 3. Prediction strength weakening exit
        # Exit when prediction strength drops significantly
        strength_drop = (df['prediction_strength'].shift(1) - df['prediction_strength']) / df['prediction_strength'].shift(1).clip(lower=0.1)
        significant_drop = strength_drop > 0.3  # 30% drop in strength
        
        df.loc[
            (df['exit_long'] == False) &
            significant_drop &
            (df['prediction_strength'] < 1.0) &  # Low absolute strength
            base_exit_condition &
            (df['trade_duration'] > 3),  # At least 3 candles in trade
            ['exit_long', 'exit_tag']
        ] = (True, 'strength_weakening')
        
        df.loc[
            (df['exit_short'] == False) &
            significant_drop &
            (df['prediction_strength'] < 1.0) &  # Low absolute strength
            base_exit_condition &
            (df['trade_duration'] > 3),  # At least 3 candles in trade
            ['exit_short', 'exit_tag']
        ] = (True, 'strength_weakening')

        # 4. Time-based exit (Last resort)
        if self.use_timed_exit.value:
            df.loc[
                (df['exit_long'] == False) &
                (df['trade_duration'] > self.timed_exit_long_threshold.value) &
                (df['prediction_confidence'] < df['confidence_threshold']),  # Low confidence
                ['exit_long', 'exit_tag']
            ] = (True, 'timed_exit')
            
            df.loc[
                (df['exit_short'] == False) &
                (df['trade_duration'] > self.timed_exit_short_threshold.value) &
                (df['prediction_confidence'] < df['confidence_threshold']),  # Low confidence
                ['exit_short', 'exit_tag']
            ] = (True, 'timed_exit')

        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Enhanced custom stoploss with advanced adaptivity features.
        Uses prediction trend and prediction strength to adjust stoploss dynamically.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty or not trade or not trade.open_rate:
            return -1  # Keep existing stoploss

        # Check if we're in a short position
        is_short = trade.is_short if hasattr(trade, 'is_short') else False

        # --- Extract key dataframe metrics ---
        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        if atr <= 0:
            return -1

        # Get prediction metrics
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        prediction_confidence = last_candle.get("prediction_confidence", 0.5)
        prediction_strength = last_candle.get("prediction_strength", 1.0)
        prediction_trend = last_candle.get("pred_trend", 0)
        
        # Calculate trade duration
        trade_duration_candles = self.calculate_trade_duration(trade, current_time, dataframe)
            
        # --- Get hyperopt parameters ---
        soft_stoploss_pct = self.soft_stoploss_pct.value
        initial_duration = self.initial_stop_duration_candles.value
        min_profit_for_trailing = self.min_profit_for_trailing.value
        atr_multiplier = self.atr_stoploss_multiplier.value
        historical_volatility_factor = self.historical_volatility_factor.value
        prediction_confidence_factor = self.prediction_confidence_factor.value
        max_loss_floor_val = self.max_loss_floor.value
        max_loss_ceiling_val = self.max_loss_ceiling.value
        max_loss_vol_multiplier_val = self.max_loss_vol_multiplier.value
        
        # --- PHASE 1: Initial fixed stoploss period ---
        if trade_duration_candles < initial_duration:
            return soft_stoploss_pct
        
        # --- PHASE 2: Dynamic stoploss calculation ---
        
        # 1. Base calculation using ATR
        dynamic_volatility_factor = 1 + historical_volatility * historical_volatility_factor
        confidence_factor = 1 - (prediction_confidence * prediction_confidence_factor)
        stoploss_buffer_abs = atr * atr_multiplier * dynamic_volatility_factor * confidence_factor
        
        # 2. Prediction trend adjustment
        # If prediction trend favors our position, we can be slightly looser with stoploss
        # If it's against our position, tighten the stoploss
        pred_trend_adjustment = 1.0
        if (not is_short and prediction_trend > 0) or (is_short and prediction_trend < 0):
            # Trend favors position - loosen stoploss slightly (up to 20%)
            pred_trend_adjustment = 1.0 + min(abs(prediction_trend) * 2, 0.2)
        elif (not is_short and prediction_trend < 0) or (is_short and prediction_trend > 0):
            # Trend against position - tighten stoploss (up to 30%)
            pred_trend_adjustment = 1.0 - min(abs(prediction_trend) * 3, 0.3)
        
        stoploss_buffer_abs *= pred_trend_adjustment
        
        # 3. Prediction strength adjustment
        # Higher strength = more confidence in direction
        strength_adjustment = 1.0
        if prediction_strength > 1.5:
            # Very strong signal - can be slightly looser with stoploss
            strength_adjustment = 1.1
        elif prediction_strength < 0.8:
            # Weak signal - be more conservative
            strength_adjustment = 0.9
        
        stoploss_buffer_abs *= strength_adjustment
        
        # 4. Convert absolute buffer to percentage relative to open rate
        stoploss_pct = stoploss_buffer_abs / trade.open_rate
        
        # 5. Ensure stoploss doesn't exceed maximum loss limit
        max_loss_pct = min(max_loss_floor_val + historical_volatility * max_loss_vol_multiplier_val, 
                        max_loss_ceiling_val)
        
        final_stoploss_pct = max(-stoploss_pct, -max_loss_pct)
        
        # --- PHASE 3: Profit protection ---
        # Once in profit, gradually tighten stoploss based on profit level
        if current_profit > min_profit_for_trailing:
            # Define clear profit brackets with corresponding tightening factors
            if current_profit > 0.1:
                tightening = 0.25  # 75% tighter at 10%+ profit
            elif current_profit > 0.05:
                tightening = 0.4   # 60% tighter at 5%+ profit
            elif current_profit > 0.03:
                tightening = 0.6   # 40% tighter at 3%+ profit
            else:
                tightening = 0.8   # 20% tighter at 1%+ profit
            
            # Apply tightening (moves stoploss closer to current price)
            final_stoploss_pct *= tightening
        
        # --- PHASE 4: Smooth transition during ramp-up period ---
        ramp_up_window = 5  # Candles to transition from initial to main stoploss
        if initial_duration <= trade_duration_candles < initial_duration + ramp_up_window:
            ramp_progress = (trade_duration_candles - initial_duration) / ramp_up_window
            transition_stoploss = soft_stoploss_pct * (1 - ramp_progress) + final_stoploss_pct * ramp_progress
            return transition_stoploss

        return final_stoploss_pct

    def calculate_trade_duration(self, trade, current_time, dataframe):
        """
        Calculates trade duration in candles, optimized for both backtest and live conditions.
        """
        tf_minutes = timeframe_to_minutes(self.timeframe)
        
        # Time-based calculation is more consistent and simpler
        elapsed_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        trade_duration_candles = elapsed_minutes / tf_minutes
        
        return trade_duration_candles

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float,
                            min_stake: float | None, max_stake: float, leverage: float, entry_tag: str | None, 
                            side: str, **kwargs) -> float:
        """
        Dynamically adjusts stake amount based on prediction metrics and market conditions.
        Key factors: prediction strength, trend alignment and confidence.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return proposed_stake
        
        # Get key metrics from last candle
        last_candle = dataframe.iloc[-1]
        
        # 1. Core prediction metrics
        prediction_value = last_candle.get("&-s_target", 0)
        prediction_confidence = last_candle.get("prediction_confidence", 0.5)
        prediction_strength = last_candle.get("prediction_strength", 1.0)
        prediction_trend = last_candle.get("pred_trend", 0)
        
        # 2. Market condition metrics
        volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1]
        volatility = min(volatility, 0.05)  # Cap at 5% for stability
        
        # 3. Position-specific metrics
        is_short = side == 'short'
        
        # 4. Signal alignment check (prediction supports position direction)
        signal_aligned = (not is_short and prediction_value > 0) or (is_short and prediction_value < 0)
        
        # --- Core stake adjustment factors ---
        
        # Base stake starts with proposed amount and general scaling
        base_stake = proposed_stake * self.stake_scaling_factor.value
        
        # Confidence factor: higher confidence = higher stake
        # Center around 1.0, allow 0.7-1.3x adjustment based on confidence
        confidence_effect = 0.6  # How much confidence influences stake
        confidence_factor = 1.0 + (prediction_confidence - 0.5) * confidence_effect
        
        # Prediction strength factor: stronger signal = higher stake
        # Maps 0.5-2.0 strength to 0.8-1.2x adjustment
        strength_factor = 0.8 + (min(prediction_strength, 2.0) / 5.0)
        
        # Volatility factor: higher volatility = lower stake
        # Maps 0.5% - 5% volatility to 1.0-0.5x adjustment 
        volatility_factor = max(0.5, 1.0 - volatility * 10)
        
        # Trend alignment factor: stronger trend = higher stake
        # Adds up to 20% stake for strong trend in position direction
        trend_factor = 1.0
        if abs(prediction_trend) > 0.01:  # Meaningful trend
            # If trend direction supports position, boost factor
            trend_supports_position = (not is_short and prediction_trend > 0) or (is_short and prediction_trend < 0)
            if trend_supports_position:
                trend_factor = 1.0 + min(abs(prediction_trend) * 2, 0.2)  # Up to 20% boost
            else:
                trend_factor = 1.0 - min(abs(prediction_trend) * 2, 0.3)  # Up to 30% reduction
        
        # Signal quality factor: reduce stake if signal doesn't support position
        alignment_factor = 1.0 if signal_aligned else 0.6  # 40% reduction if misaligned
        
        # Entry tag adjustment: stronger positions for strong signals
        tag_factor = 1.2 if entry_tag in ['long_strong', 'short_strong'] else 1.0
        
        # --- Calculate final stake amount ---
        stake_amount = base_stake * confidence_factor * strength_factor * volatility_factor * trend_factor * alignment_factor * tag_factor
        
        # --- Ensure within limits ---
        if min_stake:
            stake_amount = max(min_stake, stake_amount)
        stake_amount = min(stake_amount, max_stake)
        
        return stake_amount

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                **kwargs) -> float:
        """
        Calculates optimal leverage based on prediction strength, volatility and trend alignment.
        Higher leverage for strong signals with low volatility, lower leverage for uncertain conditions.
        """
        if not self.use_leverage:
            return 1.0
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return 1.0  # Default leverage
        
        # Get key metrics from last candle
        last_candle = dataframe.iloc[-1]
        
        # --- Extract prediction metrics ---
        prediction_value = last_candle.get("&-s_target", 0)
        prediction_confidence = last_candle.get("prediction_confidence", 0.5)
        prediction_strength = last_candle.get("prediction_strength", 1.0)
        prediction_trend = last_candle.get("pred_trend", 0)
        
        # --- Calculate market risk factors ---
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        volatility = min(historical_volatility, 0.05)  # Cap at 5% for stability
        
        # --- Position details ---
        is_short = side == 'short'
        
        # --- Define the leverage range ---
        min_leverage = 1.0
        strategy_max_leverage = min(self.leverage_range_end.value, max_leverage)
        
        # --- Base leverage calculation ---
        # 1. Signal Quality Component (0-1 scale)
        signal_quality = prediction_confidence * min(prediction_strength / 2, 1.0)
        
        # 2. Risk Component (0-1 scale, inverted)
        risk_factor = 1.0 - (volatility * 10)  # 0.5% vol -> 0.95, 5% vol -> 0.5
        risk_factor = max(0.2, min(risk_factor, 1.0))  # Limit range to 0.2-1.0
        
        # 3. Trend Alignment Component (-0.2 to +0.2 adjustment)
        trend_alignment = 0.0
        if abs(prediction_trend) > 0.001:
            trend_supports_position = (not is_short and prediction_trend > 0) or (is_short and prediction_trend < 0)
            if trend_supports_position:
                trend_alignment = min(abs(prediction_trend) * 4, 0.2)  # Boost up to 0.2
            else:
                trend_alignment = -min(abs(prediction_trend) * 4, 0.2)  # Reduce up to 0.2
        
        # 4. Entry Tag Component (0.0 to 0.2 boost)
        tag_boost = 0.2 if entry_tag in ['long_strong', 'short_strong'] else 0.0
        
        # --- Calculate core leverage factors ---
        # Scale signal quality by confidence influence
        signal_factor = min_leverage + (signal_quality * self.confidence_influence.value * (strategy_max_leverage - min_leverage))
        
        # Scale risk factor by volatility influence
        risk_adjustment = risk_factor * self.volatility_influence.value * (strategy_max_leverage - min_leverage)
        
        # Apply trend alignment adjustment
        trend_adjustment = trend_alignment * (strategy_max_leverage - min_leverage)
        
        # Apply tag boost
        tag_adjustment = tag_boost * (strategy_max_leverage - min_leverage)
        
        # --- Combine all factors for final leverage ---
        leverage_value = signal_factor + risk_adjustment + trend_adjustment + tag_adjustment
        
        # --- Ensure leverage stays within bounds ---
        leverage_value = min(max(min_leverage, leverage_value), strategy_max_leverage)
        
        # --- Ensure integer leverage for most exchanges ---
        leverage_value = round(leverage_value)
        
        return leverage_value
    
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
        MODIFIED: Only the prediction_confidence calculation is changed to incorporate rolling accuracy.
        """
        prediction_mean = prediction_col + "_mean"
        prediction_std = prediction_col + "_std"

        # --- Keep Existing Logging ---
        if log_metrics:
            # Add checks to prevent errors if columns don't exist or are all NaN
            if label_col in dataframe and dataframe[label_col].notna().any():
                logger.info(f"🔍 {label_col} | mean: {dataframe[label_col].mean()} , min: {dataframe[label_col].min()}, max: {dataframe[label_col].max()}")
            if prediction_col in dataframe and dataframe[prediction_col].notna().any():
                logger.info(f"🔍 {prediction_col} | mean: {dataframe[prediction_col].mean()} , min: {dataframe[prediction_col].min()}, max: {dataframe[prediction_col].max()}")
            if prediction_mean in dataframe and dataframe[prediction_mean].notna().any():
                logger.info(f"🔍 {prediction_mean} | mean: {dataframe[prediction_mean].mean()} , min: {dataframe[prediction_mean].min()}, max: {dataframe[prediction_mean].max()}")
            if prediction_std in dataframe and dataframe[prediction_std].notna().any():
                logger.info(f"🔍 {prediction_std} | mean: {dataframe[prediction_std].mean()} , min: {dataframe[prediction_std].min()}, max: {dataframe[prediction_std].max()}")

        # --- Keep Existing Column Check ---
        if prediction_col not in dataframe.columns:
            logger.warning(f"❌ Column '{prediction_col}' not found in dataframe. Skipping prediction metrics.")
            return dataframe
        # Also check for label column needed for accuracy
        if label_col not in dataframe.columns:
            logger.warning(f"❌ Column '{label_col}' not found in dataframe. Skipping prediction metrics.")
            return dataframe

        # --- Ensure Numeric Types (Needed for calculations) ---
        # Coerce necessary columns, keeping NaNs in label_col for now
        for col in [prediction_col, prediction_mean, prediction_std]:
             if col in dataframe.columns:
                 dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce').ffill().fillna(0)
        dataframe[label_col] = pd.to_numeric(dataframe[label_col], errors='coerce')


        # ✅ Step 1: Directional Accuracy (Sign Match) - Handles Label NaNs
        dataframe["prediction_correct"] = np.nan # Initialize with NaN
        valid_labels_mask = dataframe[label_col].notna()
        if valid_labels_mask.sum() > 0:
            dataframe.loc[valid_labels_mask, "prediction_correct"] = np.where(
                np.sign(dataframe.loc[valid_labels_mask, label_col]) == np.sign(dataframe.loc[valid_labels_mask, prediction_col]),
                1, 0
            )

        # ✅ Step 2: Rolling Accuracy (Last 50 candles) - Needed for refined confidence
        # Calculate rolling mean of correctness. Fill initial NaNs with 0.5 (neutral assumption)
        rolling_accuracy_window = 50 # Or use a hyperparameter
        dataframe["rolling_accuracy"] = dataframe["prediction_correct"].rolling(
            rolling_accuracy_window, min_periods=max(1, rolling_accuracy_window // 5)
        ).mean().ffill().fillna(0.5)

        # ✅ Step 3: Mean Absolute Error (MAE) - Keep Existing Calculation
        dataframe["mae"] = np.abs(dataframe[label_col] - dataframe[prediction_col]).rolling(100, min_periods=1).mean()

        # ✅ Step 4: *** MODIFIED Confidence Calculation ***
        std_col = prediction_std
        if std_col in dataframe.columns:
            # 4a. Calculate Base Confidence (Signal-to-Noise Ratio)
            base_confidence = (np.abs(dataframe[prediction_col]) / (dataframe[std_col] + 1e-6)).clip(0, 1)

            # 4b. Calculate Refined Prediction Confidence (Modulated by Rolling Accuracy)
            dataframe["prediction_confidence"] = (base_confidence * dataframe["rolling_accuracy"]).clip(0, 1).fillna(0) # MODIFIED LINE

            # 4c. Confidence score is only counted for correct predictions (Using Refined Confidence)
            # Use np.nan_to_num to handle potential NaNs from prediction_correct
            confidence_correct_array = np.where(
                dataframe["prediction_correct"] == 1, dataframe["prediction_confidence"], 0
            )
            dataframe["confidence_correct"] = np.nan_to_num(confidence_correct_array, nan=0.0) # Uses refined confidence

            # 4d. Normalize avg confidence over correct predictions - Keep Existing Calculation
            correct_preds = dataframe["prediction_correct"].rolling(100, min_periods=1).sum()
            # Ensure confidence_correct exists before rolling on it
            if "confidence_correct" in dataframe.columns:
                 dataframe["avg_confidence_correct"] = dataframe["confidence_correct"].rolling(100, min_periods=1).sum() / (correct_preds + 1e-6)
                 dataframe["avg_confidence_correct"] = dataframe["avg_confidence_correct"].fillna(0) # Fill potential NaNs from division
            else:
                 dataframe["avg_confidence_correct"] = np.nan

        else: # Keep Existing Warning
            logger.warning(f"⚠️ Column '{std_col}' not found. Skipping confidence tracking.")
            dataframe["prediction_confidence"] = 0.0 # Ensure column exists if skipped
            dataframe["confidence_correct"] = 0.0
            dataframe["avg_confidence_correct"] = np.nan

        # ✅ Step 5: Calculate Fraction of Predicted Targets - Keep Existing Calculation
        total_predictions = (dataframe["do_predict"] == 1).sum() if "do_predict" in dataframe.columns else 0
        if log_metrics and "do_predict" in dataframe.columns:
            logger.info(f"🔍 `do_predict=1` Count: {total_predictions}, `do_predict=-1` Count: {(dataframe['do_predict'] == -1).sum()}")
        total_targets_available = dataframe[label_col].notna().sum()
        fraction_predicted = total_predictions / total_targets_available if total_targets_available > 0 else 0

        # ✅ Step 6: Store Metrics in Class-Level List - Keep Existing Calculation
        pair = metadata["pair"]
        # Calculate correlation safely
        valid_df_for_corr = dataframe.dropna(subset=[label_col, prediction_col])
        correlation = valid_df_for_corr[prediction_col].corr(valid_df_for_corr[label_col]) if not valid_df_for_corr.empty else np.nan

        metrics = {
            "pair": pair,
            "total_predictions": total_predictions,
            "fraction_predicted": fraction_predicted,
            "rolling_accuracy": dataframe["rolling_accuracy"].iloc[-1] if "rolling_accuracy" in dataframe.columns else np.nan,
            "mae": dataframe["mae"].iloc[-1] if "mae" in dataframe.columns else np.nan,
            "avg_confidence_correct": dataframe["avg_confidence_correct"].iloc[-1] if "avg_confidence_correct" in dataframe.columns else np.nan,
            "correlation": correlation
        }
        # Ensure storage exists
        if not hasattr(self, 'prediction_metrics_storage'):
            self.prediction_metrics_storage = []
        self.prediction_metrics_storage.append(metrics)

        # ✅ Step 7: Log Key Statistics - Keep Existing Calculation
        if log_metrics:
            # Safely get values for logging
            log_roll_acc = metrics.get("rolling_accuracy", np.nan)
            log_mae = metrics.get("mae", np.nan)
            log_avg_conf_corr = metrics.get("avg_confidence_correct", np.nan)
            log_corr = metrics.get("correlation", np.nan)

            # Format potential NaNs
            f_roll_acc = f"{log_roll_acc:.4f}" if pd.notna(log_roll_acc) else "NaN"
            f_mae = f"{log_mae:.6f}" if pd.notna(log_mae) else "NaN"
            f_avg_conf_corr = f"{log_avg_conf_corr:.4f}" if pd.notna(log_avg_conf_corr) else "NaN"
            f_corr = f"{log_corr:.4f}" if pd.notna(log_corr) else "NaN"

            logger.info(
                f"🔍 Pair: {pair} | Total Pred: {total_predictions} | Frac Pred: {fraction_predicted:.4f} | "
                f"Roll Acc: {f_roll_acc} | MAE: {f_mae} | Avg Conf Corr: {f_avg_conf_corr} | Corr: {f_corr}"
            )

        return dataframe

    # Keep your save_prediction_metrics function exactly as provided
    def save_prediction_metrics(self, filename="prediction_metrics.csv"):
        """
        Saves the accumulated prediction metrics to a CSV file after backtesting.
        """
        # Use class attribute directly as it's defined at class level
        if not LSTMStrategy_v50.prediction_metrics_storage:
            logger.warning("⚠️ No prediction metrics found to save.")
            return

        try:
            # Use class attribute directly
            df = pd.DataFrame(LSTMStrategy_v50.prediction_metrics_storage)
            # Drop duplicates based on pair, keeping the last entry
            df = df.drop_duplicates(subset=['pair'], keep='last')
            output_path = os.path.join(self.config["user_data_dir"], filename)
            # Overwrite file each time (mode='w' is default for to_csv)
            df.to_csv(output_path, index=False)
            logger.info(f"✅ Prediction metrics saved to {output_path}")
        except Exception as e:
            logger.exception(f"Error saving prediction metrics: {e}")

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