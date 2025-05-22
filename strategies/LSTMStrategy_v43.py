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


class LSTMStrategy_v43(IStrategy):
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
            "True Label": {"color": "blue", "plot_type": "line"},
            "Prediction": {"color": "purple", "plot_type": "line"},
            "Avg Prediction": {"color": "brown", "plot_type": "line"}, 
            "long_threshold": {"color": "green", "plot_type": "line"},
            "short_threshold": {"color": "red", "plot_type": "line"},
            "exit_long_threshold": {"color": "lightgreen", "plot_type": "line"}, # Added
            "exit_short_threshold": {"color": "pink", "plot_type": "line"},      # Added
            },
            "Confidence": {
                "pred_confidence": {"color": "orange", "plot_type": "scatter"},  # Plot prediction confidence
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
                                                
    pred_metrics_storage = []  # Class-level storage for all pairs

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
    exit_long_threshold_multiplier = DecimalParameter(0.5, 1.5, default=1.0, decimals=2, space="sell", load=True, optimize=True)
    exit_short_threshold_multiplier = DecimalParameter(0.5, 1.5, default=1.0, decimals=2, space="sell", load=True, optimize=True)        
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

    # Buy hyperspace params:
    buy_params = {
        "confidence_threshold_multiplier": 0.97,
        "dynamic_long_threshold_multiplier": 1.61,
        "dynamic_short_threshold_multiplier": 0.82,
        "high_confidence_threshold": 0.88,
        "rolling_trend_threshold_multiplier": 1.66,
        "stake_scaling_factor": 1.33,
        "trend_window": 284,
        "use_confidence_filter_entry": True,
        "use_crossed_entry": False,
        "use_trend_filter": True,
        "use_vol_filter": True,
        "vol_rank_threshold": 0.81,
        "vol_window": 326,
        "confidence_influence": 0.2,  # value loaded from strategy
        "leverage_range_end": 3,  # value loaded from strategy
        "volatility_influence": 0.2,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "atr_stoploss_multiplier": 8.52,
        "historical_volatility_factor": 1.99,
        "initial_stop_duration_candles": 10,
        "max_loss_ceiling": 0.08,
        "max_loss_floor": 0.03,
        "max_loss_vol_multiplier": 2.85,
        "min_profit_for_trailing": 0.01,
        "prediction_confidence_factor": 1.36,
        "soft_stoploss_pct": -0.06,
        "timed_exit_long_threshold": 278,
        "timed_exit_short_threshold": 99,
        "use_target_exit_filter": True,
        "use_timed_exit": True,
        "use_trend_exit_filter": True,
    }

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
        Creates a new target with smoother dynamic lookahead based on volatility.
        """
        window = 48
        
        dataframe["ATR"] = ta.ATR(dataframe, timeperiod=window).ffill().bfill().fillna(1e-9)
        dataframe["close"] = dataframe["close"].replace(0, np.nan).bfill()
        
        # Smoother volatility estimation for lookahead
        vol_estimate = dataframe["ATR"] / dataframe["close"] * 100
        dataframe["smoothed_vol"] = vol_estimate.rolling(5).mean().fillna(vol_estimate)
        
        # Clip to same range but with smoother transitions
        dataframe["lookahead_dynamic"] = np.clip(dataframe["smoothed_vol"], 5, 24).fillna(10).astype(int)
        
        # Rest of your current implementation...
        future_change = []
        for i in range(len(dataframe)):
            try:
                lookahead = int(dataframe["lookahead_dynamic"].iloc[i])
                future_index = i + lookahead
                if future_index >= len(dataframe):
                    future_change.append(np.nan)
                else:
                    future_close = dataframe["close"].iloc[future_index]
                    future_change.append(future_close - dataframe["close"].iloc[i])
            except (KeyError, IndexError):
                future_change.append(np.nan)
    
        dataframe["future_change"] = future_change
        dataframe["TS"] = dataframe["future_change"].rolling(window).mean()
        dataframe["T"] = np.tanh(dataframe["TS"] / (dataframe["ATR"] + 1e-6))
        dataframe["T"] = dataframe["T"].fillna(0)
        
        return dataframe["T"]
    
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates only base indicators that don't use hyperoptable parameters.
        All parameter-dependent calculations are moved to calculate_indicators().
        """
        self.freqai_info = self.config["freqai"]
    
        # --- Initial Data Cleaning ---
        df['close'] = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill().bfill()
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').replace(0, 1e-9).ffill().fillna(1e-9)
    
        # Ensure fit_live_predictions_candles is properly set
        fit_live_candles = self.freqai_info.get("fit_live_predictions_candles", 0)
        if fit_live_candles <= 0 and getattr(self.dp, "runmode", None) in [RunMode.BACKTEST, RunMode.HYPEROPT]:
            logger.warning("fit_live_predictions_candles not properly configured. Using default of 48.")
            self.freqai_info["fit_live_predictions_candles"] = 48
        
        # Call FreqAI for ML Predictions - independent of hyperopt parameters
        df = self.freqai.start(df, metadata, self)
        
        # Process FreqAI outputs - store base values without parameter influence
        freqai_cols = ["&-s_target", "&-s_target_mean", "&-s_target_std", "do_predict"]
        for col in freqai_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').ffill().fillna(0)
            else:
                logger.warning(f"FreqAI column '{col}' missing after freqai.start(). Assigning 0.")
                df[col] = 0.0
                
        # Keep "T" for Plotting Purposes Only
        df["T"] = self.create_target_T(df)
        df["Prediction"] = df["&-s_target"]
        df["Avg Prediction"] = df["&-s_target_mean"]
        df["True Label"] = df["T"]
        
        # Compute prediction metrics - will calculate prediction_confidence
        log_metrics = (
            not hasattr(self, "dp")
            or getattr(self.dp, "runmode", None) in [RunMode.BACKTEST, RunMode.HYPEROPT, RunMode.PLOT]
        )
        self.compute_prediction_metrics(df, metadata, log_metrics=log_metrics)
        self.save_prediction_metrics()
        
        # Process confidence (created in compute_prediction_metrics)
        if "pred_confidence" in df.columns:
            df["pred_confidence"] = pd.to_numeric(df["pred_confidence"], errors='coerce').ffill().fillna(0)
        else:
            logger.warning("'pred_confidence' missing after compute_prediction_metrics. Assigning 0.")
            df["pred_confidence"] = 0.0
                
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
    
    def calculate_indicators(self, df: DataFrame) -> DataFrame:
        """
        Calculates all parameter-dependent indicators for each hyperopt epoch.
        Called from populate_entry_trend and populate_exit_trend.
        """
        # Create a copy to avoid modifying original dataframe
        # result = df.copy()
        
        # --- Get hyperopt parameter values WITH TYPE CONVERSION ---
        volatility_window = int(self.vol_window.value)  # Convert to native Python int
        trend_window = int(self.trend_window.value)     # Convert to native Python int
    
        # --- Calculate indicators with hyperopt parameters ---
        
        # 1. ATR (Volatility) with parameter-based timeperiod
        df["atr"] = ta.ATR(df, timeperiod=volatility_window).ffill().bfill().fillna(1e-9)
        df["atr_normalized"] = df["atr"] / df["close"].clip(lower=1e-9)

        # 2. Volume Rank with parameter-based window
        df["vol_rank"] = df["volume"].rolling(volatility_window).rank(pct=True).fillna(0)

        # 3. Rolling Trend with parameter-based periods
        pct_change_period = trend_window
        rolling_window = max(2, int(trend_window // 4))  # Ensure integer division result is converted to Python int
        df["rolling_trend"] = df["close"].pct_change(pct_change_period).rolling(rolling_window).mean().fillna(0)

        # 4. ATR Scaling with parameter-based window
        atr_quantile = 0.90
        quantile_val = df["atr_normalized"].rolling(trend_window).quantile(atr_quantile).ffill().bfill().fillna(1e-9)
        df["atr_scaled"] = (df["atr_normalized"] / quantile_val.clip(lower=1e-9)).clip(0, 1).fillna(0)

        # 5. Rolling Trend Scaling with parameter-based window
        mean_trend = df["rolling_trend"].rolling(trend_window).mean().ffill().fillna(0)
        std_trend = df["rolling_trend"].rolling(trend_window).std().ffill().fillna(1e-9)
        df["rolling_trend_scaled"] = (df["rolling_trend"] - mean_trend) / std_trend.clip(lower=1e-9)
        df["rolling_trend_scaled"] = df["rolling_trend_scaled"].fillna(0)

        # 6. Dynamic Thresholds with parameter-based multipliers
        df["dynamic_long_threshold_base"] = df["&-s_target_mean"] + df["&-s_target_std"] * df["atr_scaled"]
        df["dynamic_short_threshold_base"] = df["&-s_target_mean"] - df["&-s_target_std"] * df["atr_scaled"]
        df["rolling_trend_threshold_base"] = df["rolling_trend_scaled"].rolling(100, min_periods=10).median().ffill().fillna(0)

        # 7. Apply threshold multipliers from hyperopt parameters - convert to float
        df["long_threshold"] = df["dynamic_long_threshold_base"] * float(self.dynamic_long_threshold_multiplier.value)
        df["exit_long_threshold"] = df["&-s_target_mean"] * float(self.exit_long_threshold_multiplier.value)
        df["exit_short_threshold"] = df["&-s_target_mean"] * float(self.exit_short_threshold_multiplier.value)            
        df["short_threshold"] = df["dynamic_short_threshold_base"] * float(self.dynamic_short_threshold_multiplier.value)
        df["rolling_trend_threshold"] = df["rolling_trend_threshold_base"] * float(self.rolling_trend_threshold_multiplier.value)

        # 8. Confidence threshold with parameter-based multiplier - convert to float
        df["confidence_threshold_base"] = df["pred_confidence"].rolling(100).quantile(0.5).ffill().fillna(0)
        df["confidence_threshold"] = df["confidence_threshold_base"] * float(self.confidence_threshold_multiplier.value)

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines the entry signal for both long and short trades using a vectorized approach.
        Position tracking and overlap prevention are handled by populate_exit_trend,
        confirm_trade_entry, and Freqtrade's core logic.
        """
        # Initialize entry columns
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None
        
        # Get indicators with current hyperopt parameters
        processed_df = self.calculate_indicators(df)
        
        # Calculate base signals (vectorized)
        if self.use_crossed_entry.value:
            long_signal_active = crossed_above(processed_df["&-s_target"], processed_df["long_threshold"])
            short_signal_active = crossed_below(processed_df["&-s_target"], processed_df["short_threshold"])
        else:
            long_signal_active = processed_df["&-s_target"] > processed_df["long_threshold"]
            short_signal_active = processed_df["&-s_target"] < processed_df["short_threshold"]
        
        # Define base entry conditions (vectorized)
        # This function returns a boolean Series
        def base_entry_condition_series(side: str = None):
            condition = (processed_df["do_predict"] == 1)
            
            # Volume filter
            if self.use_vol_filter.value:
                condition &= (processed_df["vol_rank"] > self.vol_rank_threshold.value)
                
            # Confidence filter
            if self.use_confidence_filter_entry.value:
                condition &= (processed_df["pred_confidence"] > processed_df["confidence_threshold"])
                
            # Trend filter (direction-specific)
            if self.use_trend_filter.value:
                if side == "long":
                    condition &= (processed_df["rolling_trend_scaled"] > processed_df["rolling_trend_threshold"])
                elif side == "short":
                    condition &= (processed_df["rolling_trend_scaled"] < processed_df["rolling_trend_threshold"])
            return condition
        
        # Combine base signals with base entry conditions
        final_long_entry_condition = long_signal_active & base_entry_condition_series(side="long")
        final_short_entry_condition = short_signal_active & base_entry_condition_series(side="short")

        # Log conditions for a specific problematic candle (replace with your actual candle's index/date)
        # For example, if your problematic candle is at index 12345
        # target_candle_index = 12345 
        # if df.index[target_candle_index] == df.iloc[final_short_entry_condition.index[final_short_entry_condition]].index: # A bit complex to get specific candle
        
        # Simpler: Log when final_short_entry_condition is true
        if final_short_entry_condition.any():
            logger.info(f"DEBUG: Pair: {metadata['pair']}, Candle: {df.loc[final_short_entry_condition].index.values[0] if final_short_entry_condition.any() else 'N/A'}")
            logger.info(f"DEBUG: short_signal_active: {short_signal_active[final_short_entry_condition].iloc[0] if final_short_entry_condition.any() else 'N/A'}")
            logger.info(f"DEBUG: base_entry_condition_series(short): {base_entry_condition_series(side='short')[final_short_entry_condition].iloc[0] if final_short_entry_condition.any() else 'N/A'}")
            logger.info(f"DEBUG: df['enter_long'] on that candle: {df.loc[final_short_entry_condition, 'enter_long'].iloc[0] if final_short_entry_condition.any() else 'N/A'}")

        # Apply standard long entries
        df.loc[final_long_entry_condition, ["enter_long", "enter_tag"]] = (1, "long")
        
        # Apply standard short entries
        # Ensure short entries don't overwrite long entries on the same candle if both somehow trigger
        condition_to_set_short = final_short_entry_condition & (df["enter_long"] == 0)
        if condition_to_set_short.any():
            logger.info(f"DEBUG: SETTING enter_short=1 for Pair: {metadata['pair']}, Candle: {df.loc[condition_to_set_short].index.values[0]}")
        df.loc[condition_to_set_short, ["enter_short", "enter_tag"]] = (1, "short")
        
        # Apply standard long entries
        df.loc[final_long_entry_condition, ["enter_long", "enter_tag"]] = (1, "long")
        
        # Apply standard short entries
        # Ensure short entries don't overwrite long entries on the same candle if both somehow trigger
        df.loc[final_short_entry_condition & (df["enter_long"] == 0), ["enter_short", "enter_tag"]] = (1, "short")

        # Prepare fallback signal conditions (vectorized)
        high_confidence = processed_df["pred_confidence"] > self.high_confidence_threshold.value
        
        fallback_long_condition_met = (
            (processed_df["do_predict"] == 1) & 
            (processed_df["&-s_target"] > 0.01 * processed_df["atr_scaled"]) & 
            high_confidence &
            crossed_above(processed_df["rolling_trend_scaled"], processed_df["rolling_trend_threshold"])
        )
        
        fallback_short_condition_met = (
            (processed_df["do_predict"] == 1) & 
            (processed_df["&-s_target"] < -0.01 * processed_df["atr_scaled"]) & 
            high_confidence &
            crossed_below(processed_df["rolling_trend_scaled"], processed_df["rolling_trend_threshold"])
        )

        # Apply fallback long entries
        # Only apply if no standard long or short entry was set for this candle
        df.loc[
            fallback_long_condition_met &
            (df["enter_long"] == 0) &
            (df["enter_short"] == 0), # Ensures no standard short entry took precedence
            ["enter_long", "enter_tag"]
        ] = (1, "long_fallback")

        # Apply fallback short entries
        # Only apply if no standard long or short entry, and no fallback long entry was set for this candle
        df.loc[
            fallback_short_condition_met &
            (df["enter_long"] == 0) & # Ensures no standard long or fallback long took precedence
            (df["enter_short"] == 0),
            ["enter_short", "enter_tag"]
        ] = (1, "short_fallback")
        
        return df
    
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines the exit signal for both long and short trades.
        Uses entry columns directly for opposite signal exits.
        """
        # Initialize exit columns
        df['exit_long'] = False # Initialize to False, will be set to True if any exit condition met
        df['exit_short'] = False # Initialize to False
        df['exit_tag'] = None
        
        # Get indicators with current hyperopt parameters
        processed_df = self.calculate_indicators(df) # Assuming this df is the one from populate_indicators
        
        # Define base exit condition for other exit types
        base_exit_condition = (
            (processed_df['pred_confidence'] > processed_df["confidence_threshold"]) & 
            (processed_df["do_predict"] == 1)
        )
        
        if self.use_vol_filter.value: # Ensure this hyperopt param is defined if used
            base_exit_condition &= processed_df['vol_rank'] > self.vol_rank_threshold.value
        
        # --- Specific Exit Reasons (These run first) ---

        # 1. Target Reversal Exit 
        if self.use_target_exit_filter.value:
            exit_long_threshold = processed_df["&-s_target_mean"] * float(self.exit_long_threshold_multiplier.value)
            exit_short_threshold = processed_df["&-s_target_mean"] * float(self.exit_short_threshold_multiplier.value)
            
            # Long exit - prediction crosses below exit threshold
            df.loc[
                (df['exit_long'] == False) & # Only if not already exited by a prior rule in this function
                crossed_below(processed_df['&-s_target'], exit_long_threshold) &
                base_exit_condition,
                ['exit_long', 'exit_tag']
            ] = (True, 'target_exit_long') # More specific tag
            
            # Short exit - prediction crosses above exit threshold
            df.loc[
                (df['exit_short'] == False) & # Only if not already exited
                crossed_above(processed_df['&-s_target'], exit_short_threshold) &
                base_exit_condition,
                ['exit_short', 'exit_tag']
            ] = (True, 'target_exit_short') # More specific tag
    
        # 2. Trend Reversal Exit (Secondary - if target exit not triggered)
        if self.use_trend_exit_filter.value:
            df.loc[
                (df['exit_long'] == False) & # Only if not already exited
                crossed_below(processed_df['rolling_trend_scaled'], processed_df["rolling_trend_threshold"]) &
                base_exit_condition,
                ['exit_long', 'exit_tag']
            ] = (True, 'trend_exit_long') # More specific tag
            
            df.loc[
                (df['exit_short'] == False) & # Only if not already exited
                crossed_above(processed_df['rolling_trend_scaled'], processed_df["rolling_trend_threshold"]) &
                base_exit_condition,
                ['exit_short', 'exit_tag']
            ] = (True, 'trend_exit_short') # More specific tag
    
        # 3. Timed Exit (Tertiary - if other exits not triggered)
        if self.use_timed_exit.value:
            df.loc[
                (df['exit_long'] == False) & # Only if not already exited
                (df['trade_duration'] > self.timed_exit_long_threshold.value) &
                (processed_df['pred_confidence'] < processed_df['confidence_threshold']), # Example condition
                ['exit_long', 'exit_tag']
            ] = (True, 'timed_exit_long') # More specific tag
            
            df.loc[
                (df['exit_short'] == False) & # Only if not already exited
                (df['trade_duration'] > self.timed_exit_short_threshold.value) &
                (processed_df['pred_confidence'] < processed_df['confidence_threshold']), # Example condition
                ['exit_short', 'exit_tag']
            ] = (True, 'timed_exit_short') # More specific tag

        # --- Opposite Signal Exits (Applied last to ensure they can trigger a flip) ---
        # These conditions will set exit_long/short to True if an opposite entry was signaled,
        # potentially overriding a previous exit_tag if another same-direction exit condition was also met.
        # This is usually desired for ensuring flips.
        
        # Exit long positions if a short entry signal is present for the same candle
        # Note: We use `df['enter_short']` which was populated by `populate_entry_trend`
        # --- Opposite Signal Exits (Applied last to ensure they can trigger a flip) ---
        if (df['enter_short'] == 1).any():
            logger.info(f"DEBUG_EXIT: Pair: {metadata['pair']}, Candle: {df.loc[df['enter_short'] == 1].index.values[0]}, enter_short IS 1. Applying exit_long.")
            
        df.loc[
            df['enter_short'] == 1,  # If populate_entry_trend signaled a short entry
            ['exit_long', 'exit_tag'] # Then signal an exit for any open long position
        ] = (True, 'opposite_signal_short')
        
        # Exit short positions if a long entry signal is present for the same candle
        df.loc[
            df['enter_long'] == 1,   # If populate_entry_trend signaled a long entry
            ['exit_short', 'exit_tag'] # Then signal an exit for any open short position
        ] = (True, 'opposite_signal_long')
    
        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Custom stoploss handling both backtesting and live trading accurately.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return -1  # Keep existing stoploss
    
        # Ensure trade open rate is valid
        if not trade or not trade.open_rate:
            return -1  # Keep existing stoploss
    
        # --- Get Data & Parameters ---
        last_candle = dataframe.iloc[-1]
        
        # Check if ATR is available, calculate if not
        if 'atr' not in last_candle or last_candle.get('atr', 0) <= 0:
            # Calculate ATR directly here if not available
            volatility_window = int(self.vol_window.value)
            atr_series = ta.ATR(dataframe, timeperiod=volatility_window)
            if atr_series is not None and not atr_series.empty:
                atr = atr_series.iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    # If still not available, use a basic alternative
                    high_low = dataframe['high'] - dataframe['low']
                    close_prev_close = abs(dataframe['close'] - dataframe['close'].shift(1))
                    atr = max(high_low.iloc[-14:].mean(), close_prev_close.iloc[-14:].mean())
            else:
                # Last resort fallback - use a percentage of current price
                atr = current_rate * 0.01  # Use 1% as a fallback
        else:
            atr = last_candle.get('atr', 0)
        
        # Ensure ATR is always positive
        atr = max(atr, current_rate * 0.005)  # At least 0.5% of price
    
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        prediction_confidence = last_candle.get("pred_confidence", 0.5)
    
        # --- IMPROVED: Calculate trade duration in candles ---
        tf_minutes = timeframe_to_minutes(self.timeframe)
        
        # Method 1: Time-based calculation - works for both live and backtest
        elapsed_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        trade_duration_candles_time = elapsed_minutes / tf_minutes
        
        # Method 2: Dataframe-based calculation - more accurate for backtest
        trade_duration_candles_df = 0  # Default value
        
        if 'date' in dataframe.columns:
            try:
                # Find the candle where the trade was opened
                trade_open_date = trade.open_date_utc
                
                # Convert index to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(dataframe.index):
                    df_dates = pd.to_datetime(dataframe['date'])
                else:
                    df_dates = dataframe.index
                    
                # Find candles after trade open date
                candles_after_open = df_dates >= trade_open_date
                
                if candles_after_open.any():
                    # Get index of first candle at or after trade open
                    first_candle_idx = candles_after_open.argmax() 
                    last_candle_idx = len(dataframe) - 1
                    
                    # Calculate candles difference (+1 to include the current candle)
                    trade_duration_candles_df = last_candle_idx - first_candle_idx + 1
                    
                    # If we're in live trading, we haven't completed the current candle yet,
                    # so adjust for partial candle progression
                    if self.dp.runmode.value != 'backtest':
                        current_candle_date = df_dates.iloc[-1]
                        # Calculate how far we are into the current candle (0.0 to 1.0)
                        if current_time > current_candle_date:
                            partial_candle_progress = min(1.0, (current_time - current_candle_date).total_seconds() / 
                                                        (tf_minutes * 60))
                        else:
                            partial_candle_progress = 0.0
                        
                        # Subtract the incomplete portion of the current candle
                        trade_duration_candles_df -= (1.0 - partial_candle_progress)
            except Exception:
                pass
        
        # Choose the better method based on running mode
        if self.dp.runmode.value == 'backtest' and trade_duration_candles_df > 0:
            trade_duration_candles = trade_duration_candles_df  # Use dataframe method in backtest
        else:
            trade_duration_candles = trade_duration_candles_time  # Use time method in live trading
        
        # Get hyperopt parameters
        soft_stoploss_pct = self.soft_stoploss_pct.value
        initial_duration = self.initial_stop_duration_candles.value
        min_profit_for_trailing = self.min_profit_for_trailing.value
        atr_multiplier = self.atr_stoploss_multiplier.value
        historical_volatility_factor = self.historical_volatility_factor.value
        prediction_confidence_factor = self.prediction_confidence_factor.value
        max_loss_floor_val = self.max_loss_floor.value
        max_loss_ceiling_val = self.max_loss_ceiling.value
        max_loss_vol_multiplier_val = self.max_loss_vol_multiplier.value
    
        # --- Initial Stoploss Phase ---
        if trade_duration_candles < initial_duration:
            return soft_stoploss_pct
    
        # --- Soft Stoploss Phase (After Initial Duration, Before Profit Target) ---
        if current_profit < min_profit_for_trailing:
            return soft_stoploss_pct
    
        # --- Main Stoploss Calculation ---
        dynamic_volatility_factor = 1 + historical_volatility * historical_volatility_factor
        confidence_factor = 1 - (prediction_confidence * prediction_confidence_factor)
        stoploss_buffer_abs = atr * atr_multiplier * dynamic_volatility_factor * confidence_factor
    
        # Convert absolute buffer to percentage relative to OPEN rate
        stoploss_pct_calculated = stoploss_buffer_abs / trade.open_rate  # Positive value
    
        # Calculate max loss percentage
        max_loss_abs_pct = min(max_loss_floor_val + historical_volatility * max_loss_vol_multiplier_val, max_loss_ceiling_val)
    
        # Determine final percentage
        final_stoploss_pct = max(-stoploss_pct_calculated, -max_loss_abs_pct)  # Result is negative
    
        # Ensure the calculated percentage is negative
        if final_stoploss_pct >= 0:
            stoploss_pct_to_use = soft_stoploss_pct
        else:
            stoploss_pct_to_use = final_stoploss_pct
    
        return stoploss_pct_to_use

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float,
                            min_stake: float | None, max_stake: float, leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            logger.warning(f"Dataframe unavailable for {pair}. Using proposed stake: {proposed_stake}")
            return proposed_stake

        last_candle = dataframe.iloc[-1]
        prediction_confidence = last_candle.get("pred_confidence", 0.5)
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
        prediction_confidence = last_candle.get("pred_confidence", 0.5)

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
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        """
        Called before placing an order.
        In live/dry mode, if an opposite position exists, we exit it immediately.
        In backtesting, this check is bypassed as populate_exit_trend handles opposite exits.
        """
        # Access runmode via self.dp (data provider)
        runmode = getattr(self.dp, 'runmode', None)

        # Optional: Add this log to see what runmode is being detected
        # logger.info(f"CONFIRM_TRADE_ENTRY: Pair: {pair}, Side: {side}, Detected Runmode: {runmode}")

        if runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            # Live/Dry-run: Check for open trades and handle opposite positions
            # This is the block that should only run in live/dry.
            open_trades = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True)])
            
            if open_trades:
                for trade_obj in open_trades: # Renamed to avoid conflict
                    if (side == "long" and trade_obj.is_short) or \
                       (side == "short" and not trade_obj.is_short):
                        # Opposite position exists - force exit first
                        logger.info(
                            f"CONFIRM_ENTRY (LIVE/DRY): Forcing exit of {pair} "
                            f"{'short' if trade_obj.is_short else 'long'} position "
                            f"before entering {side} position."
                        )
                        
                        if hasattr(self.freqtrade, 'execute_trade_exit'):
                            self.freqtrade.execute_trade_exit(
                                trade=trade_obj, 
                                limit=rate, 
                                exit_check=True, 
                                exit_tag=f"confirm_exit_opposite_{side}"
                            )
                            logger.info(f"CONFIRM_ENTRY (LIVE/DRY): Opposite trade {trade_obj.id} for {pair} exited.")
                        else:
                            logger.warning("CONFIRM_ENTRY (LIVE/DRY): self.freqtrade.execute_trade_exit not available.")
                        
                        return True # Allow new entry after attempting exit
            
            return True # No opposite trade, or exit handled. Allow entry.

        elif runmode == RunMode.BACKTEST:
            # Backtesting: The logic for exiting opposite trades is handled by
            # populate_exit_trend. We simply allow the entry signal here.
            return True
            
        else:
            # Other modes (e.g., HYPEROPT, PLOT) or if runmode is None
            # Default to allowing the trade.
            return True

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
        dataframe["pred_correct"] = np.nan # Initialize with NaN
        valid_labels_mask = dataframe[label_col].notna()
        if valid_labels_mask.sum() > 0:
            dataframe.loc[valid_labels_mask, "pred_correct"] = np.where(
                np.sign(dataframe.loc[valid_labels_mask, label_col]) == np.sign(dataframe.loc[valid_labels_mask, prediction_col]),
                1, 0
            )

        # ✅ Step 2: Rolling Accuracy (Last 50 candles) - Needed for refined confidence
        # Calculate rolling mean of correctness. Fill initial NaNs with 0.5 (neutral assumption)
        rolling_accuracy_window = 50 # Or use a hyperparameter
        dataframe["rolling_accuracy"] = dataframe["pred_correct"].rolling(
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
            dataframe["pred_confidence"] = (base_confidence * dataframe["rolling_accuracy"]).clip(0, 1).fillna(0) # MODIFIED LINE

            # 4c. Confidence score is only counted for correct predictions (Using Refined Confidence)
            # Use np.nan_to_num to handle potential NaNs from prediction_correct
            confidence_correct_array = np.where(
                dataframe["pred_correct"] == 1, dataframe["pred_confidence"], 0
            )
            dataframe["confidence_correct"] = np.nan_to_num(confidence_correct_array, nan=0.0) # Uses refined confidence

            # 4d. Normalize avg confidence over correct predictions - Keep Existing Calculation
            correct_preds = dataframe["pred_correct"].rolling(100, min_periods=1).sum()
            # Ensure confidence_correct exists before rolling on it
            if "confidence_correct" in dataframe.columns:
                 dataframe["avg_confidence_correct"] = dataframe["confidence_correct"].rolling(100, min_periods=1).sum() / (correct_preds + 1e-6)
                 dataframe["avg_confidence_correct"] = dataframe["avg_confidence_correct"].fillna(0) # Fill potential NaNs from division
            else:
                 dataframe["avg_confidence_correct"] = np.nan

        else: # Keep Existing Warning
            logger.warning(f"⚠️ Column '{std_col}' not found. Skipping confidence tracking.")
            dataframe["pred_confidence"] = 0.0 # Ensure column exists if skipped
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
        if not hasattr(self, 'pred_metrics_storage'):
            self.pred_metrics_storage = []
        self.pred_metrics_storage.append(metrics)

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
    def save_prediction_metrics(self, filename="pred_metrics.csv"):
        """
        Saves the accumulated prediction metrics to a CSV file after backtesting.
        """
        # Use class attribute directly as it's defined at class level
        if not LSTMStrategy_v43.pred_metrics_storage:
            logger.warning("⚠️ No prediction metrics found to save.")
            return

        try:
            # Use class attribute directly
            df = pd.DataFrame(LSTMStrategy_v43.pred_metrics_storage)
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