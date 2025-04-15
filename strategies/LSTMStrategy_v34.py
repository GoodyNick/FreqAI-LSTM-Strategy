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
                "rolling_accuracy": {"color": "purple", "plot_type": "scatter"},  # Plot rolling accuracy
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

    # ✅ Entry hyperopt parameters
    dynamic_long_threshold_multiplier = RealParameter(0.5, 1.2, default=1.0, space="buy", load=True, optimize=True)
    dynamic_short_threshold_multiplier = RealParameter(0.5, 1.2, default=1.0, space="buy", load=True, optimize=True)
    confidence_threshold_multiplier = RealParameter(0.1, 1.0, default=1.0, space="buy", load=True, optimize=True)
    high_confidence_threshold = RealParameter(0.7, 0.95, default=0.85, space="buy", load=True, optimize=True)
    use_confidence_filter_entry = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    use_trend_filter = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    rolling_trend_threshold_multiplier = RealParameter(0.2, 2.0, default=1.0, space="buy", load=True, optimize=True)
    vol_rank_threshold = RealParameter(0.05, 0.5, default=0.25, space="buy", load=True, optimize=True)

    # Exit trend hyperopt parameters
    dynamic_long_exit_threshold_multiplier = RealParameter(0.5, 1.5, default=1.0, space="sell", load=True, optimize=True)
    dynamic_short_exit_threshold_multiplier = RealParameter(0.5, 1.5, default=1.0, space="sell", load=True, optimize=True)
    timed_exit_long_threshold = IntParameter(10, 30, default=20, space="sell", load=True, optimize=True)
    timed_exit_short_threshold = IntParameter(10, 30, default=20, space="sell", load=True, optimize=True)
    use_target_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    use_trend_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    use_timed_exit = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)

    # ✅ Stoploss Hyperopt Parameters
    soft_stoploss_pct = RealParameter(-0.25, -0.03, default=-0.05, space="sell", load=True, optimize=True)
    min_profit_for_trailing = RealParameter(0.001, 0.05, default=0.005, space="sell", load=True, optimize=True)
    atr_stoploss_multiplier = RealParameter(1.0, 10.0, default=3.0, space="sell", load=True, optimize=True)
    historical_volatility_factor = RealParameter(0.3, 1.2, default=0.5, space="sell", load=True, optimize=True) # Used in ATR-based calculation scaling
    prediction_confidence_factor = RealParameter(0.2, 1.0, default=0.6, space="sell", load=True, optimize=True) # Used in ATR-based calculation scaling
    max_loss_floor = RealParameter(0.01, 0.05, default=0.03, space="sell", load=True, optimize=True) # Min % for max loss cap
    max_loss_ceiling = RealParameter(0.05, 0.15, default=0.08, space="sell", load=True, optimize=True) # Max % for max loss cap
    max_loss_vol_multiplier = RealParameter(0.5, 5.0, default=2.0, space="sell", load=True, optimize=True) # Volatility sensitivity for max loss cap
    initial_stop_duration_candles = IntParameter(1, 10, default=3, space="sell", load=True, optimize=True) # Duration in candles for initial stop

    # ✅ Stake amount hyperopt parameters
    stake_scaling_factor = RealParameter(0.4, 2.0, default=1.0, space="buy", load=True, optimize=True)

    # ✅ Leverage Hyperopt Parameters
    leverage_range_start = 1
    leverage_range_end = 5
    volatility_influence = RealParameter(0.0, 0.5, default=0.2, space="buy", load=True, optimize=True)
    confidence_influence = RealParameter(0.0, 0.5, default=0.2, space="buy", load=True, optimize=True)    

    # # Buy hyperspace params:
    # buy_params = {
    #     "base_risk": 0.05071,
    #     "confidence_influence": 0.04532,
    #     "confidence_threshold_multiplier": 0.43721,
    #     "dynamic_long_threshold_multiplier": 0.65036,
    #     "dynamic_short_threshold_multiplier": 0.84277,
    #     "high_confidence_threshold": 0.89692,
    #     "rolling_trend_threshold_multiplier": 0.79379,
    #     "stake_scaling_factor": 0.8746,
    #     "use_confidence_filter_entry": True,
    #     "use_trend_filter": True,
    #     "vol_rank_threshold": 0.47073,
    #     "volatility_influence": 0.28005,
    # }

    # # Sell hyperspace params:
    # sell_params = {
    #     "atr_stoploss_multiplier": 1.02987,
    #     "dynamic_long_exit_threshold_multiplier": 0.97523,
    #     "dynamic_short_exit_threshold_multiplier": 0.79514,
    #     "historical_volatility_factor": 0.67703,
    #     "max_loss_ceiling": 0.06973,
    #     "max_loss_floor": 0.02332,
    #     "max_loss_vol_multiplier": 2.13439,
    #     "min_profit_for_trailing": 0.0416,
    #     "prediction_confidence_factor": 0.86341,
    #     "soft_stoploss_pct": -0.19217,
    #     "timed_exit_long_threshold": 29,
    #     "timed_exit_short_threshold": 26,
    #     "use_target_exit_filter": True,
    #     "use_timed_exit": True,
    #     "use_trend_exit_filter": True,
    # }

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
        """
    
        self.freqai_info = self.config["freqai"]
        # logger.info(f"[DEBUG] Entered populate_indicators for {metadata['pair']}")
    
        # ✅ 1. ATR (Volatility)
        # - Timeperiod: Dynamically calculated based on recent volatility.
        # - Scaling: Normalized by close price.
        volatility_window = 50  # Window to assess recent volatility
        atr_timeperiod = int(10 + 20 * (df["close"].pct_change().rolling(volatility_window).std().iloc[-1] / 0.05))  # Dynamic timeperiod
        atr_timeperiod = max(5, min(atr_timeperiod, 30))  # Clip to reasonable range
        df["atr"] = ta.ATR(df, timeperiod=atr_timeperiod).bfill()
        df["atr_normalized"] = df["atr"] / df["close"]  # Normalize by close price
    
        # ✅ 2. Volume Rank (Volume)
        # - Window: Dynamically adjusted based on ATR.
        atr_mean = df["atr"].rolling(24).mean().iloc[-1]  # Average ATR over 24 periods
        vol_rank_window = int(12 + 24 * (atr_mean / (df["close"].iloc[-1] * 0.05)))  # Dynamic window
        vol_rank_window = max(6, min(vol_rank_window, 48))  # Clip to reasonable range
        df["vol_rank"] = df["volume"].rolling(vol_rank_window).rank(pct=True).fillna(0)
    
        # ✅ 3. Rolling Trend (Trend)
        # - Pct Change: Dynamically adjusted based on ATR.
        # - Rolling Window: Dynamically adjusted based on ATR.
        atr_scaling_factor = atr_mean / (df["close"].iloc[-1] * 0.05) # ✅ Combine ATR scaling factor
        pct_change_period = int(12 + 24 * atr_scaling_factor)  # Dynamic period
        pct_change_period = max(6, min(pct_change_period, 48))  # Clip to reasonable range
        rolling_window = int(3 + 6 * atr_scaling_factor)  # Dynamic window
        rolling_window = max(2, min(rolling_window, 12))  # Clip to reasonable range
        df["rolling_trend"] = df["close"].pct_change(pct_change_period).rolling(rolling_window).mean().fillna(0)
    
        # ✅ 4. ATR Scaling (Volatility Scaling)
        # - ATR Window: Dynamically adjusted based on recent volatility.
        atr_scaling_window = int(24 + 48 * (df["close"].pct_change().rolling(volatility_window).std().iloc[-1] / 0.05))  # Dynamic window
        atr_scaling_window = max(12, min(atr_scaling_window, 72))  # Clip to reasonable range
        atr_quantile = 0.90  # Keep quantile as a hyperopt parameter
        df["atr_scaled"] = df["atr_normalized"] / df["atr_normalized"].rolling(atr_scaling_window).quantile(atr_quantile)
    
        # ✅ 5. MinMax Scaling ATR Scaled
        atr_scaled_min = df["atr_scaled"].min()
        atr_scaled_max = df["atr_scaled"].max()
        df["atr_scaled"] = (df["atr_scaled"] - atr_scaled_min) / (atr_scaled_max - atr_scaled_min)
    
        # ✅ 6. Improved Rolling Trend Scaling: Adaptive Window + Normalization
        trend_window = 24
        adaptive_window = int(min(trend_window, max(20, df["atr"].rolling(50).mean().iloc[-1] * 10)))
        mean_trend = df["rolling_trend"].rolling(adaptive_window).mean()
        std_trend = df["rolling_trend"].rolling(adaptive_window).std()
        df["rolling_trend_scaled"] = (df["rolling_trend"] - mean_trend) / (std_trend + 1e-6)
    
        # ✅ Call FreqAI for ML Predictions
        df = self.freqai.start(df, metadata, self)
        logger.info(f"[DEBUG] FreqAI finished start() call for {metadata['pair']}")
    
        # ✅ Store Threshold Base Values (No Hyperopt Parameters Here, Same as Original)
        df["dynamic_long_threshold_base"] = df["&-s_target_mean"] + df["&-s_target_std"] * df["atr_scaled"]
        df["dynamic_short_threshold_base"] = df["&-s_target_mean"] - df["&-s_target_std"] * df["atr_scaled"]
        df["rolling_trend_threshold_base"] = df["rolling_trend_scaled"].rolling(100, min_periods=10).median()
        df["dynamic_exit_threshold_base"] = (
            df["&-s_target"].ewm(span=50).mean() +
            df["atr_scaled"] * df["&-s_target_std"] * (0.6 + df["vol_rank"] * 0.3)
        )
        df["exit_trend_threshold_base"] = df["rolling_trend_scaled"].rolling(50).median()
    
        # ✅ Keeping "T" for Plotting Purposes Only (Not Used in Trade Logic)
        df["T"] = self.create_target_T(df)
        df["Prediction"] = df["&-s_target"]
        df["Avg Prediction"] = df["&-s_target_mean"]
        df["True Label"] = df["T"]
        # indicators calculated to simply use in trade logic
        df["long_threshold"] = df["dynamic_long_threshold_base"] * self.dynamic_long_threshold_multiplier.value * df["atr_scaled"]
        df["short_threshold"] = df["dynamic_short_threshold_base"] * self.dynamic_short_threshold_multiplier.value * df["atr_scaled"]
        df["long_exit_threshold"] = df['dynamic_exit_threshold_base'] * self.dynamic_long_exit_threshold_multiplier.value * df["atr_scaled"]
        df["short_exit_threshold"] = df['dynamic_exit_threshold_base'] * self.dynamic_short_exit_threshold_multiplier.value * df["atr_scaled"]
        df["rolling_trend_threshold"] = df["rolling_trend_threshold_base"] * self.rolling_trend_threshold_multiplier.value

        # ✅ Compute & Save Prediction Metrics (Same as Original)
        if hasattr(self, "dp") and hasattr(self.dp, "runmode"):
            logger.info(f"📦 runmode: {self.dp.runmode.value}")
        else:
            logger.info("📦 runmode: unavailable (likely during training phase)")
    
        # only enable prediction metrics logs at training mode
        log_metrics = (
            not hasattr(self, "dp")
            or getattr(self.dp, "runmode", None) in [RunMode.BACKTEST, RunMode.HYPEROPT, RunMode.PLOT]
        )        
        # ✅ Compute & Save Prediction Metrics (Same as Original)
        self.compute_prediction_metrics(df, metadata, log_metrics=log_metrics)
        self.save_prediction_metrics()
    
        df["confidence_threshold_base"] = df["prediction_confidence"].rolling(100).quantile(0.5).bfill()
        df["confidence_threshold"] = df["confidence_threshold_base"] * self.confidence_threshold_multiplier.value
    
        # ✅ Calculate trade duration (in candles)
        if metadata['pair'] not in self.trades:
            df['trade_duration'] = 0
        else:
            open_since = self.trades[metadata['pair']]['open_since']
            df['trade_duration'] = df.index - open_since
            df['trade_duration'] = df['trade_duration'].apply(lambda x: x.total_seconds() / 60 / self.timeframe_minutes)
    
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines the entry signal for both long and short trades using crossed_above/below.
        Leverages indicators calculated in populate_indicators().
        """
    
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None  # Initialize enter_tag
    
        # ✅ Combined Entry Condition (Direction-Aware)
        def combined_entry_condition(df, side: str):
            condition = (df["do_predict"] == 1 & (df["vol_rank"] > self.vol_rank_threshold.value))
            if self.use_confidence_filter_entry.value:
                condition &= (df["prediction_confidence"] > df["confidence_threshold"])
            if self.use_trend_filter.value:
                if side == "long":
                    condition &= (df["rolling_trend_scaled"] > df["rolling_trend_threshold"])
                elif side == "short":
                    condition &= (df["rolling_trend_scaled"] < df["rolling_trend_threshold"])
            return condition
    
        # ─── Long Entry Logic ───────────────────────────
        # Apply crossed_above for dynamic_long_threshold
        df.loc[
            crossed_above(df["&-s_target"], df["long_threshold"]) &
            combined_entry_condition(df, "long"),
            ["enter_long", "enter_tag"]
        ] = (1, "long")
    
        # ─── Short Entry Logic ──────────────────────────
        # Apply crossed_below for dynamic_short_threshold
        df.loc[
            crossed_below(df["&-s_target"], df["short_threshold"]) &
             combined_entry_condition(df, "short"),
            ["enter_short", "enter_tag"]
        ] = (1, "short")
    
        # ─── Fallback Entry ─────────────────────────────
        high_confidence = df["prediction_confidence"] > self.high_confidence_threshold.value
    
        # ✅ Make fallback entries more selective
        df.loc[
            (df["do_predict"] == 1) & (df["&-s_target"] > 0.01 * df["atr_scaled"]) & high_confidence & (df["enter_long"] == 0) &
            crossed_above(df["rolling_trend_scaled"], df["rolling_trend_threshold"]),
            ["enter_long", "enter_tag"]
        ] = (1, "long_fallback")
    
        df.loc[
            (df["do_predict"] == 1) & (df["&-s_target"] < -0.01 * df["atr_scaled"]) & high_confidence & (df["enter_short"] == 0) & 
            crossed_below(df["rolling_trend_scaled"], df["rolling_trend_threshold"]),
            ["enter_short", "enter_tag"]
        ] = (1, "short_fallback")
    
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
                (df['&-s_target'] < df["long_exit_threshold"]) &
                combined_exit_condition(df),
                ['exit_long', 'exit_tag']
            ] = (True, 'exit_long')
            df.loc[
                (df['&-s_target'] > df["short_exit_threshold"]) &
                combined_exit_condition(df),
                ['exit_short', 'exit_tag']
            ] = (True, 'exit_short')

        # 2. Trend Reversal Exit (Secondary - if target exit not triggered)
        if self.use_trend_exit_filter.value:
            df.loc[
                (df['rolling_trend_scaled'] < df["rolling_trend_threshold"]) &
                (df['exit_long'] == False) & combined_exit_condition(df),  # Only if target exit not triggered
                ['exit_long', 'exit_tag']
            ] = (True, 'exit_long_trend')
            df.loc[
                (df['rolling_trend_scaled'] > df["rolling_trend_threshold"]) &
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
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            # logger.warning(f"Dataframe unavailable for {pair}. Keeping existing stoploss.")
            return -1 # Keep existing stoploss

        # Ensure trade open rate is valid
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
        soft_stoploss_pct = self.soft_stoploss_pct.value
        initial_duration = self.initial_stop_duration_candles.value
        min_profit_for_trailing = self.min_profit_for_trailing.value # Profit threshold
        atr_multiplier = self.atr_stoploss_multiplier.value
        historical_volatility_factor = self.historical_volatility_factor.value
        prediction_confidence_factor = self.prediction_confidence_factor.value
        max_loss_floor_val = self.max_loss_floor.value # Positive value (e.g., 0.03 for 3%)
        max_loss_ceiling_val = self.max_loss_ceiling.value # Positive value (e.g., 0.08 for 8%)
        max_loss_vol_multiplier_val = self.max_loss_vol_multiplier.value

        # --- Determine Stoploss Percentage ---
        stoploss_pct_to_use: float

        # --- Initial Stoploss Phase ---
        if trade_duration_candles < initial_duration:
            stoploss_pct_to_use = soft_stoploss_pct
            # logger.info(f"[Stoploss Initial] Pair: {pair} | Duration Candles: {trade_duration_candles:.1f} < {initial_duration} | Using Soft Stop Pct: {stoploss_pct_to_use:.4f}")

        # --- Soft Stoploss Phase (After Initial Duration, Before Profit Target) ---
        elif current_profit < min_profit_for_trailing:
            stoploss_pct_to_use = soft_stoploss_pct
            # logger.info(f"[Stoploss Soft] Pair: {pair} | Profit: {current_profit:.4f} < {min_profit_for_trailing:.4f} | Using Soft Stop Pct: {stoploss_pct_to_use:.4f}")

        # --- Main Stoploss Calculation (After Initial Duration & Profit Target Met) ---
        else:
            dynamic_volatility_factor = 1 + historical_volatility * historical_volatility_factor
            confidence_factor = 1 - (prediction_confidence * prediction_confidence_factor) # Lower buffer for high confidence
            stoploss_buffer_abs = atr * atr_multiplier * dynamic_volatility_factor * confidence_factor
            stoploss_pct_calculated = stoploss_buffer_abs / trade.open_rate # Positive value
            # Ensure max loss values are positive for comparison
            max_loss_abs_pct = min(max_loss_floor_val + historical_volatility * max_loss_vol_multiplier_val, max_loss_ceiling_val) # Positive value

            # final_stoploss_pct is the LARGER loss (closer to zero) between calculated and max_loss
            # We need the negative percentage, so negate the positive calculated values
            final_stoploss_pct = max(-stoploss_pct_calculated, -max_loss_abs_pct) # Result is negative

            # Ensure the calculated percentage is actually negative
            if final_stoploss_pct >= 0:
                # logger.info(f"[Stoploss Error] Pair: {pair} | Calculated positive stoploss {final_stoploss_pct:.4f}. Using soft stoploss {soft_stoploss_pct:.4f} instead.")
                stoploss_pct_to_use = soft_stoploss_pct # Fallback to soft stoploss percentage
            else:
                stoploss_pct_to_use = final_stoploss_pct
                # logger.info(
                #     f"[Stoploss Main] Pair: {pair} | DVF: {dynamic_volatility_factor:.4f} | CF: {confidence_factor:.4f} | "
                #     f"Buffer Abs: {stoploss_buffer_abs:.4f} | Calc SL %: {-stoploss_pct_calculated:.4f} | "
                #     f"Max Loss %: {-max_loss_abs_pct:.4f} | Final SL Pct: {stoploss_pct_to_use:.4f} | "
                #     f"Profit: {current_profit:.2f} | Duration Candles: {trade_duration_candles:.1f}"
                # )

        # --- Convert Percentage to Absolute Price ---
        # Ensure the percentage is negative before applying
        if stoploss_pct_to_use >= 0:
                logger.warning(f"Stoploss percentage {stoploss_pct_to_use:.4f} is not negative for {pair}. Using soft stoploss percentage {soft_stoploss_pct:.4f}.")
                stoploss_pct_to_use = soft_stoploss_pct

        # Calculate the absolute stop price based on the open rate
        stop_price = trade.open_rate * (1 + stoploss_pct_to_use)

        # logger.info(f"[Stoploss Final] Pair: {pair} | Using SL Pct: {stoploss_pct_to_use:.4f} | Open Rate: {trade.open_rate:.4f} | Stop Price: {stop_price:.4f}")

        # Return the absolute stop price
        return stop_price

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
        min_leverage = self.leverage_range_start
        strategy_max_leverage = self.leverage_range_end

        # ✅ Calculate the base leverage (midpoint of the range)
        base_leverage = (min_leverage + max_leverage) / 2

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
        Saves the results to a CSV file after backtesting.
        """
        prediction_mean = prediction_col + "_mean"
        prediction_std = prediction_col + "_std"

        if log_metrics:
            logger.info(f"🔍 {label_col} mean: {dataframe[label_col].mean()}, min: {dataframe[label_col].min()}, max: {dataframe[label_col].max()}")
            logger.info(f"🔍 {prediction_col} mean: {dataframe[prediction_col].mean()}, min: {dataframe[prediction_col].min()}, max: {dataframe[prediction_col].max()}")
            logger.info(f"🔍 {prediction_mean} mean: {dataframe[prediction_mean].mean()}, min: {dataframe[prediction_mean].min()}, max: {dataframe[prediction_mean].max()}")
            logger.info(f"🔍 {prediction_std} mean: {dataframe[prediction_std].mean()}, min: {dataframe[prediction_std].min()}, max: {dataframe[prediction_std].max()}")

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