import logging
import operator
from functools import reduce
from tracemalloc import Traceback
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
from torch import log_

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


class LSTMStrategy_v44_1h(IStrategy):
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
            "Prediction": {"color": "purple", "plot_type": "line"},      # &-s_target
            "Avg Prediction": {"color": "brown", "plot_type": "line"},   # &-s_target_mean (FreqAI's)
            "custom_pred_mean": {"color": "orange", "plot_type": "line"}, # Custom smoothed prediction
            "long_threshold": {"color": "green", "plot_type": "line"},
            "short_threshold": {"color": "red", "plot_type": "line"},
            "exit_long_threshold": {"color": "lightgreen", "plot_type": "line"}, 
            "exit_short_threshold": {"color": "pink", "plot_type": "line"},      
            },
            "Confidence": {
                "pred_confidence": {"color": "orange", "plot_type": "scatter"},
                "confidence_threshold" : {"color": "brown", "plot_type": "scatter"},
                "do_predict": {"color": "purple", "plot_type": "scatter"},
            },
            "Indicators": {
                "atr_scaled": {"color": "green", "plot_type": "line"},
                "vol_rank": {"color": "blue", "plot_type": "line"},
                "vol_rank_dynamic_threshold": {"color": "lightblue", "plot_type": "line"},
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
    hyperopt_categorical = False

    # use leverage
    use_leverage = False

    # --- FIXED PARAMETERS (optimize=False) ---
    # These are set to sensible defaults and not hyperoptimized to reduce complexity.
    fixed_vol_window = 24 # For 1h timeframe, daily ATR-like context
    fixed_trend_window = 72 # For 1h timeframe, 3-day trend context
    fixed_stake_scaling_factor = 1.0
    fixed_confidence_threshold_multiplier = 1.0 # Base is already dynamic
    fixed_rolling_trend_threshold_multiplier = 1.0 # Base is already dynamic
    fixed_soft_stoploss_pct = -0.15 # Disaster stop
    fixed_min_profit_for_trailing = 0.001 # Minimal profit to enable dynamic stop
    fixed_initial_stop_duration_candles = 3 # Short period for initial stop
    fixed_high_confidence_threshold = 0.90 # For fallback entries, should be rare

    # --- HYPEROPTABLE PARAMETERS (Core Sensitivities) ---

    # ✅ Prediction Smoothing
    prediction_smoothing_window = IntParameter(3, 50, default=10, space="buy", load=True, optimize=True)

    # ✅ Entry Thresholds
    dynamic_long_threshold_multiplier = DecimalParameter(0.5, 2.0, default=1.0, decimals=2, space="buy", load=True, optimize=True)
    dynamic_short_threshold_multiplier = DecimalParameter(0.5, 2.0, default=1.0, decimals=2, space="buy", load=True, optimize=True)
    
    # ✅ Volume Rank Filter (Dynamic Threshold)
    vol_rank_quantile = DecimalParameter(0.1, 0.5, default=0.25, decimals=2, space="buy", load=True, optimize=True) # Replaces vol_rank_threshold

    # ✅ Exit Thresholds
    exit_long_threshold_multiplier = DecimalParameter(0.5, 1.5, default=0.8, decimals=2, space="sell", load=True, optimize=True)
    exit_short_threshold_multiplier = DecimalParameter(0.5, 1.5, default=0.8, decimals=2, space="sell", load=True, optimize=True)        
    
    # ✅ Timed Exit (Combined)
    timed_exit_duration = IntParameter(24, 168, default=72, space="sell", load=True, optimize=True) # e.g., 1 to 7 days on 1h

    # ✅ Stoploss Parameters (Core Dynamic Parts)
    atr_stoploss_multiplier = DecimalParameter(1.0, 8.0, default=3.0, decimals=1, space="sell", load=True, optimize=True)
    historical_volatility_factor = DecimalParameter(0.1, 1.5, default=0.5, decimals=2, space="sell", load=True, optimize=True) 
    prediction_confidence_factor = DecimalParameter(0.1, 1.5, default=0.5, decimals=2, space="sell", load=True, optimize=True) 
    hard_max_stoploss_pct = DecimalParameter(-0.25, -0.05, default=-0.15, decimals=2, space="sell", load=True, optimize=True) # Replaces max_loss_floor/ceiling/vol_multiplier

    # ✅ Stake/Leverage Dynamics (If use_leverage is True or for dynamic stake)
    leverage_range_end = IntParameter(2, 10, default=3, space="buy", load=True, optimize=use_leverage)
    volatility_influence = DecimalParameter(0.0, 1.0, default=0.2, decimals=2, space="buy", load=True, optimize=True) # For stake and leverage
    confidence_influence = DecimalParameter(0.0, 1.0, default=0.2, decimals=2, space="buy", load=True, optimize=True) # For stake and leverage  

    # --- CATEGORICAL PARAMETERS (Controlled by hyperopt_categorical switch) ---
    use_crossed_entry = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    use_vol_filter = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical) # Default True for the new dynamic filter
    use_confidence_filter_entry = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    use_trend_filter = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    use_mean_prediction_for_signal = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical) # Default to True to test new smoothing

    use_target_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    use_trend_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    use_timed_exit = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    use_opposite_signal_exit = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)  

    # Buy hyperspace params:
    buy_params = {
        "confidence_influence": 0.22,
        "dynamic_long_threshold_multiplier": 0.61,
        "dynamic_short_threshold_multiplier": 0.7,
        "prediction_smoothing_window": 3,
        "vol_rank_quantile": 0.19,
        "volatility_influence": 0.49,
        "leverage_range_end": 3,  # value loaded from strategy
        "use_confidence_filter_entry": True,  # value loaded from strategy
        "use_crossed_entry": True,  # value loaded from strategy
        "use_mean_prediction_for_signal": True,  # value loaded from strategy
        "use_trend_filter": True,  # value loaded from strategy
        "use_vol_filter": True,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "atr_stoploss_multiplier": 1.0,
        "exit_long_threshold_multiplier": 1.3,
        "exit_short_threshold_multiplier": 0.85,
        "hard_max_stoploss_pct": -0.14,
        "historical_volatility_factor": 0.26,
        "prediction_confidence_factor": 1.28,
        "timed_exit_duration": 81,
        "use_opposite_signal_exit": True,  # value loaded from strategy
        "use_target_exit_filter": True,  # value loaded from strategy
        "use_timed_exit": True,  # value loaded from strategy
        "use_trend_exit_filter": True,  # value loaded from strategy
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

    
    # Consistent ATR period for feature normalization within feature_engineering_standard
    # This could also be a class attribute or derived from config if needed.
    # Let's align it with the base_window used in create_target_T for context.
    FEATURE_ATR_PERIOD = 72

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, metadata: Dict, **kwargs):
        """
        Features available to all FreqAI components.
        Called multiple times, once for each period in `indicator_periods_candles`.
        Focus: Robust indicators that benefit from multiple timeperiod evaluations.
        """
        # RSI
        dataframe[f'%-rsi-{period}'] = ta.RSI(dataframe, timeperiod=period)

        # Smoothed ROC
        # Raw ROC can be noisy, especially for short periods.
        # Smoothing it provides a more stable momentum signal.
        roc = ta.ROC(dataframe, timeperiod=period)
        dataframe[f'%-roc_smooth-{period}'] = roc.rolling(window=3, min_periods=1).mean()

        # Bollinger Band Width Percentage
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2)
        dataframe[f'%-bb_width_pct-{period}'] = (
            (bollinger['upper'] - bollinger['lower']) / bollinger['mid'] * 100
        )
        
        # Stochastic Oscillator %K (Momentum)
        # Provides a momentum indicator bounded between 0 and 100.
        # Use ta.STOCH from talib.abstract
        stoch = ta.STOCH(dataframe, 
                         fastk_period=period, 
                         slowk_period=3, 
                         slowk_matype=0, # SMA for slowk smoothing
                         slowd_period=3, 
                         slowd_matype=0  # SMA for slowd smoothing
                        )
        dataframe[f'%-stoch_k-{period}'] = stoch['slowk'] # slowk is the %K line
        # dataframe[f'%-stoch_d-{period}'] = stoch['slowd'] # slowd is the %D line

        # Rolling Volume Weighted Average Price (VWAP) deviation (if volume is available)
        if 'volume' in dataframe.columns and 'close' in dataframe.columns and 'high' in dataframe.columns and 'low' in dataframe.columns:
            # Calculate typical price * volume
            tpv = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3 * dataframe['volume']
            # Calculate rolling sum of tpv and rolling sum of volume
            rolling_tpv_sum = tpv.rolling(window=period, min_periods=1).sum()
            rolling_volume_sum = dataframe['volume'].rolling(window=period, min_periods=1).sum()
            # Calculate rolling VWAP
            rolling_vwap = rolling_tpv_sum / (rolling_volume_sum + 1e-9) # Add epsilon to avoid division by zero
            
            dataframe[f'%-vwap_dev-{period}'] = (dataframe['close'] - rolling_vwap) / (rolling_vwap + 1e-9) * 100
        else:
            dataframe[f'%-vwap_dev-{period}'] = 0 # Placeholder if required columns are missing

        # Clean up NaNs at the beginning
        for col in dataframe.columns:
            if col.startswith('%-') and f'-{period}' in col:
                # MODIFIED fillna
                dataframe[col] = dataframe[col].bfill().ffill().fillna(0)
        
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        """
        Minimal set of features.
        Focus: Basic price/volume info, slightly smoothed for stability.
        """
        # Smoothed Percentage Change (less noisy than raw pct_change)
        # MODIFIED: Convert ta.EMA output to Series before pct_change
        ema3_series = pd.Series(ta.EMA(dataframe['close'], timeperiod=3), index=dataframe.index)
        dataframe['%-ema3_pct_change'] = ema3_series.pct_change() * 100

        # Smoothed Volume (if available)
        if 'volume' in dataframe.columns:
            # MODIFIED: Convert ta.EMA output to Series (good practice, though not strictly necessary if not chaining methods)
            dataframe['%-volume_ema3'] = pd.Series(ta.EMA(dataframe['volume'], timeperiod=3), index=dataframe.index)
        else:
            dataframe['%-volume_ema3'] = 0 # Placeholder if volume is not available

        # Raw Price (Close) - already present, but explicitly stating its role
        dataframe['%-raw_price'] = dataframe['close']
        
        # Basic short-term volatility (ATR over a short window)
        # MODIFIED: Convert ta.ATR output to Series (good practice)
        dataframe['%-atr_short'] = pd.Series(ta.ATR(dataframe, timeperiod=5), index=dataframe.index) # Using a small, fixed period

        # Clean up NaNs
        for col in ['%-ema3_pct_change', '%-volume_ema3', '%-atr_short']:
            if col in dataframe.columns:
                # MODIFIED fillna
                dataframe[col] = dataframe[col].bfill().ffill().fillna(0)

        return dataframe

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
        
    # def create_target_T(self, dataframe: pd.DataFrame) -> pd.Series:
    #     """
    #     Creates a target 'T' for FreqAI with:
    #     1. Inverse dynamic lookahead (smoother transition).
    #     2. Future change calculated as a percentage.
    #     3. Target 'T' using linear scaling and clipping (less saturation than tanh).
    #     4. Enhanced smoothing for reduced noise.
    #     """
    #     # --- Tunable Parameters for Target Generation ---
    #     base_window = self.FEATURE_ATR_PERIOD  # Main window for ATR and TS smoothing
    #     min_lookahead = 4
    #     max_lookahead = 24
        
    #     # Volatility thresholds for lookahead scaling (ATR % of price)
    #     low_vol_threshold_pct = 0.5  # Lower smoothed_vol_pct -> max_lookahead
    #     high_vol_threshold_pct = 3.0 # Higher smoothed_vol_pct -> min_lookahead
        
    #     # Smoothing for dynamic lookahead's volatility input
    #     vol_smoothing_window = 10 # Increased from 5 for smoother lookahead changes
        
    #     # Min periods for TS_pct rolling mean (smoother TS_pct)
    #     ts_pct_min_periods = base_window // 2 # Increased from base_window // 3
        
    #     # Parameters for adaptive linear scaling of T_raw (replaces tanh)
    #     # We'll scale T_raw so that its 98th percentile (absolute) maps to this value.
    #     target_scaled_magnitude_at_percentile = 1.5 
    #     # Final hard clipping range for T after scaling.
    #     final_clip_range_abs = 2.0 
        
    #     # Optional: Light final smoothing for T
    #     final_T_smoothing_window = 3 # Set to 0 or 1 to disable

    #     # --- 1. Prepare Data & Calculate ATR ---
    #     df_copy = dataframe.copy()
        
    #     for col in ['close', 'high', 'low']:
    #         if col in df_copy.columns:
    #             # MODIFIED fillna
    #             df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').replace(0, np.nan).bfill().ffill()
    #         else:
    #             logger.warning(f"Column {col} missing in create_target_T for {dataframe['pair'].iloc[0] if 'pair' in dataframe.columns else 'N/A'}.")
    #             if col == 'close' and col not in df_copy.columns: df_copy[col] = 1.0 
    #             elif col not in df_copy.columns: df_copy[col] = df_copy['close'] if 'close' in df_copy.columns else 1.0

    #     # MODIFIED: Convert ta.ATR output to Series and use .bfill().ffill()
    #     atr_series_target = pd.Series(ta.ATR(df_copy, timeperiod=base_window), index=df_copy.index)
    #     df_copy["ATR"] = atr_series_target.bfill().ffill().fillna(1e-9)
        
    #     # --- 2. Calculate Smoother Inverse Dynamic Lookahead ---
    #     # MODIFIED fillna
    #     current_close_safe = df_copy["close"].replace(0, np.nan).bfill().ffill().fillna(1e-9)
    #     vol_estimate_pct = (df_copy["ATR"] / current_close_safe) * 100
    #     # Increased smoothing for smoothed_vol_pct
    #     df_copy["smoothed_vol_pct"] = vol_estimate_pct.rolling(
    #         vol_smoothing_window, min_periods=max(1, vol_smoothing_window // 3)
    #     ).mean().bfill().ffill().fillna((low_vol_threshold_pct + high_vol_threshold_pct) / 2) # MODIFIED fillna
        
    #     inverted_progress = (high_vol_threshold_pct - df_copy["smoothed_vol_pct"]) / \
    #                         (high_vol_threshold_pct - low_vol_threshold_pct + 1e-9)
    #     clipped_inverted_progress = np.clip(inverted_progress, 0, 1)
    #     df_copy["lookahead_dynamic"] = min_lookahead + clipped_inverted_progress * (max_lookahead - min_lookahead)
    #     df_copy["lookahead_dynamic"] = df_copy["lookahead_dynamic"].round().fillna((min_lookahead + max_lookahead) // 2).astype(int)

    #     # --- 3. Calculate Future Percentage Change ---
    #     future_change_pct_list = []
    #     close_series = df_copy["close"].to_numpy()
    #     lookahead_series = df_copy["lookahead_dynamic"].to_numpy()
    #     len_df = len(df_copy)

    #     for i in range(len_df):
    #         current_close_val = close_series[i]
    #         if np.isnan(current_close_val) or current_close_val == 0:
    #             future_change_pct_list.append(np.nan)
    #             continue
    #         lookahead = lookahead_series[i]
    #         future_index = i + lookahead
    #         if future_index >= len_df:
    #             future_change_pct_list.append(np.nan)
    #         else:
    #             future_close_val = close_series[future_index]
    #             if np.isnan(future_close_val):
    #                 future_change_pct_list.append(np.nan)
    #             else:
    #                 change_pct = (future_close_val - current_close_val) / current_close_val
    #                 future_change_pct_list.append(change_pct)
    
    #     df_copy["future_change_pct"] = future_change_pct_list
        
    #     # --- 4. Calculate Smoothed Target Score (TS_pct) ---
    #     # Increased min_periods for more smoothing
    #     df_copy["TS_pct"] = df_copy["future_change_pct"].rolling(base_window, min_periods=ts_pct_min_periods).mean()

    #     # --- 5. Calculate Final Target 'T' with Adaptive Linear Scaling & Clipping ---
    #     # MODIFIED fillna
    #     df_copy["ATR_pct_for_norm"] = (df_copy["ATR"] / (current_close_safe + 1e-9)).replace(0, np.nan).bfill().ffill().fillna(1e-7)
    #     df_copy["T_raw"] = df_copy["TS_pct"] / (df_copy["ATR_pct_for_norm"] + 1e-7)
        
    #     # Adaptive scaling for T_raw
    #     abs_t_raw = df_copy["T_raw"].abs().dropna()
    #     scale_factor = 1.0 # Default scale_factor
    #     if not abs_t_raw.empty:
    #         # Scale T_raw so its 98th percentile maps to target_scaled_magnitude_at_percentile
    #         percentile_val = abs_t_raw.quantile(0.98) 
    #         if percentile_val > 1e-6: # Avoid division by zero
    #             scale_factor = target_scaled_magnitude_at_percentile / percentile_val
        
    #     df_copy["T_scaled"] = df_copy["T_raw"] * scale_factor
    #     df_copy["T"] = np.clip(df_copy["T_scaled"], -final_clip_range_abs, final_clip_range_abs)
        
    #     # Optional: Light final smoothing on T
    #     if final_T_smoothing_window > 1:
    #         df_copy["T"] = df_copy["T"].rolling(final_T_smoothing_window, min_periods=1).mean()

    #     df_copy["T"] = df_copy["T"].fillna(0) 
        
    #     return df_copy["T"].reindex(dataframe.index)
        
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates only base indicators that don't use hyperoptable parameters.
        All parameter-dependent calculations are moved to calculate_indicators().
        """
        self.freqai_info = self.config["freqai"]
    
        df['close'] = pd.to_numeric(df['close'], errors='coerce').replace(0, np.nan).ffill().bfill()
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').replace(0, 1e-9).ffill().bfill().fillna(1e-9) 

        fit_live_candles = self.freqai_info.get("fit_live_predictions_candles", 0)
        if fit_live_candles <= 0 and getattr(self.dp, "runmode", None) in [RunMode.BACKTEST, RunMode.HYPEROPT]:
            logger.warning("fit_live_predictions_candles not properly configured. Using default of 48.")
            self.freqai_info["fit_live_predictions_candles"] = 48 # Example default
        
        df = self.freqai.start(df, metadata, self)
        
        freqai_cols = ["&-s_target", "&-s_target_mean", "&-s_target_std", "do_predict"]
        for col in freqai_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill().fillna(0)
            else:
                logger.warning(f"FreqAI column '{col}' missing after freqai.start(). Assigning 0.")
                df[col] = 0.0
                
        df["T"] = self.create_target_T(df)
        df["Prediction"] = df["&-s_target"]
        df["Avg Prediction"] = df["&-s_target_mean"] # FreqAI's mean for plotting
        # "custom_pred_mean" will be calculated in calculate_indicators and can be added to plot_config
        df["True Label"] = df["T"]
        
        log_metrics = (
            not hasattr(self, "dp")
            or getattr(self.dp, "runmode", None) in [RunMode.BACKTEST, RunMode.HYPEROPT, RunMode.PLOT]
        )
        self.compute_prediction_metrics(df, metadata, log_metrics=log_metrics)
        if log_metrics: self.save_prediction_metrics()

        if "pred_confidence" in df.columns:
            df["pred_confidence"] = pd.to_numeric(df["pred_confidence"], errors='coerce').ffill().bfill().fillna(0)
        else:
            logger.warning("'pred_confidence' missing after compute_prediction_metrics. Assigning 0.")
            df["pred_confidence"] = 0.0
                
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
                    else: df['trade_duration'] = 0
            else: df['trade_duration'] = 0
        
        return df
    
    def calculate_indicators(self, df: DataFrame) -> DataFrame:
        """
        Calculates all parameter-dependent indicators for each hyperopt epoch.
        Called from populate_entry_trend and populate_exit_trend.
        """
        # Use FIXED window sizes
        volatility_window = self.fixed_vol_window
        trend_window = self.fixed_trend_window
    
        # 1. ATR (Volatility)
        atr_series_calc = pd.Series(ta.ATR(df, timeperiod=volatility_window), index=df.index)
        df["atr"] = atr_series_calc.bfill().ffill().fillna(1e-9)

        safe_close = df["close"].replace(0, np.nan).bfill().ffill().fillna(1e-9)
        df["atr_normalized"] = (df["atr"] / safe_close) 

        # 2. Volume Rank & Dynamic Threshold
        df["vol_rank"] = df["volume"].rolling(volatility_window).rank(pct=True).fillna(0)
        df['vol_rank_dynamic_threshold'] = df['vol_rank'].rolling(volatility_window).quantile(self.vol_rank_quantile.value).ffill().bfill().fillna(0.1)


        # 3. Rolling Trend
        pct_change_period = trend_window
        rolling_window_trend = max(2, int(trend_window // 4))
        df["rolling_trend"] = df["close"].pct_change(pct_change_period).rolling(rolling_window_trend).mean().fillna(0)

        # 4. ATR Scaling
        atr_quantile_val = 0.90 # Fixed quantile for scaling ATR
        quantile_val_atr = df["atr_normalized"].rolling(trend_window).quantile(atr_quantile_val).ffill().bfill().fillna(1e-9)
        df["atr_scaled"] = (df["atr_normalized"] / quantile_val_atr.clip(lower=1e-9)).clip(0, 1).fillna(0)

        # 5. Rolling Trend Scaling
        mean_trend = df["rolling_trend"].rolling(trend_window).mean().ffill().fillna(0)
        std_trend = df["rolling_trend"].rolling(trend_window).std().ffill().fillna(1e-9)
        df["rolling_trend_scaled"] = (df["rolling_trend"] - mean_trend) / std_trend.clip(lower=1e-9)
        df["rolling_trend_scaled"] = df["rolling_trend_scaled"].fillna(0)

        # ✅ Calculate custom smoothed predictions using the hyperopt parameter
        if "&-s_target" in df.columns:
            custom_smooth_window = int(self.prediction_smoothing_window.value)
            df["custom_pred_mean"] = df["&-s_target"].rolling(window=custom_smooth_window, min_periods=1).mean().fillna(0)
            df["custom_pred_std"] = df["&-s_target"].rolling(window=custom_smooth_window, min_periods=1).std().fillna(0)
        else:
            logger.warning("Column '&-s_target' not found in calculate_indicators. Using FreqAI's mean/std if available.")
            df["custom_pred_mean"] = df.get("&-s_target_mean", 0.0) 
            df["custom_pred_std"] = df.get("&-s_target_std", 0.0)  

        # 6. Dynamic Thresholds Bases
        pred_mean_col_to_use = "custom_pred_mean" # Always try to use custom
        pred_std_col_to_use = "custom_pred_std"   # Always try to use custom
        
        # Fallback if custom calculation failed (e.g., &-s_target was missing)
        if not ("custom_pred_mean" in df.columns and df["custom_pred_mean"].notna().any()):
            pred_mean_col_to_use = "&-s_target_mean"
            logger.warning("custom_pred_mean is NaN or missing, falling back to &-s_target_mean for thresholds.")
        if not ("custom_pred_std" in df.columns and df["custom_pred_std"].notna().any()):
            pred_std_col_to_use = "&-s_target_std"
            logger.warning("custom_pred_std is NaN or missing, falling back to &-s_target_std for thresholds.")


        df["dynamic_long_threshold_base"] = df[pred_mean_col_to_use] + df[pred_std_col_to_use] * df["atr_scaled"]
        df["dynamic_short_threshold_base"] = df[pred_mean_col_to_use] - df[pred_std_col_to_use] * df["atr_scaled"]
        df["exit_long_threshold_base"] = df[pred_mean_col_to_use] 
        df["exit_short_threshold_base"] = df[pred_mean_col_to_use] 
        df["rolling_trend_threshold_base"] = df["rolling_trend_scaled"].rolling(100, min_periods=10).median().ffill().fillna(0)

        # 7. Apply Threshold Multipliers (Hyperoptable or Fixed)
        df["long_threshold"] = df["dynamic_long_threshold_base"] * float(self.dynamic_long_threshold_multiplier.value)
        df["exit_long_threshold"] = df["exit_long_threshold_base"] * float(self.exit_long_threshold_multiplier.value) 
        df["exit_short_threshold"] = df["exit_short_threshold_base"] * float(self.exit_short_threshold_multiplier.value)           
        df["short_threshold"] = df["dynamic_short_threshold_base"] * float(self.dynamic_short_threshold_multiplier.value)
        # Use fixed multiplier for rolling trend threshold
        df["rolling_trend_threshold"] = df["rolling_trend_threshold_base"] * self.fixed_rolling_trend_threshold_multiplier

        # 8. Confidence Threshold (Base is dynamic, multiplier is fixed)
        # 'pred_confidence' is from populate_indicators (based on FreqAI's mean/std)
        df["confidence_threshold_base"] = df["pred_confidence"].rolling(100).quantile(0.5).ffill().fillna(0)
        df["confidence_threshold"] = df["confidence_threshold_base"] * self.fixed_confidence_threshold_multiplier
        
        # Add custom_pred_mean to the main df for plotting if not already there (for hyperopt runs)
        if "custom_pred_mean" not in df.columns and pred_mean_col_to_use == "custom_pred_mean":
             if "&-s_target" in df.columns:
                custom_smooth_window = int(self.prediction_smoothing_window.value)
                df["custom_pred_mean"] = df["&-s_target"].rolling(window=custom_smooth_window, min_periods=1).mean().fillna(0)


        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None
        
        processed_df = self.calculate_indicators(df)

        prediction_signal_col = "custom_pred_mean" if self.use_mean_prediction_for_signal.value else "&-s_target"
        # Ensure the chosen column exists, fallback if necessary
        if prediction_signal_col not in processed_df.columns or not processed_df[prediction_signal_col].notna().any():
            logger.warning(f"Prediction signal column '{prediction_signal_col}' is missing or all NaN. Falling back to '&-s_target'.")
            prediction_signal_col = "&-s_target"
            if prediction_signal_col not in processed_df.columns: # Ultimate fallback
                 logger.error(f"Ultimate fallback column '&-s_target' also missing. Cannot generate entry signals.")
                 return df # Or raise error


        if self.use_crossed_entry.value:
            long_signal_active = crossed_above(processed_df[prediction_signal_col], processed_df["long_threshold"])
            short_signal_active = crossed_below(processed_df[prediction_signal_col], processed_df["short_threshold"])
        else:
            long_signal_active = processed_df[prediction_signal_col] > processed_df["long_threshold"]
            short_signal_active = processed_df[prediction_signal_col] < processed_df["short_threshold"]
        
        def base_entry_condition_series(side: str = None):
            condition = (processed_df["do_predict"] == 1)
            
            if self.use_vol_filter.value: # Uses dynamic threshold now
                condition &= (processed_df["vol_rank"] > processed_df["vol_rank_dynamic_threshold"])
                
            if self.use_confidence_filter_entry.value:
                condition &= (processed_df["pred_confidence"] > processed_df["confidence_threshold"])
                
            if self.use_trend_filter.value:
                if side == "long":
                    condition &= (processed_df["rolling_trend_scaled"] > processed_df["rolling_trend_threshold"])
                elif side == "short":
                    condition &= (processed_df["rolling_trend_scaled"] < processed_df["rolling_trend_threshold"])
            return condition
        
        final_long_entry_condition = long_signal_active & base_entry_condition_series(side="long")
        final_short_entry_condition = short_signal_active & base_entry_condition_series(side="short")

        df.loc[final_long_entry_condition, ["enter_long", "enter_tag"]] = (1, "long")
        df.loc[final_short_entry_condition & (df["enter_long"] == 0), ["enter_short", "enter_tag"]] = (1, "short")

        # Fallback uses FIXED high_confidence_threshold
        high_confidence = processed_df["pred_confidence"] > self.fixed_high_confidence_threshold
        
        fallback_long_condition_met = (
            (processed_df["do_predict"] == 1) & 
            (processed_df[prediction_signal_col] > 0.01 * processed_df["atr_scaled"]) & 
            high_confidence &
            crossed_above(processed_df["rolling_trend_scaled"], processed_df["rolling_trend_threshold"])
        )
        
        fallback_short_condition_met = (
            (processed_df["do_predict"] == 1) & 
            (processed_df[prediction_signal_col] < -0.01 * processed_df["atr_scaled"]) & 
            high_confidence &
            crossed_below(processed_df["rolling_trend_scaled"], processed_df["rolling_trend_threshold"])
        )

        df.loc[
            fallback_long_condition_met & (df["enter_long"] == 0) & (df["enter_short"] == 0),
            ["enter_long", "enter_tag"]
        ] = (1, "long_fallback")

        df.loc[
            fallback_short_condition_met & (df["enter_long"] == 0) & (df["enter_short"] == 0),
            ["enter_short", "enter_tag"]
        ] = (1, "short_fallback")
        
        return df
    
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = False 
        df['exit_short'] = False 
        df['exit_tag'] = None
        
        processed_df = self.calculate_indicators(df)

        prediction_signal_col = "custom_pred_mean" if self.use_mean_prediction_for_signal.value else "&-s_target"
        if prediction_signal_col not in processed_df.columns or not processed_df[prediction_signal_col].notna().any():
            logger.warning(f"Prediction signal column '{prediction_signal_col}' is missing or all NaN in exit. Falling back to '&-s_target'.")
            prediction_signal_col = "&-s_target"
            if prediction_signal_col not in processed_df.columns:
                 logger.error(f"Ultimate fallback column '&-s_target' also missing. Cannot generate exit signals.")
                 return df


        base_exit_condition = (
            (processed_df['pred_confidence'] > processed_df["confidence_threshold"]) & 
            (processed_df["do_predict"] == 1)
        )
        
        if self.use_vol_filter.value: 
            base_exit_condition &= processed_df['vol_rank'] > processed_df["vol_rank_dynamic_threshold"]
        
        if self.use_target_exit_filter.value:
            df.loc[
                (df['exit_long'] == False) & 
                crossed_below(processed_df[prediction_signal_col], processed_df["exit_long_threshold"]) & 
                base_exit_condition,
                ['exit_long', 'exit_tag']
            ] = (True, 'target_exit_long')
            
            df.loc[
                (df['exit_short'] == False) & 
                crossed_above(processed_df[prediction_signal_col], processed_df["exit_short_threshold"]) & 
                base_exit_condition,
                ['exit_short', 'exit_tag']
            ] = (True, 'target_exit_short')
    
        if self.use_trend_exit_filter.value:
            df.loc[
                (df['exit_long'] == False) & 
                crossed_below(processed_df['rolling_trend_scaled'], processed_df["rolling_trend_threshold"]) &
                base_exit_condition,
                ['exit_long', 'exit_tag']
            ] = (True, 'trend_exit_long') 
            
            df.loc[
                (df['exit_short'] == False) & 
                crossed_above(processed_df['rolling_trend_scaled'], processed_df["rolling_trend_threshold"]) &
                base_exit_condition,
                ['exit_short', 'exit_tag']
            ] = (True, 'trend_exit_short') 
    
        if self.use_timed_exit.value: # Uses new combined timed_exit_duration
            df.loc[
                (df['exit_long'] == False) & 
                (df['trade_duration'] > self.timed_exit_duration.value) &
                (processed_df['pred_confidence'] < processed_df['confidence_threshold']), 
                ['exit_long', 'exit_tag']
            ] = (True, 'timed_exit_long') 
            
            df.loc[
                (df['exit_short'] == False) & 
                (df['trade_duration'] > self.timed_exit_duration.value) &
                (processed_df['pred_confidence'] < processed_df['confidence_threshold']), 
                ['exit_short', 'exit_tag']
            ] = (True, 'timed_exit_short') 

        if self.use_opposite_signal_exit.value:    
            df.loc[df['enter_short'] == 1, ['exit_long', 'exit_tag']] = (True, 'opposite_signal_short')
            df.loc[df['enter_long'] == 1, ['exit_short', 'exit_tag']] = (True, 'opposite_signal_long')
    
        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty: return -1.0 
        if not trade or not trade.open_rate: return -1.0
    
        last_candle = dataframe.iloc[-1]
        
        # ATR: Use fixed_vol_window for consistency if ATR needs recalculation
        if 'atr' not in last_candle or pd.isna(last_candle.get('atr')) or last_candle.get('atr', 0) <= 0:
            atr_series = ta.ATR(dataframe, timeperiod=self.fixed_vol_window) # Use fixed window
            atr = atr_series.iloc[-1] if atr_series is not None and not atr_series.empty and pd.notna(atr_series.iloc[-1]) else current_rate * 0.01
        else:
            atr = last_candle.get('atr', current_rate * 0.01)
        atr = max(atr, current_rate * 0.005) 
    
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        prediction_confidence = last_candle.get("pred_confidence", 0.5)
    
        tf_minutes = timeframe_to_minutes(self.timeframe)
        elapsed_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        trade_duration_candles = elapsed_minutes / tf_minutes
            
        # Use FIXED parameters
        initial_duration = self.fixed_initial_stop_duration_candles
        soft_stop_val = self.fixed_soft_stoploss_pct # Already negative
        min_profit_val = self.fixed_min_profit_for_trailing

        # Get HYPEROPTABLE parameters
        atr_multiplier = self.atr_stoploss_multiplier.value
        hist_vol_factor = self.historical_volatility_factor.value
        pred_conf_factor = self.prediction_confidence_factor.value
        hard_max_stop_val = self.hard_max_stoploss_pct.value # Already negative
    
        if trade_duration_candles < initial_duration:
            return soft_stop_val
    
        if current_profit < min_profit_val:
            return soft_stop_val
    
        dynamic_volatility_factor = 1 + historical_volatility * hist_vol_factor
        confidence_factor_sl = 1 - (prediction_confidence * pred_conf_factor) # Higher confidence -> smaller factor -> tighter stop
        confidence_factor_sl = max(0.1, confidence_factor_sl) # Ensure it doesn't go to zero or negative

        stoploss_buffer_abs = atr * atr_multiplier * dynamic_volatility_factor * confidence_factor_sl
        
        # Ensure stoploss_buffer_abs is not excessively small, e.g., at least 0.1 * ATR
        stoploss_buffer_abs = max(stoploss_buffer_abs, atr * 0.1)

        stoploss_pct_calculated = stoploss_buffer_abs / trade.open_rate
    
        # Final stoploss is the tighter of the calculated dynamic stop or the hard max stop
        # Both hard_max_stop_val and -stoploss_pct_calculated are negative. We want the one closer to zero (less loss).
        final_stoploss_pct = max(-stoploss_pct_calculated, hard_max_stop_val) 
    
        return final_stoploss_pct if final_stoploss_pct < 0 else soft_stop_val


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float,
                            min_stake: float | None, max_stake: float, leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return proposed_stake

        last_candle = dataframe.iloc[-1]
        prediction_confidence = last_candle.get("pred_confidence", 0.5)
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        scaled_volatility = min(historical_volatility, 0.05)

        conf_influence = self.confidence_influence.value # Hyperoptable
        confidence_factor = 1.0 + (prediction_confidence - 0.5) * conf_influence * 2

        vol_influence = self.volatility_influence.value # Hyperoptable
        volatility_factor = max(0.1, 1.0 - (scaled_volatility * vol_influence * 10)) 

        stake_amount = proposed_stake * confidence_factor * volatility_factor
        stake_amount *= self.fixed_stake_scaling_factor # Use FIXED factor

        stake_amount = min(stake_amount, max_stake) 
        if min_stake and stake_amount < min_stake:
            stake_amount = min_stake
        stake_amount = min(stake_amount, max_stake)

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
            return min_leverage
    
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
        if not LSTMStrategy_v44_1h.pred_metrics_storage:
            logger.warning("⚠️ No prediction metrics found to save.")
            return

        try:
            # Use class attribute directly
            df = pd.DataFrame(LSTMStrategy_v44_1h.pred_metrics_storage)
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