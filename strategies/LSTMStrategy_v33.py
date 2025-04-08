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
from freqtrade.optimize.analysis import lookahead
from freqtrade.strategy import IStrategy, IntParameter, RealParameter, CategoricalParameter
from freqtrade.persistence import Trade
from freqtrade.enums import RunMode
from freqtrade.vendor.qtpylib.indicators import crossed_above, crossed_below

logger = logging.getLogger(__name__)


class LSTMStrategy_v33(IStrategy):
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
                "prediction_confidence": {"color": "orange", "plot_type": "line"},  # Plot prediction confidence
                "confidence_threshold": {"color": "brown", "plot_type": "line"},
            },
            "Indicators": {
                "atr_scaled": {"color": "blue", "plot_type": "line"},
            },
            "Thresholds": {
                "rolling_trend_threshold": {"color": "blue", "plot_type": "line"},
                "vol_rank": {"color": "orange", "plot_type": "line"},
                "dynamic_long_threshold": {"color": "green", "plot_type": "line"},
                "dynamic_short_threshold": {"color": "red", "plot_type": "line"},
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

    # ✅ Entry/Exit hyperopt parameters
    dynamic_long_threshold_multiplier = RealParameter(0.7, 1.5, default=1.0, space="buy", load=True, optimize=True)
    dynamic_short_threshold_multiplier = RealParameter(0.7, 1.5, default=1.0, space="buy", load=True, optimize=True)
    confidence_threshold_multiplier = RealParameter(0.3, 1.0, default=1.0, space="buy", load=True, optimize=True)
    high_confidence_threshold = RealParameter(0.7, 0.95, default=0.85, space="buy", load=True, optimize=True)
    use_confidence_filter_entry = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    use_trend_filter = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    rolling_trend_threshold_multiplier = RealParameter(0.2, 2.0, default=1.1, space="buy", load=True, optimize=True)
    leverage_scaling_factor = RealParameter(0.01, 0.2, default=0.04, space="buy", load=True, optimize=True)

    # exit trend hyperopt parameters
    dynamic_long_exit_threshold_multiplier = RealParameter(0.4, 2.0, default=1.0, space="sell", load=True, optimize=True)
    dynamic_short_exit_threshold_multiplier = RealParameter(0.4, 2.0, default=1.0, space="sell", load=True, optimize=True)
    timed_exit_long_threshold = IntParameter(10, 40, default=20, space="sell", load=True, optimize=True)
    timed_exit_short_threshold = IntParameter(10, 40, default=20, space="sell", load=True, optimize=True)
    exit_trend_threshold_multiplier = RealParameter(0.2, 0.6, default=0.35, space="sell", load=True, optimize=True)
    use_target_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    use_trend_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    use_timed_exit = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)

    # ✅ Stoploss Hyperopt Parameters
    soft_stoploss_pct = RealParameter(-0.30, -0.01, default=-0.10, space="sell", load=True, optimize=True)
    min_trade_duration = IntParameter(6, 24, default=12, space="sell", load=True, optimize=True)
    min_profit_for_trailing = RealParameter(0.02, 0.06, default=0.04, space="sell", load=True, optimize=True)
    atr_stoploss_multiplier = RealParameter(1.0, 3.0, default=1.5, space="sell", load=True, optimize=True)
    historical_volatility_factor = RealParameter(0.3, 1.2, default=0.5, space="sell", load=True, optimize=True)
    prediction_confidence_factor = RealParameter(0.2, 1.0, default=0.6, space="sell", load=True, optimize=True)
    max_risk_per_trade_multiplier = RealParameter(0.003, 0.07, default=0.02, space="sell", load=True, optimize=True)

    # ✅ Stake amount hyperopt parameters
    stake_scaling_factor = RealParameter(0.4, 1.5, default=1.0, space="buy", load=True, optimize=True)
    base_risk = RealParameter(0.005, 0.10, default=0.02, space="sell", load=True, optimize=True)

    # Buy hyperspace params:
    buy_params = {
        "confidence_threshold_multiplier": 0.62404,
        "dynamic_long_threshold_multiplier": 1.20256,
        "dynamic_short_threshold_multiplier": 1.48173,
        "rolling_trend_threshold_multiplier": 0.83203,
        "stake_scaling_factor": 1.45356,
        "use_confidence_filter_entry": False,
        "use_trend_filter": False,
    }

    # Sell hyperspace params:
    sell_params = {
        "atr_stoploss_multiplier": 2.85347,
        "base_risk": 0.07642,
        "dynamic_long_exit_threshold_multiplier": 0.9028,
        "dynamic_short_exit_threshold_multiplier": 1.15029,
        "exit_trend_threshold_multiplier": 0.2044,
        "historical_volatility_factor": 0.86773,
        "max_risk_per_trade_multiplier": 0.01267,
        "min_profit_for_trailing": 0.02362,
        "min_trade_duration": 14,
        "prediction_confidence_factor": 0.87805,
        "soft_stoploss_pct": -0.06259,
        "timed_exit_long_threshold": 21,
        "timed_exit_short_threshold": 18,
        "use_target_exit_filter": False,
        "use_trend_exit_filter": True,
        "use_timed_exit": True,
    }

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.trades: Dict[str, datetime] = {}  # Initialize self.trades in the constructor
        # ✅ Load historical Fear & Greed Index data if available
        self.historical_fng_data = self.load_historical_fng_data()

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, metadata: Dict, **kwargs):
        """
        Expands all features for FreqAI while keeping feature count optimized.
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

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]

        # ✅ Temporarily Remove Lower-Impact Indicators (Can Reintroduce if Needed)
        drop_columns = [
            "%-cci-period", "%-momentum-period", "%-macd-period",
            "%-macdsignal-period", "%-macdhist-period"
        ]
        dataframe.drop(columns=[col for col in drop_columns if col in dataframe.columns], inplace=True, errors="ignore")

        # ✅ Fix NaNs
        dataframe.fillna(0, inplace=True)

        # ✅ **Optimized Lag-Based Features**
        lag_amount = 3  # ⬇ Reduced from 6 to 3
        lag_features = ["close", "%-rsi-period"]  # **Limited to key trend indicators**

        # ✅ Efficient lagging using `pd.concat()`
        lagged_data = {f"{feature}_lag{lag}": dataframe[feature].shift(lag) for feature in lag_features for lag in range(1, lag_amount + 1)}
        dataframe = pd.concat([dataframe, pd.DataFrame(lagged_data, index=dataframe.index)], axis=1)

        # ✅ Fill NaNs from Lagged Features (Backfill to Avoid Data Loss)
        dataframe.loc[:, dataframe.columns.str.contains("_lag")] = dataframe.loc[:, dataframe.columns.str.contains("_lag")].bfill()

        # ✅ Apply Z-Score Normalization to **volatile features only**
        zscore_columns = ["%-bb_width-period", "%-rsi-period", "%-roc-period"]
        for col in zscore_columns:
            dataframe.loc[:, f"{col}-zscore"] = pd.Series(zscore(dataframe[col]), index=dataframe.index).fillna(0)

        # logger.info(f"🔍 Strict feature selection applied. Total features: {len(dataframe.columns)}")

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, metadata: Dict, **kwargs):
        """
        Defines features that should remain in their original timeframe.
        """

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

        # Ensure ATR exist in live/dry-run mode
        if "atr" not in dataframe.columns:
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=14).bfill()

        # ✅ Normalize Fourier Features using ATR
        dataframe.loc[:, "%-fourier_price_norm"] = dataframe["%-fourier_price"] / (dataframe["atr"] + 1e-6)

        # ✅ Apply Z-Score Normalization to **volatile features only**
        zscore_columns = ["%-rolling_volatility", "%-rolling_mean", "%-fourier_price_norm"]
        for col in zscore_columns:
            dataframe.loc[:, f"{col}-zscore"] = pd.Series(zscore(dataframe[col]), index=dataframe.index).fillna(0)

        # ✅ Incorporate order flow features
        dataframe = self.get_order_flow_features(dataframe, metadata)

        # ✅ Fetch Fear & Greed Index
        current_date = dataframe['date'].iloc[-1] if 'date' in dataframe else None
        fear_greed_value, fear_greed_classification = self.get_fear_and_greed_index(current_date)
        # ✅ Add Fear & Greed Index to DataFrame
        dataframe['fear_greed_index'] = fear_greed_value
        # ✅ Fill missing values (if any)
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
        """

        dataframe["ATR"] = ta.ATR(dataframe, timeperiod=14).bfill()  # ATR-based normalization
        dataframe["close"] = dataframe["close"].replace(0, np.nan).bfill()  # Prevent division by zero

        # ✅ Compute dynamic lookahead (ensuring valid values)
        dataframe["lookahead_dynamic"] = np.clip((dataframe["ATR"] / dataframe["close"]) * 100, 5, 20).fillna(10).astype(int)

        # ✅ Compute Future Price Change dynamically using `.apply()`
        dataframe["future_change"] = dataframe.apply(
            lambda row: dataframe["close"].shift(-int(row["lookahead_dynamic"])).iloc[row.name] - row["close"],
            axis=1
        )

        # ✅ Compute Trend Strength Using Future Price Change
        dataframe["TS"] = dataframe["future_change"].rolling(14).mean()

        # ✅ Normalize Trend Strength Using ATR + Std Dev
        dataframe["T"] = dataframe["TS"] / (
            0.5 * dataframe["ATR"] + 0.5 * dataframe["close"].rolling(14).std() + 1e-6
        )

        # ✅ Apply `tanh()` to Limit Extreme Values
        dataframe["T"] = np.tanh(dataframe["T"])

        # 🔧 Fix: No more inplace modification
        dataframe["T"] = dataframe["T"].fillna(0)

        return dataframe["T"]
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]
        # logger.info(f"[DEBUG] Entered populate_indicators for {metadata['pair']}")

        # ✅ ATR Calculation (Same as Original)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14).bfill()

        # ✅ Volume Rank (Same as Original)
        dataframe["vol_rank"] = dataframe["volume"].rolling(24).rank(pct=True).fillna(0)

        # ✅ Improved Rolling Trend (Same concept but more adaptive)
        dataframe["rolling_trend"] = dataframe["close"].pct_change(24).rolling(6).mean().fillna(0)

        # ✅ Improved ATR Scaling: Using Moving Percentile Instead of Min/Max
        atr_window = 100
        dataframe["atr_scaled"] = dataframe["atr"] / dataframe["atr"].rolling(atr_window).quantile(0.90)
        dataframe["atr_scaled"] = dataframe["atr_scaled"].bfill().clip(0.05, 1)

        # ✅ Improved Rolling Trend Scaling: Adaptive Window + Normalization
        trend_window = 100
        adaptive_window = int(min(trend_window, max(20, dataframe["atr"].rolling(50).mean().iloc[-1] * 10)))
        mean_trend = dataframe["rolling_trend"].rolling(adaptive_window).mean()
        std_trend = dataframe["rolling_trend"].rolling(adaptive_window).std()
        dataframe["rolling_trend_scaled"] = (dataframe["rolling_trend"] - mean_trend) / (std_trend + 1e-6)

        # ✅ Call FreqAI for ML Predictions
        dataframe = self.freqai.start(dataframe, metadata, self)
        logger.info(f"[DEBUG] FreqAI finished start() call for {metadata['pair']}")

        # ✅ Store Threshold Base Values (No Hyperopt Parameters Here, Same as Original)
        dataframe["dynamic_long_threshold_base"] = dataframe["&-s_target_mean"] + dataframe["&-s_target_std"] * dataframe["atr_scaled"]
        dataframe["dynamic_short_threshold_base"] = dataframe["&-s_target_mean"] - dataframe["&-s_target_std"] * dataframe["atr_scaled"]
        dataframe["rolling_trend_threshold_base"] = dataframe["rolling_trend_scaled"].rolling(100, min_periods=10).median()
        dataframe["dynamic_exit_threshold_base"] = (
            dataframe["&-s_target"].ewm(span=50).mean() +
            dataframe["atr_scaled"] * dataframe["&-s_target_std"] * (0.6 + dataframe["vol_rank"] * 0.3)
        )
        dataframe["exit_trend_threshold_base"] = dataframe["rolling_trend_scaled"].rolling(50).median()

        # ✅ Keeping `T` for Plotting Purposes Only (Not Used in Trade Logic)
        dataframe["T"] = self.create_target_T(dataframe)
        dataframe["Prediction"] = dataframe["&-s_target"]
        dataframe["Avg Prediction"] = dataframe["&-s_target_mean"]
        dataframe["True Label"] = dataframe["T"]

        # ✅ Compute & Save Prediction Metrics (Same as Original)
        if hasattr(self, "dp") and hasattr(self.dp, "runmode"):
            logger.info(f"📦 runmode: {self.dp.runmode.value}")
        else:
            logger.info("📦 runmode: unavailable (likely during training phase)")

        # only enable prediction metrics logs at training mode
        log_metrics = (
            not hasattr(self, "dp")
            or getattr(self.dp, "runmode", None) in [RunMode.BACKTEST, RunMode.HYPEROPT]
        )        
        # ✅ Compute & Save Prediction Metrics (Same as Original)
        self.compute_prediction_metrics(dataframe, metadata, log_metrics=log_metrics)
        self.save_prediction_metrics()

        dataframe["confidence_threshold_base"] = dataframe["prediction_confidence"].rolling(100).quantile(0.5).bfill()

        # ✅ Calculate trade duration (in candles)
        if metadata['pair'] not in self.trades:
            dataframe['trade_duration'] = 0
        else:
            open_since = self.trades[metadata['pair']]['open_since']
            dataframe['trade_duration'] = dataframe.index - open_since
            dataframe['trade_duration'] = dataframe['trade_duration'].apply(lambda x: x.total_seconds() / 60 / self.timeframe_minutes)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines the entry signal for both long and short trades using crossed_above/below.
        """

        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None  # Initialize enter_tag

        # ─── Long Entry Logic ───────────────────────────
        long_entry_condition = (df["do_predict"] == 1) & (df["vol_rank"] > 0.10)
        if self.use_confidence_filter_entry.value:
            long_entry_condition &= (df["prediction_confidence"] > df["confidence_threshold_base"] * self.confidence_threshold_multiplier.value)
        if self.use_trend_filter.value:
            long_entry_condition &= (df["rolling_trend_scaled"] > df["rolling_trend_threshold_base"] * self.rolling_trend_threshold_multiplier.value)

        # Apply crossed_above for dynamic_long_threshold
        df.loc[
            crossed_above(df["&-s_target"], (df["dynamic_long_threshold_base"] * self.dynamic_long_threshold_multiplier.value)) &
            long_entry_condition,
            ["enter_long", "enter_tag"]
        ] = (1, "long")

        # ─── Short Entry Logic ──────────────────────────
        short_entry_condition = (df["do_predict"] == 1) & (df["vol_rank"] > 0.10)
        if self.use_confidence_filter_entry.value:
            short_entry_condition &= (df["prediction_confidence"] > df["confidence_threshold_base"] * self.confidence_threshold_multiplier.value)
        if self.use_trend_filter.value:
            short_entry_condition &= (df["rolling_trend_scaled"] < df["rolling_trend_threshold_base"] * self.rolling_trend_threshold_multiplier.value)

        # Apply crossed_below for dynamic_short_threshold
        df.loc[
            crossed_below(df["&-s_target"], (df["dynamic_short_threshold_base"] * self.dynamic_short_threshold_multiplier.value)) &
            short_entry_condition,
            ["enter_short", "enter_tag"]
        ] = (1, "short")

        # ─── Fallback Entry ─────────────────────────────
        high_confidence = df["prediction_confidence"] > self.high_confidence_threshold.value

        df.loc[
            (df["do_predict"] == 1) & (df["&-s_target"] > 0.01) & high_confidence & (df["enter_long"] == 0),
            ["enter_long", "enter_tag"]
        ] = (1, "long_fallback")

        df.loc[
            (df["do_predict"] == 1) & (df["&-s_target"] < -0.01) & high_confidence & (df["enter_short"] == 0),
            ["enter_short", "enter_tag"]
        ] = (1, "short_fallback")

        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines the exit signal for both long and short trades using crossed_above/below.
        """

        df['exit_long'] = False
        df['exit_short'] = False

        # 1. Target Reversal Exit (Primary)
        if self.use_target_exit_filter.value:
            df.loc[
                crossed_below(df['&-s_target'], (df['dynamic_exit_threshold_base'] * self.dynamic_long_exit_threshold_multiplier.value)) &
                (df['prediction_confidence'] > df["confidence_threshold_base"] * self.confidence_threshold_multiplier.value),
                'exit_long'
            ] = True
            df.loc[
                crossed_above(df['&-s_target'], (df['dynamic_exit_threshold_base'] * self.dynamic_short_exit_threshold_multiplier.value)) &
                (df['prediction_confidence'] > df["confidence_threshold_base"] * self.confidence_threshold_multiplier.value),
                'exit_short'
            ] = True

        # 2. Trend Reversal Exit (Secondary - if target exit not triggered)
        if self.use_trend_exit_filter.value:
            df.loc[
                crossed_below(df['rolling_trend'], (df['rolling_trend_threshold_base'] * self.exit_trend_threshold_multiplier.value)) &
                (df['exit_long'] == False),  # Only if target exit not triggered
                'exit_long'
            ] = True
            df.loc[
                crossed_above(df['rolling_trend'], (df['rolling_trend_threshold_base'] * self.exit_trend_threshold_multiplier.value)) &
                (df['exit_short'] == False),  # Only if target exit not triggered
                'exit_short'
            ] = True

        # 3. Timed Exit (Tertiary - if other exits not triggered)
        # Timed exit doesn't lend itself well to crossed_above/below, so we keep it as is.
        if self.use_timed_exit.value:
            df.loc[
                (df['trade_duration'] > self.timed_exit_long_threshold.value) &
                (df['exit_long'] == False),  # Only if other exits not triggered
                'exit_long'
            ] = True
            df.loc[
                (df['trade_duration'] > self.timed_exit_short_threshold.value) &
                (df['exit_short'] == False),  # Only if other exits not triggered
                'exit_short'
            ] = True

        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss  # Default stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        prediction_confidence = last_candle.get("prediction_confidence", 0.5)

        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600  

        # ✅ Use optimized Hyperopt parameters
        soft_stoploss_pct = self.soft_stoploss_pct.value  # Dynamic soft stoploss
        min_trade_duration = self.min_trade_duration.value  # Minimum time before stoploss applies
        min_profit_for_trailing = self.min_profit_for_trailing.value  # When trailing stop activates
        atr_multiplier = self.atr_stoploss_multiplier.value  # ATR scaling factor
        max_risk_per_trade_multiplier = self.max_risk_per_trade_multiplier.value  # Max risk multiplier

        # ✅ Apply ATR-based early stoploss buffer before full stoploss activates
        if trade_duration < min_trade_duration:
            return -atr * 1.5  # ✅ Adaptive early stoploss instead of static %

        # ✅ Trailing stop only applies if trade reaches optimized profit threshold
        if current_profit < min_profit_for_trailing:
            return soft_stoploss_pct  # ✅ Keep loose stoploss if not profitable yet

        # ✅ Use historical volatility to dynamically scale stoploss
        dynamic_volatility_factor = 1 + historical_volatility * self.historical_volatility_factor.value  # ✅ More volatile markets get looser stoploss

        # ✅ Use prediction confidence to fine-tune stoploss flexibility
        confidence_factor = 1 - (prediction_confidence * self.prediction_confidence_factor.value)  # ✅ Less aggressive tightening for high-confidence trades

        # ✅ ATR-based stoploss that adjusts dynamically
        stoploss_buffer = atr * atr_multiplier * 2.5 * dynamic_volatility_factor * confidence_factor

        # ✅ Set max stoploss dynamically based on market conditions
        max_loss_pct = min(0.03 + historical_volatility * 1.5, 0.08) * max_risk_per_trade_multiplier  # ✅ Allow up to 8% stoploss

        # ✅ Adjust stoploss logic for long and short trades
        if trade.is_short:
            stoploss_value = current_rate + stoploss_buffer * 1.7  # ✅ More room for shorts
        else:
            stoploss_value = current_rate - stoploss_buffer * 1.4  # ✅ Slightly looser for longs

        # ✅ Ensure stoploss never exceeds dynamic max loss threshold
        return min(stoploss_value, -max_loss_pct)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float,
                            min_stake: float | None, max_stake: float, leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return proposed_stake

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01

        adjusted_risk = self.base_risk.value * (1 + historical_volatility)
        max_risk = max_stake * adjusted_risk

        stake_amount = (max_risk / (atr * leverage)) * self.stake_scaling_factor.value if atr > 0 else max_risk
        stake_amount = min(stake_amount, max_stake, proposed_stake)
        if min_stake and stake_amount < min_stake:
            stake_amount = min_stake

        # logger.info(f"[STAKE] Pair: {pair} | ATR: {atr:.4f} | HV: {historical_volatility:.4f} | Risk: {adjusted_risk:.4f} | Stake: {stake_amount:.2f}")

        return stake_amount

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade based on market risk and prediction confidence.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            logger.warn(f"dataframe empty!")
            return 1.0  # Default leverage

        last_candle = dataframe.iloc[-1]
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        prediction_confidence = last_candle.get("prediction_confidence", 0.5)

        # ✅ Calculate leverage based on volatility and confidence
        volatility_factor = 1 - historical_volatility  # Lower volatility -> higher leverage
        confidence_factor = prediction_confidence  # Higher confidence -> higher leverage

        # ✅ Apply hyperopt scaling factor
        leverage_value = self.leverage_scaling_factor.value * volatility_factor * confidence_factor * max_leverage

        # ✅ Clip leverage to be within the allowed range
        leverage_value = int(min(max(1.0, leverage_value), max_leverage))

        logger.info(f"[LEVERAGE] Pair: {pair} | Side: {side} | Confidence: {prediction_confidence:.2f} | Leverage: {leverage_value:.2f}")
        logger.info(f"[LEVERAGE] Vol: {volatility_factor:.2f} | Conf: {confidence_factor:.2f} | Max: {max_leverage:.2f}")
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
                logger.warning(f"⚠️ Required order flow columns not found in dataframe for {metadata['pair']}. Skipping order flow features.")
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
                    logger.info(f"fng: {latest_data}")
                    return int(latest_data['value']), latest_data['value_classification']
                else:
                    return None, None
            except requests.exceptions.RequestException as e:
                print(f"Error fetching Fear & Greed Index: {e}")
                return None, None