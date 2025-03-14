import logging
from functools import reduce
from typing import Dict
import joblib
import os
from datetime import datetime

import numpy as np
import pandas as pd
import talib.abstract as ta
from technical import qtpylib

from pandas import DataFrame
from technical import qtpylib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.fftpack import fft
from scipy.stats import zscore
from torch import mul

from freqtrade import data
from freqtrade.exchange.exchange_utils import *
from freqtrade.optimize.analysis import lookahead
from freqtrade.strategy import IStrategy, RealParameter
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class ExampleLSTMStrategy_v2(IStrategy):
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

    threshold_buy = RealParameter(-1, 1, default=0, space='buy')
    threshold_sell = RealParameter(-1, 1, default=0, space='sell')

    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True
    use_custom_stoploss = True

    startup_candle_count = 20
                                                
    prediction_metrics_storage = []  # Class-level storage for all pairs

    from freqtrade.strategy import IntParameter, RealParameter, CategoricalParameter

    # ✅ Hyperopt Parameters
    # ✅ Real Parameters (Fine-tuned continuous values)
    dynamic_long_threshold_multiplier = RealParameter(0.8, 1.2, default=1.0, space='buy')
    dynamic_short_threshold_multiplier = RealParameter(0.8, 1.2, default=1.0, space='sell')
    confidence_threshold_multiplier = RealParameter(0.5, 0.8, default=0.65, space='buy')
    dynamic_exit_threshold_multiplier = RealParameter(0.5, 1.0, default=0.6, space='sell')
    exit_trend_threshold_multiplier = RealParameter(0.2, 0.5, default=0.35, space='sell')
    atr_multiplier = RealParameter(0.5, 2.0, default=1.5, space='sell')
    base_risk = RealParameter(0.005, 0.05, default=0.02, space='sell')
    rolling_trend_threshold_multiplier = RealParameter(0.8, 1.2, default=1.0, space='sell')
    dynamic_long_threshold_multiplier = RealParameter(0.8, 1.2, default=1.0, space='buy')
    dynamic_short_threshold_multiplier = RealParameter(0.8, 1.2, default=1.0, space='sell')
    confidence_threshold_multiplier = RealParameter(0.5, 0.8, default=0.65, space='buy')
    # ✅ Integer Parameters (Stepwise tuning)
    timed_exit_long_threshold = IntParameter(10, 30, default=20, space='sell')
    timed_exit_short_threshold = IntParameter(-30, -10, default=-20, space='sell')
    max_trade_duration_long = IntParameter(1, 5, default=2, space='sell')
    max_trade_duration_short = IntParameter(1, 3, default=1, space='sell')
    # ✅ Categorical Parameters (Fixed choices)
    stake_scaling_factor = CategoricalParameter([0.5, 1.0, 1.5], default=1.0, space='buy')
    max_risk_per_trade_multiplier = CategoricalParameter([0.01, 0.02, 0.03], default=0.02, space='sell')

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

        # ✅ Normalize Fourier Features using ATR
        dataframe.loc[:, "%-fourier_price_norm"] = dataframe["%-fourier_price"] / (dataframe["atr"] + 1e-6)

        # ✅ Apply Z-Score Normalization to **volatile features only**
        zscore_columns = ["%-rolling_volatility", "%-rolling_mean", "%-fourier_price_norm"]
        for col in zscore_columns:
            dataframe.loc[:, f"{col}-zscore"] = pd.Series(zscore(dataframe[col]), index=dataframe.index).fillna(0)

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

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14).bfill()
        dataframe["vol_rank"] = dataframe["volume"].rolling(24).rank(pct=True).fillna(0)
        dataframe["rolling_trend"] = dataframe["close"].pct_change(12).rolling(6).mean().fillna(0)

        atr_window = 100
        atr_min = dataframe["atr"].rolling(atr_window, min_periods=10).min()
        atr_max = dataframe["atr"].rolling(atr_window, min_periods=10).max()
        dataframe["atr_scaled"] = (dataframe["atr"] - atr_min) / (atr_max - atr_min + 1e-6)
        dataframe["atr_scaled"] = dataframe["atr_scaled"].fillna(method="bfill").clip(0.05, 1)

        trend_window = 100
        mean_trend = dataframe["rolling_trend"].rolling(trend_window).mean()
        std_trend = dataframe["rolling_trend"].rolling(trend_window).std()
        dataframe["rolling_trend_scaled"] = (dataframe["rolling_trend"] - mean_trend) / (std_trend + 1e-6)

        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["dynamic_long_threshold"] = dataframe["&-s_target_mean"] + dataframe["&-s_target_std"] * dataframe["atr_scaled"] * self.dynamic_long_threshold_multiplier.value
        dataframe["dynamic_short_threshold"] = dataframe["&-s_target_mean"] - dataframe["&-s_target_std"] * dataframe["atr_scaled"] * self.dynamic_short_threshold_multiplier.value
        dataframe["confidence_threshold"] = 0.25 + dataframe["atr_scaled"] * 0.20 * self.confidence_threshold_multiplier.value
        dataframe["rolling_trend_threshold"] = dataframe["rolling_trend_scaled"].rolling(100, min_periods=10).median() * self.rolling_trend_threshold_multiplier.value
        dataframe["dynamic_exit_threshold"] = (
            dataframe["&-s_target"].ewm(span=50).mean() +
            dataframe["atr_scaled"] * dataframe["&-s_target_std"] * (0.6 + dataframe["vol_rank"] * 0.3) * self.dynamic_exit_threshold_multiplier.value
        )
        dataframe["exit_trend_threshold"] = dataframe["rolling_trend_scaled"].rolling(50).median() * self.exit_trend_threshold_multiplier.value

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_short"] = 0
        df["enter_long"] = 0

        df["valid_volume"] = df["vol_rank"] > 0.15

        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_target"] > (df["dynamic_long_threshold"]),
            df["rolling_trend_scaled"] > (df["rolling_trend_threshold"]),
            df["vol_rank"] > 0.10,
            df["prediction_confidence"] > (df["confidence_threshold"])
        ]

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_target"] < (df["dynamic_short_threshold"]),
            df["rolling_trend_scaled"] < (df["rolling_trend_threshold"]),
            df["vol_rank"] > 0.18,
            df["prediction_confidence"] > (df["confidence_threshold"])
        ]

        df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (1, "long")
        df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_short"] = 0
        df["exit_long"] = 0

        df["active_short_trade"] = (df["enter_short"].cumsum() - df["exit_short"].cumsum()) > 0
        df["active_long_trade"] = (df["enter_long"].cumsum() - df["exit_long"].cumsum()) > 0

        df["timed_exit_long"] = (
            df["active_long_trade"] & (df["rolling_trend"].rolling(self.timed_exit_long_threshold.value).max().fillna(0) > 0.02)
        ).astype(int)

        df["timed_exit_short"] = (
            df["active_short_trade"] & (df["rolling_trend"].rolling(self.timed_exit_short_threshold.value).min().fillna(0) < -0.01)
        ).astype(int)

        strong_exit_long_conditions = [
            df["do_predict"] >= 0,
            df["&-s_target"] < (df["dynamic_exit_threshold"]),
            df["rolling_trend_scaled"] < (df["exit_trend_threshold"]),
            df["timed_exit_long"] | (df["vol_rank"] > 0.70),
            df["active_long_trade"],
            df["prediction_confidence"] > df["confidence_threshold"]
        ]

        strong_exit_short_conditions = [
            df["do_predict"] >= 0,
            df["&-s_target"] > (df["dynamic_exit_threshold"]),
            df["rolling_trend_scaled"] > (df["exit_trend_threshold"]),
            df["timed_exit_short"] | (df["vol_rank"] > 0.65),
            df["active_short_trade"],
            df["prediction_confidence"] > df["confidence_threshold"]
        ]

        df.loc[reduce(lambda x, y: x & y, strong_exit_long_conditions), ["exit_long", "exit_tag"]] = (1, "strong_exit_long")
        df.loc[reduce(lambda x, y: x & y, strong_exit_short_conditions), ["exit_short", "exit_tag"]] = (1, "strong_exit_short")

        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        prediction_confidence = last_candle.get("prediction_confidence", 0.5)

        atr_multiplier = self.atr_multiplier.value * (1.5 + historical_volatility if current_profit > 0.03 else
                                                    1.2 + historical_volatility if current_profit > 0.01 else
                                                    0.8 - prediction_confidence * 0.5 if current_profit < -0.02 else
                                                    1.0 + prediction_confidence * 0.3)

        stoploss_buffer = atr * atr_multiplier
        max_loss_pct = min(0.03 + historical_volatility, 0.06)

        dynamic_stoploss = current_rate + stoploss_buffer if trade.is_short else current_rate - stoploss_buffer

        return -max_loss_pct if (trade.is_short and current_rate > trade.open_rate * (1 + max_loss_pct)) or \
                            (not trade.is_short and current_rate < trade.open_rate * (1 - max_loss_pct)) else dynamic_stoploss
    
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

        stake_amount = max_risk / (atr * leverage) if atr > 0 else max_risk
        stake_amount = min(stake_amount, max_stake, proposed_stake)
        if min_stake and stake_amount < min_stake:
            stake_amount = min_stake

        return stake_amount


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


    def compute_prediction_metrics(self, dataframe: pd.DataFrame, metadata: dict, label_col: str= "T", prediction_col: str = "&-s_target") -> pd.DataFrame: 
        """
        Computes and stores prediction accuracy metrics for all trading pairs.
        Saves the results to a CSV file after backtesting.
        """
        prediction_mean = prediction_col + "_mean"
        prediction_std = prediction_col + "_std"

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

    def remove_highly_correlated_features(self, dataframe: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
        """
        Removes features that are highly correlated with each other.
        """

        # ✅ Ensure only numeric columns are used for correlation calculation
        numeric_df = dataframe.select_dtypes(include=[np.number])

        # ✅ Compute absolute correlation matrix
        corr_matrix = numeric_df.corr().abs()

        # ✅ Identify upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # ✅ Find features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        logger.info(f"🔍 Removing {len(to_drop)} highly correlated features: {to_drop}")

        # ✅ Drop correlated columns from original dataframe
        dataframe.drop(columns=to_drop, inplace=True, errors="ignore")

        return dataframe
          
    def filter_important_features(self, dataframe):
        """
        Removes all columns that start with '%' unless they are in the important features list.
        """
        important_features = {
            "%-hour_of_day",
            "%-day_of_week",
            "%-cci-period_50_BTC/USDTUSDT_4h",
            "%-pct-change_gen_BTC/USDTUSDT_1h",
            "%-roc-period_20_BTC/USDTUSDT_2h",
            "%-rsi-period_10_BTC/USDTUSDT_4h",
            "%-rsi-period_50_ETH/USDTUSDT_4h"
            # "%-bb_width-period_50_BTC/USDTUSDT_4h"
        }

        # Drop all columns starting with '%' unless they are in the important_features set
        columns_to_keep = [col for col in dataframe.columns if not col.startswith("%") or col in important_features]
        
        return dataframe[columns_to_keep]