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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.freqai_info = self.config["freqai"]

        # ✅ Ensure ATR is calculated
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14).bfill()

        # ✅ Compute Rolling Volume Rank
        dataframe["vol_rank"] = dataframe["volume"].rolling(24).rank(pct=True).fillna(0)

        # ✅ Compute Rolling Trend Indicator
        dataframe["rolling_trend"] = dataframe["close"].pct_change(12).rolling(6).mean().fillna(0)

        # ✅ Standardize ATR
        atr_window = 100
        atr_min = dataframe["atr"].rolling(100, min_periods=10).min()  # Allow earlier calculations
        atr_max = dataframe["atr"].rolling(100, min_periods=10).max()
        dataframe["atr_scaled"] = (dataframe["atr"] - atr_min) / (atr_max - atr_min + 1e-6)
        dataframe["atr_scaled"] = dataframe["atr_scaled"].fillna(method="bfill").clip(0.05, 1)  # Ensure no NaNs


        # ✅ Standardize Rolling Trend (Keep Negative Values)
        trend_window = 100
        mean_trend = dataframe["rolling_trend"].rolling(trend_window).mean()
        std_trend = dataframe["rolling_trend"].rolling(trend_window).std()
        dataframe["rolling_trend_scaled"] = (dataframe["rolling_trend"] - mean_trend) / (std_trend + 1e-6)

        dataframe = self.freqai.start(dataframe, metadata, self)          

        # ✅ Compute dynamic thresholds once (to be used in trade logic)
        dataframe["dynamic_long_threshold"] = dataframe["&-s_target_mean"] + dataframe["&-s_target_std"] * dataframe["atr_scaled"]
        dataframe["dynamic_short_threshold"] = dataframe["&-s_target_mean"] - dataframe["&-s_target_std"] * dataframe["atr_scaled"]
        dataframe["confidence_threshold"] = 0.25 + dataframe["atr_scaled"] * 0.20 
        dataframe["rolling_trend_threshold"] = dataframe["rolling_trend_scaled"].rolling(100, min_periods=10).median() * 0.35
        dataframe["dynamic_exit_threshold"] = (
            dataframe["&-s_target"].ewm(span=50).mean() +
            dataframe["atr_scaled"] * dataframe["&-s_target_std"] * (0.6 + dataframe["vol_rank"] * 0.3)
        )
        dataframe["exit_trend_threshold"] = dataframe["rolling_trend_scaled"].rolling(50).median() * 0.35

        """
        ✅ Keeping `T` for Plotting Purposes Only (Not To Be Used in Trade Logic Because of Lookahead Bias!)
        """
        dataframe["T"] = self.create_target_T(dataframe)
        dataframe["Prediction"] = dataframe["&-s_target"]
        dataframe["Avg Prediction"] = dataframe["&-s_target_mean"]
        dataframe["True Label"] = dataframe["T"]

        self.compute_prediction_metrics(dataframe, metadata)
        self.save_prediction_metrics()

        # ✅ Save dataframe for debugging
        dataframe.to_csv("./user_data/debug_data.csv", index=False)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # ✅ Ensure `enter_short` and `enter_long` columns exist before assignment
        df["enter_short"] = 0
        df["enter_long"] = 0

        df["valid_volume"] = df["vol_rank"] > 0.10  # More permissive

        enter_long_conditions = [
            df["do_predict"] == 1,
            (df["&-s_target"] > df["dynamic_long_threshold"]) & (df["rolling_trend_scaled"] > df["rolling_trend_threshold"]),
            df["vol_rank"] > 0.10,
            df["prediction_confidence"] > (df["confidence_threshold"] * 0.6)
        ]

        enter_short_conditions = [
            df["do_predict"] == 1,
            (df["&-s_target"] < df["dynamic_short_threshold"]) & (df["rolling_trend_scaled"] < df["rolling_trend_threshold"]),
            df["vol_rank"] > 0.10,
            df["prediction_confidence"] > (df["confidence_threshold"] * 0.6) 
        ]

        df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (1, "long")
        df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines exit conditions for long and short trades, ensuring exits actually trigger.
        """
        confidence_threshold = 0.35  # Lowered to allow more exits

        # ✅ Ensure exit signals exist
        df["exit_short"] = 0
        df["exit_long"] = 0

        # ✅ Fix Active Trade Tracking
        df["active_short_trade"] = (df["enter_short"].cumsum() - df["exit_short"].cumsum()) > 0
        df["active_long_trade"] = (df["enter_long"].cumsum() - df["exit_long"].cumsum()) > 0

        df["timed_exit_long"] = (
            df["active_long_trade"] &
            (df["rolling_trend"].rolling(30).max().fillna(0) > 0.03)  # Reduce threshold slightly
        ).astype(int)

        df["timed_exit_short"] = (
            df["active_short_trade"] &
            (df["rolling_trend"].rolling(20).min().fillna(0) < -0.02)
        ).astype(int)

        strong_exit_long_conditions = [
            df["do_predict"] >= 0,
            df["&-s_target"] < df["dynamic_exit_threshold"],
            df["rolling_trend_scaled"] < (df["exit_trend_threshold"] * 1.1),  # Make exits more reactive
            df["timed_exit_long"] | (df["vol_rank"] > 0.75),
            df["active_long_trade"],
            df["prediction_confidence"] > df["confidence_threshold"]
        ]

        strong_exit_short_conditions = [
            df["do_predict"] >= 0,
            df["&-s_target"] > df["dynamic_exit_threshold"],
            df["rolling_trend_scaled"] < (df["exit_trend_threshold"] * 0.9),  # Make exits more reactive
            df["timed_exit_short"] | (df["vol_rank"] > 0.75),
            df["active_short_trade"],
            df["prediction_confidence"] > df["confidence_threshold"]
        ]

        df.loc[reduce(lambda x, y: x & y, strong_exit_long_conditions), ["exit_long", "exit_tag"]] = (1, "strong_exit_long")
        df.loc[reduce(lambda x, y: x & y, strong_exit_short_conditions), ["exit_short", "exit_tag"]] = (1, "strong_exit_short")

        # ✅ Select only useful columns (price data, trade metrics, and signals)
        cols_to_keep = [
            "date", "open", "high", "low", "close", "volume",
            "do_predict", "&-s_target", "&-s_target_mean", "rolling_trend_scaled", "atr_scaled",
            "vol_rank", "prediction_confidence", "confidence_threshold", "dynamic_long_threshold", "dynamic_short_threshold",
            "dynamic_exit_threshold", "exit_trend_threshold",
            "enter_long", "enter_short", "exit_long", "exit_short"
        ]

        # filtered_dataframe = df[[col for col in cols_to_keep if col in df.columns]]
        filtered_dataframe = df[cols_to_keep].copy()
        # ✅ Save final DataFrame containing all trade signals
        filtered_dataframe.to_csv("./user_data/final_trading_data.csv", mode='a', header=not os.path.exists("./user_data/final_trading_data.csv"), index=False)
        logger.info("✅ Final trading data saved to `final_trading_data.csv`")

        return df

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

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Dynamically adjusts stoploss based on ATR, market volatility, and max risk per trade.
        Ensures correct differentiation between long and short trades.
        """

        # ✅ Load dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss  # Fallback to strategy-defined stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01

        # ✅ Compute dynamic ATR multiplier based on profit & market conditions
        base_atr_multiplier = 2.0  # Default ATR multiplier
        atr_multiplier = (
            base_atr_multiplier * 1.5 if current_profit > 0.03 else
            base_atr_multiplier * 1.2 if current_profit > 0.01 else
            base_atr_multiplier * 0.8 if current_profit < -0.02 else
            base_atr_multiplier
        )

        # ✅ Compute stoploss buffer
        stoploss_buffer = atr * atr_multiplier

        # ✅ **Differentiate Between Long and Short Trades**
        if trade.is_short:
            # **For SHORT trades:** Stoploss is ABOVE the entry price (buy to close)
            dynamic_stoploss = current_rate + stoploss_buffer
            max_loss_price = trade.open_rate * (1 + min(0.03 + historical_volatility, 0.06))  # Cap at 6% max loss
            if current_rate > max_loss_price:
                return -min(0.03 + historical_volatility, 0.06)  # **Force exit**

            # ✅ **Short trades should have a stricter max duration**
            max_trade_duration = timedelta(days=1.5)  # **Max 1.5 days for shorts**
            force_exit_loss = -0.004  # **Force short trade exit at -0.4% loss after max duration**

        else:
            # **For LONG trades:** Stoploss is BELOW the entry price (sell to close)
            dynamic_stoploss = current_rate - stoploss_buffer
            max_loss_price = trade.open_rate * (1 - min(0.03 + historical_volatility, 0.06))  # Cap at 6% max loss
            if current_rate < max_loss_price:
                return -min(0.03 + historical_volatility, 0.06)  # **Force exit**

            # ✅ **Long trades may have slightly more room**
            max_trade_duration = timedelta(days=2.5)  # **Max 2.5 days for longs**
            force_exit_loss = -0.005  # **Force long trade exit at -0.5% loss after max duration**

        # ✅ **Force exit if trade exceeds max duration**
        if (current_time - trade.open_date_utc) > max_trade_duration:
            return force_exit_loss  # **Apply different exit loss for longs vs. shorts**

        # ✅ Store stoploss in dataframe for tracking
        if "stoploss" not in dataframe.columns:
            dataframe["stoploss"] = np.nan
        dataframe.at[last_candle.name, "stoploss"] = dynamic_stoploss

        return dynamic_stoploss
    
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Dynamically determines position size based on account balance, ATR, and market conditions.
        """

        # ✅ Load latest market data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return proposed_stake  # Use default stake if no data

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get('atr', 0)
        historical_volatility = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01

        # ✅ Compute max risk per trade dynamically (adjusting for market volatility)
        base_risk = 0.02  # Base risk: 2% per trade
        adjusted_risk = base_risk * (1 + historical_volatility)  # Adjust risk based on volatility

        max_risk = max_stake * adjusted_risk  

        # ✅ ATR-based position sizing
        if atr > 0:
            stake_amount = max_risk / (atr * leverage)  # **Adjust stake based on leverage**
        else:
            stake_amount = max_risk  # Fallback if ATR is zero

        # ✅ Ensure stake does not exceed available balance or max_stake
        stake_amount = min(stake_amount, max_stake, proposed_stake)

        # ✅ Ensure stake meets min_stake requirement
        if min_stake and stake_amount < min_stake:
            stake_amount = min_stake

        return stake_amount

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, 
                            current_time, entry_tag, side: str, **kwargs) -> bool:
        """
        Dynamically adjusts trade size based on prediction confidence, 
        while ensuring it stays within defined risk limits.
        """

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1]

        # ✅ Get prediction confidence (fallback to 0.5 if missing)
        confidence = last_candle.get("prediction_confidence", 0.5)

        # ✅ Apply dynamic scaling to trade size
        min_trade_size = amount * 0.5  # Ensure at least 50% of the original trade size
        max_trade_size = amount * 1.5  # Prevent exceeding 150% of the original trade size

        adjusted_size = amount * confidence
        adjusted_size = max(min_trade_size, min(adjusted_size, max_trade_size))  # Ensure within bounds

        # ✅ Log trade confirmation details
        logger.info(f"🚀 Confirming trade entry | Pair: {pair} | Confidence: {confidence:.2f} | Adjusted Size: {adjusted_size:.4f}")

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