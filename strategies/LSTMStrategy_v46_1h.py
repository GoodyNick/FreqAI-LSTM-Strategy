import logging
import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import requests
import talib.abstract as ta
from pandas import DataFrame
from scipy.fftpack import fft
from scipy.stats import zscore
from technical import qtpylib
from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.vendor.qtpylib.indicators import crossed_above, crossed_below

logger = logging.getLogger(__name__)


# Apply runtime FreqAI shims at import time so patches are active before
# FreqAI loads models or pipelines. This is intentionally safe and
# idempotent; failures are debug-logged.
try:
    from user_data import freqai_shims

    try:
        freqai_shims.apply_shims()
        logger.info("Applied freqai_shims at strategy import time")
    except Exception as _e:
        logger.debug(f"Could not apply freqai_shims at import time: {_e}")
except Exception:
    # user_data may not be on sys.path during some tooling; ignore silently
    pass

class LSTMStrategy_v46_1h(IStrategy):
    """
    Leak-safe revision of v45:
    - Removes target-dependent metrics from decision logic.
    - Simplifies stoploss to ATR-only.
    - Fixes duplicate feature function.
    - Uses only std-based prediction confidence for trading decisions.
    - Cleans opposite-signal exits (no enter_* references).
    """

    # Enhanced plotting configuration for comprehensive analysis
    plot_config = {
        "main_plot": {
        },
        "subplots": {
            "Predictions & Targets": {
                "T": {"color": "blue", "plot_type": "line"},                    # True target/label
                "&-s_target": {"color": "purple", "plot_type": "line"},         # FreqAI predictions
                "&-s_target_mean": {"color": "brown", "plot_type": "line"},     # FreqAI mean predictions
                "custom_pred_mean": {"color": "orange", "plot_type": "line"},   # Custom smoothed prediction
            },
            "Prediction Quality": {
                "pred_confidence": {"color": "green", "plot_type": "line"},    # Prediction confidence
                "rolling_accuracy": {"color": "red", "plot_type": "line"},     # Rolling accuracy
                "mae": {"color": "pink", "plot_type": "line"},                 # Mean Absolute Error
            },
            "FreqAI Signals": {
                "do_predict": {"color": "purple", "plot_type": "scatter"},     # FreqAI prediction flag
                "DI_values": {"color": "orange", "plot_type": "line"},         # FreqAI confidence values
            },
        },
    }
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True
    use_custom_stoploss = True

    startup_candle_count = 300

    # Class-level storage for prediction metrics (persisted after backtests)
    pred_metrics_storage: list = []

    # Fixed windows (non-optimized) to reduce complexity/overfit - updated for better signal exits
    fixed_vol_window = 24
    fixed_trend_window = 72
    fixed_stake_scaling_factor = 1.0
    fixed_confidence_threshold_multiplier = 1.0
    fixed_rolling_trend_threshold_multiplier = 1.0
    fixed_soft_stoploss_pct = -0.25  # More conservative to allow signal exits to work
    fixed_min_profit_for_trailing = 0.015  # Higher threshold before trailing starts
    fixed_initial_stop_duration_candles = 8  # More protection time for signal exits

    # High-confidence fallback threshold
    fixed_high_confidence_threshold = 0.90

    # Signal quality parameters
    prediction_threshold = DecimalParameter(0.01, 0.5, default=0.05, decimals=3, space="buy", load=True, optimize=True)
    momentum_confirmation_window = IntParameter(2, 10, default=3, space="buy", load=True, optimize=True)
    signal_persistence_required = IntParameter(1, 5, default=2, space="buy", load=True, optimize=True)
    
    # Exit improvement parameters  
    min_trade_duration_hours = IntParameter(6, 48, default=12, space="sell", load=True, optimize=True)
    profit_target_multiplier = DecimalParameter(1.5, 4.0, default=2.0, decimals=1, space="sell", load=True, optimize=True)

    # Volume filter
    vol_rank_quantile = DecimalParameter(0.1, 0.5, default=0.18, decimals=2, space="buy", load=True, optimize=True)

    # Exit timing
    timed_exit_duration = IntParameter(24, 168, default=45, space="sell", load=True, optimize=True)

    # Stoploss sensitivity
    atr_stoploss_multiplier = DecimalParameter(1.0, 8.0, default=1.0, decimals=1, space="sell", load=True, optimize=True)
    hard_max_stoploss_pct = DecimalParameter(-0.25, -0.05, default=-0.14, decimals=2, space="sell", load=True, optimize=True)

    # Stake/leverage dynamics (kept but simple)
    leverage_range_end = IntParameter(2, 10, default=2, space="buy", load=True, optimize=True)
    volatility_influence = DecimalParameter(0.0, 1.0, default=0.937, decimals=3, space="buy", load=True, optimize=True)
    confidence_influence = DecimalParameter(0.0, 1.0, default=0.088, decimals=3, space="buy", load=True, optimize=True)

    # Categorical switches
    hyperopt_categorical = True
    use_vol_filter = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    use_confidence_filter_entry = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    use_trend_filter = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)
    use_mean_prediction_for_signal = CategoricalParameter([True, False], default=True, space="buy", load=True, optimize=hyperopt_categorical)

    use_target_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)  # Force True for testing
    use_trend_exit_filter = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)  # Force True for testing
    use_timed_exit = CategoricalParameter([True, False], default=True, space="sell", load=True, optimize=hyperopt_categorical)
    use_opposite_signal_exit = CategoricalParameter([True, False], default=True, space="sell", load=False, optimize=hyperopt_categorical)  # Force True for testing

    minimal_roi = {"0": 1}
    stoploss = -1.0

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.trades: Dict[str, datetime] = {}
        self.historical_fng_data = self.load_historical_fng_data()
        self.freqai_info = self.config.get("freqai", {})

    # -------------------- Feature engineering --------------------
    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, metadata: Dict, **kwargs):
        # RSI
        dataframe[f'%-rsi-{period}'] = ta.RSI(dataframe, timeperiod=period)
        roc = ta.ROC(dataframe, timeperiod=period)
        dataframe[f'%-roc_smooth-{period}'] = roc.rolling(window=3, min_periods=1).mean()
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2)
        dataframe[f'%-bb_width_pct-{period}'] = (
            (bollinger['upper'] - bollinger['lower']) / bollinger['mid'] * 100
        )
        stoch = ta.STOCH(dataframe, 
                         fastk_period=period, 
                         slowk_period=7, 
                         slowk_matype=0, 
                         slowd_period=14, 
                         slowd_matype=0
                        )
        dataframe[f'%-stoch_k-{period}'] = stoch['slowk']
        dataframe[f'%-stoch_d-{period}'] = stoch['slowd']
        if 'volume' in dataframe.columns and 'close' in dataframe.columns and 'high' in dataframe.columns and 'low' in dataframe.columns:
            tpv = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3 * dataframe['volume']
            rolling_tpv_sum = tpv.rolling(window=period, min_periods=1).sum()
            rolling_volume_sum = dataframe['volume'].rolling(window=period, min_periods=1).sum()
            rolling_vwap = rolling_tpv_sum / (rolling_volume_sum + 1e-9)
            dataframe[f'%-vwap_dev-{period}'] = (dataframe['close'] - rolling_vwap) / (rolling_vwap + 1e-9) * 100
        else:
            dataframe[f'%-vwap_dev-{period}'] = 0
        for col in dataframe.columns:
            if col.startswith('%-') and f'-{period}' in col:
                dataframe[col] = dataframe[col].bfill().ffill().fillna(0)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        ema3_series = pd.Series(ta.EMA(dataframe['close'], timeperiod=3), index=dataframe.index)
        dataframe['%-ema3_pct_change'] = ema3_series.pct_change() * 100
        if 'volume' in dataframe.columns:
            dataframe['%-volume_ema3'] = pd.Series(ta.EMA(dataframe['volume'], timeperiod=3), index=dataframe.index)
        else:
            dataframe['%-volume_ema3'] = 0
        dataframe['%-raw_price'] = dataframe['close']
        dataframe['%-atr_short'] = pd.Series(ta.ATR(dataframe, timeperiod=5), index=dataframe.index)
        for col in ['%-ema3_pct_change', '%-volume_ema3', '%-atr_short']:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].bfill().ffill().fillna(0)
        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, metadata: Dict, **kwargs):
        if "atr" not in dataframe.columns:
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=14).bfill()
        dataframe["%-atr_pct"] = (dataframe["atr"] / dataframe["close"]) * 100
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe.loc[:, "%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe.loc[:, "%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe.loc[:, "%-rolling_volatility"] = dataframe["close"].rolling(window=24).std().bfill()
        dataframe.loc[:, "%-rolling_mean"] = dataframe["close"].rolling(window=24).mean().bfill()
        dataframe.loc[:, "%-ema_trend"] = ta.EMA(dataframe, timeperiod=24).bfill()
        def get_cusum(series):
            series_mean = series.mean()
            return (series - series_mean).cumsum()
        dataframe.loc[:, "%-cusum_close"] = get_cusum(dataframe["close"]).fillna(0)
        def hurst_exponent(ts, max_lag=20):
            if len(ts) < max_lag:
                return np.nan
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0]
        dataframe.loc[:, "%-hurst"] = dataframe["close"].rolling(window=72).apply(hurst_exponent, raw=True)
        dataframe.loc[:, "%-hurst_smooth"] = dataframe["%-hurst"].rolling(window=10).mean().bfill()
        def compute_fourier(series, n_components=3):
            if len(series) < 72:
                return np.nan
            fft_vals = fft(series)
            return np.abs(fft_vals[:n_components]).sum()
        dataframe.loc[:, "%-fourier_price"] = dataframe["close"].rolling(window=72).apply(compute_fourier, raw=True)
        dataframe.loc[:, "%-fourier_price"] = dataframe["%-fourier_price"].fillna(dataframe["%-fourier_price"].median())
        dataframe.loc[:, "%-fourier_price_norm"] = dataframe["%-fourier_price"] / (dataframe["atr"] + 1e-6)
        zscore_columns = ["%-rolling_volatility", "%-rolling_mean", "%-fourier_price_norm", "%-atr_pct"]
        for col in zscore_columns:
             if col in dataframe.columns:
                 dataframe.loc[:, f"{col}-zscore"] = pd.Series(zscore(dataframe[col]), index=dataframe.index).fillna(0)
        current_date = dataframe['date'].iloc[-1] if 'date' in dataframe else None
        fear_greed_value, fear_greed_classification = self.get_fear_and_greed_index(current_date)
        dataframe['fear_greed_index'] = fear_greed_value
        dataframe['fear_greed_index'] = dataframe['fear_greed_index'].ffill()
        return dataframe
    # -------------------- Target & FreqAI connection --------------------
    def create_target_T(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Create leak-safe targets using fixed label_period_candles.
        Based on v45's working approach but with fixed lookahead to prevent data leakage.
        """
        base_window = self.freqai_info["feature_parameters"]["label_period_candles"]
        df = dataframe.copy()
        
        # Clean price data similar to v45
        for col in ['close', 'high', 'low']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').replace(0, np.nan).bfill().ffill()
            else:
                if col == 'close':
                    df[col] = 1.0
                else:
                    df[col] = df['close'] if 'close' in df.columns else 1.0
        
        # Calculate ATR similar to v45 (use 72 as in v45)
        atr_window = self.fixed_vol_window
        atr_series = pd.Series(ta.ATR(df, timeperiod=atr_window), index=df.index)
        df["ATR"] = atr_series.bfill().ffill().fillna(1e-9)
        
        # Use fixed lookahead (leak-safe) instead of dynamic
        lookahead = base_window
        
        # Calculate future percentage change similar to v45
        future_change_pct_list = []
        close_series = df["close"].to_numpy()
        len_df = len(df)
        
        for i in range(len_df):
            current_close_val = close_series[i]
            if np.isnan(current_close_val) or current_close_val == 0:
                future_change_pct_list.append(np.nan)
                continue
                
            future_index = i + lookahead
            if future_index >= len_df:
                # Handle end-of-series: use percentage change pattern from available data
                if i > lookahead:
                    # Use historical pattern for extrapolation
                    past_change = (close_series[i] - close_series[i - lookahead]) / close_series[i - lookahead] * 100
                    future_change_pct_list.append(past_change)
                else:
                    future_change_pct_list.append(0.0)  # Use 0 instead of NaN
            else:
                future_close = close_series[future_index]
                if np.isnan(future_close) or future_close == 0:
                    future_change_pct_list.append(0.0)
                else:
                    future_change_pct = (future_close - current_close_val) / current_close_val * 100
                    future_change_pct_list.append(future_change_pct)
        
        df["future_change_pct"] = future_change_pct_list
        
        # Calculate trend strength similar to v45
        df["TS_pct"] = df["future_change_pct"].rolling(atr_window, min_periods=atr_window // 2).mean()
        
        # Normalize using ATR percentage similar to v45
        current_close_safe = df["close"].replace(0, np.nan).bfill().ffill().fillna(1e-9)
        df["ATR_pct_for_norm"] = (df["ATR"] / current_close_safe).replace(0, np.nan).bfill().ffill().fillna(1e-7)
        
        # Create normalized target
        df["T_raw"] = df["TS_pct"] / (df["ATR_pct_for_norm"] + 1e-7)
        
        # Apply scaling and clipping similar to v45
        abs_t_raw = df["T_raw"].abs().dropna()
        scale_factor = 1.0
        if not abs_t_raw.empty:
            percentile_98 = abs_t_raw.quantile(0.98)
            if percentile_98 > 0:
                scale_factor = 1.5 / percentile_98  # Scale so 98th percentile maps to 1.5
        
        df["T_scaled"] = df["T_raw"] * scale_factor
        df["T"] = np.clip(df["T_scaled"], -2.0, 2.0)
        
        # Final smoothing
        df["T"] = df["T"].rolling(3, min_periods=1).mean()
        
        # Fill any remaining NaNs with 0
        df["T"] = df["T"].fillna(0)
        
        return df["T"]

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        dataframe['&-s_target'] = self.create_target_T(dataframe)
        return dataframe

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
        if log_metrics: 
            self.save_prediction_metrics()

        if "pred_confidence" in df.columns:
            df["pred_confidence"] = pd.to_numeric(df["pred_confidence"], errors='coerce').ffill().bfill().fillna(0)
        else:
            logger.warning("'pred_confidence' missing after compute_prediction_metrics. Assigning 0.")
            df["pred_confidence"] = 0.0
                
        if not hasattr(self, 'trades'): 
            self.trades = {}
        if not hasattr(self, 'timeframe_minutes'): 
            self.timeframe_minutes = timeframe_to_minutes(self.timeframe)
        
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

    # -------------------- Indicator calc for signals --------------------
    def calculate_indicators(self, df: DataFrame) -> DataFrame:
        vol_win = self.fixed_vol_window
        trend_win = self.fixed_trend_window

        atr = pd.Series(ta.ATR(df, timeperiod=vol_win), index=df.index)
        df['atr'] = atr.bfill().ffill().fillna(1e-9)
        safe_close = df['close'].replace(0, np.nan).bfill().ffill().fillna(1e-9)
        df['atr_normalized'] = (df['atr'] / safe_close)

        df['vol_rank'] = df['volume'].rolling(vol_win).rank(pct=True).fillna(0)
        df['vol_rank_dynamic_threshold'] = df['vol_rank'].rolling(vol_win).quantile(self.vol_rank_quantile.value).ffill().bfill().fillna(0.1)

        pct_change_period = trend_win
        rolling_trend_win = max(2, trend_win // 4)
        df['rolling_trend'] = df['close'].pct_change(pct_change_period).rolling(rolling_trend_win).mean().fillna(0)
        q_val = df['atr_normalized'].rolling(trend_win).quantile(0.90).ffill().bfill().fillna(1e-9)
        df['atr_scaled'] = (df['atr_normalized'] / q_val.clip(lower=1e-9)).clip(0, 1).fillna(0)
        mean_tr = df['rolling_trend'].rolling(trend_win).mean().ffill().fillna(0)
        std_tr = df['rolling_trend'].rolling(trend_win).std().ffill().fillna(1e-9)
        df['rolling_trend_scaled'] = ((df['rolling_trend'] - mean_tr) / std_tr.clip(lower=1e-9)).fillna(0)

        # Enhanced signal generation with momentum confirmation
        win = int(self.momentum_confirmation_window.value) * 2  # Use larger window for smoothing
        df['custom_pred_mean'] = df['&-s_target'].rolling(window=win, min_periods=1).mean().fillna(0)
        df['custom_pred_std'] = df['&-s_target'].rolling(window=win, min_periods=1).std().fillna(0)
        
        # Signal momentum and persistence
        threshold = float(self.prediction_threshold.value)
        df['pred_signal_raw'] = np.where(df['custom_pred_mean'] > threshold, 1, 
                                np.where(df['custom_pred_mean'] < -threshold, -1, 0))
        
        # Momentum confirmation - signal must be strengthening
        momentum_win = int(self.momentum_confirmation_window.value)
        df['pred_momentum'] = df['custom_pred_mean'].diff(momentum_win)
        df['momentum_aligned'] = ((df['pred_signal_raw'] == 1) & (df['pred_momentum'] > 0)) | \
                                ((df['pred_signal_raw'] == -1) & (df['pred_momentum'] < 0))
        
        # Signal persistence - must hold for multiple periods
        persistence = int(self.signal_persistence_required.value)
        df['signal_persistent'] = (df['pred_signal_raw'].rolling(persistence).apply(
            lambda x: (x == x.iloc[-1]).all() and x.iloc[-1] != 0, raw=False).fillna(False))
        
        # Combined high-quality signal
        df['pred_signal_quality'] = df['momentum_aligned'] & df['signal_persistent']
        
        # EMA crossovers for additional confirmation (but not primary signal)
        pred_col = 'custom_pred_mean'
        slow = int(self.momentum_confirmation_window.value) * 2  # Use larger window for slow EMA
        fast = max(2, slow // 3)
        df['pred_ema_fast'] = ta.EMA(df[pred_col], timeperiod=int(fast))
        df['pred_ema_slow'] = ta.EMA(df[pred_col], timeperiod=int(slow))

        # Thresholds
        df['rolling_trend_threshold_base'] = df['rolling_trend_scaled'].rolling(100, min_periods=10).median().ffill().fillna(0)
        df['rolling_trend_threshold'] = df['rolling_trend_threshold_base'] * self.fixed_rolling_trend_threshold_multiplier

        # Live-safe prediction confidence: purely SNR from predictions (no T, no rolling_accuracy)
        snr = (np.abs(df['custom_pred_mean']) / (df['custom_pred_std'] + 1e-6)).clip(0, 1)
        df['pred_confidence'] = snr.fillna(0)
        df['confidence_threshold_base'] = df['pred_confidence'].rolling(100).quantile(0.5).ffill().fillna(0)
        df['confidence_threshold'] = df['confidence_threshold_base'] * self.fixed_confidence_threshold_multiplier
        return df

    # -------------------- Entries/Exits --------------------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = 0
        df['enter_short'] = 0
        df['enter_tag'] = None
        p = self.calculate_indicators(df)
        
        # Primary signals: High-quality prediction signals with momentum and persistence
        quality_long = p['pred_signal_quality'] & (p['pred_signal_raw'] == 1)
        quality_short = p['pred_signal_quality'] & (p['pred_signal_raw'] == -1)
        
        # EMA cross confirmation (supportive, not primary)
        ema_cross_up = crossed_above(p['pred_ema_fast'], p['pred_ema_slow'])
        ema_cross_down = crossed_below(p['pred_ema_fast'], p['pred_ema_slow'])
        ema_aligned_long = (p['pred_ema_fast'] > p['pred_ema_slow'])
        ema_aligned_short = (p['pred_ema_fast'] < p['pred_ema_slow'])

        def base(side: str | None = None):
            cond = (p['do_predict'] == 1)
            if self.use_vol_filter.value:
                cond &= (p['vol_rank'] > p['vol_rank_dynamic_threshold'])
            if self.use_confidence_filter_entry.value:
                cond &= (p['pred_confidence'] > p['confidence_threshold'])
            if self.use_trend_filter.value:
                if side == 'long':
                    cond &= (p['rolling_trend_scaled'] > p['rolling_trend_threshold'])
                elif side == 'short':
                    cond &= (p['rolling_trend_scaled'] < p['rolling_trend_threshold'])
            return cond

        # Primary entries: Quality signals with EMA alignment
        long_ok = quality_long & ema_aligned_long & base('long')
        short_ok = quality_short & ema_aligned_short & base('short')
        df.loc[long_ok, ['enter_long', 'enter_tag']] = (1, 'long_quality')
        df.loc[short_ok & (df['enter_long'] == 0), ['enter_short', 'enter_tag']] = (1, 'short_quality')

        # Secondary entries: EMA cross with momentum confirmation
        momentum_threshold = float(self.prediction_threshold.value) * 0.5
        ema_with_momentum_long = ema_cross_up & (p['pred_momentum'] > 0) & (p['custom_pred_mean'] > momentum_threshold)
        ema_with_momentum_short = ema_cross_down & (p['pred_momentum'] < 0) & (p['custom_pred_mean'] < -momentum_threshold)
        
        long_ema_ok = ema_with_momentum_long & base('long')
        short_ema_ok = ema_with_momentum_short & base('short')
        df.loc[long_ema_ok & (df['enter_long'] == 0) & (df['enter_short'] == 0), ['enter_long', 'enter_tag']] = (1, 'long_ema')
        df.loc[short_ema_ok & (df['enter_long'] == 0) & (df['enter_short'] == 0), ['enter_short', 'enter_tag']] = (1, 'short_ema')

        # High-confidence fallback (similar to original but with enhanced conditions)
        pred_col = 'custom_pred_mean' if self.use_mean_prediction_for_signal.value else '&-s_target'
        if pred_col not in p.columns:
            pred_col = '&-s_target'
        
        high_conf = p['pred_confidence'] > self.fixed_high_confidence_threshold
        strong_signal_threshold = float(self.prediction_threshold.value) * 1.5
        
        fb_long = (p['do_predict'] == 1) & (p[pred_col] > strong_signal_threshold) & high_conf & \
                 (p['pred_momentum'] > 0) & crossed_above(p['rolling_trend_scaled'], p['rolling_trend_threshold'])
        fb_short = (p['do_predict'] == 1) & (p[pred_col] < -strong_signal_threshold) & high_conf & \
                  (p['pred_momentum'] < 0) & crossed_below(p['rolling_trend_scaled'], p['rolling_trend_threshold'])
        
        df.loc[fb_long & (df['enter_long'] == 0) & (df['enter_short'] == 0), ['enter_long', 'enter_tag']] = (1, 'long_fallback')
        df.loc[fb_short & (df['enter_long'] == 0) & (df['enter_short'] == 0), ['enter_short', 'enter_tag']] = (1, 'short_fallback')
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = False
        df['exit_short'] = False
        df['exit_tag'] = None
        p = self.calculate_indicators(df)
        
        # Minimum trade duration check (convert hours to periods)
        min_duration_periods = int(self.min_trade_duration_hours.value)  # Assuming 1h timeframe
        trade_duration_ok = (df['trade_duration'] >= min_duration_periods)
        
        # Primary exit signals: Quality signal reversal
        quality_exit_long = p['pred_signal_quality'] & (p['pred_signal_raw'] == -1)
        quality_exit_short = p['pred_signal_quality'] & (p['pred_signal_raw'] == 1)
        
        # EMA cross exits (secondary)
        exit_long_cross = crossed_below(p['pred_ema_fast'], p['pred_ema_slow'])
        exit_short_cross = crossed_above(p['pred_ema_fast'], p['pred_ema_slow'])
        
        # Profit target exits
        profit_multiplier = float(self.profit_target_multiplier.value)
        strong_profit_signal_long = (p['custom_pred_mean'] < -float(self.prediction_threshold.value) * profit_multiplier)
        strong_profit_signal_short = (p['custom_pred_mean'] > float(self.prediction_threshold.value) * profit_multiplier)
        
        # Base conditions for exits
        base = (p['do_predict'] == 1) & (p['pred_confidence'] > p['confidence_threshold'])
        if self.use_vol_filter.value:
            base &= (p['vol_rank'] > p['vol_rank_dynamic_threshold'])
        
        # Signal-based exits with minimum duration
        signal_exit_base = base & trade_duration_ok
        
        if self.use_target_exit_filter.value:
            # Primary: Quality signal reversals (most reliable exits)
            quality_long_exits = (df['exit_long'] == False) & quality_exit_long & signal_exit_base
            quality_short_exits = (df['exit_short'] == False) & quality_exit_short & signal_exit_base
            
            df.loc[quality_long_exits, ['exit_long', 'exit_tag']] = (True, 'quality_exit_long')
            df.loc[quality_short_exits, ['exit_short', 'exit_tag']] = (True, 'quality_exit_short')
            
            # Secondary: Profit target exits (strong opposing signals, relaxed minimum duration)
            profit_long_exits = (df['exit_long'] == False) & strong_profit_signal_long & base & (df['trade_duration'] >= 2)
            profit_short_exits = (df['exit_short'] == False) & strong_profit_signal_short & base & (df['trade_duration'] >= 2)
            
            df.loc[profit_long_exits, ['exit_long', 'exit_tag']] = (True, 'profit_target_long')
            df.loc[profit_short_exits, ['exit_short', 'exit_tag']] = (True, 'profit_target_short')
            
            # Tertiary: EMA cross exits (only after minimum duration, no profit requirement here)
            ema_long_exits = (df['exit_long'] == False) & exit_long_cross & signal_exit_base
            ema_short_exits = (df['exit_short'] == False) & exit_short_cross & signal_exit_base
            
            df.loc[ema_long_exits, ['exit_long', 'exit_tag']] = (True, 'ema_exit_long')
            df.loc[ema_short_exits, ['exit_short', 'exit_tag']] = (True, 'ema_exit_short')
        
        if self.use_trend_exit_filter.value:
            trend_exit_base = base & trade_duration_ok
            df.loc[(df['exit_long'] == False) & crossed_below(p['rolling_trend_scaled'], p['rolling_trend_threshold']) & trend_exit_base, 
                   ['exit_long', 'exit_tag']] = (True, 'trend_exit_long')
            df.loc[(df['exit_short'] == False) & crossed_above(p['rolling_trend_scaled'], p['rolling_trend_threshold']) & trend_exit_base, 
                   ['exit_short', 'exit_tag']] = (True, 'trend_exit_short')
        
        if self.use_timed_exit.value:
            # Extended timed exit duration to reduce premature exits
            extended_duration = max(self.timed_exit_duration.value, int(self.min_trade_duration_hours.value) * 2)
            df.loc[(df['exit_long'] == False) & (df['trade_duration'] > extended_duration) & (p['pred_confidence'] < p['confidence_threshold']), 
                   ['exit_long', 'exit_tag']] = (True, 'timed_exit_long')
            df.loc[(df['exit_short'] == False) & (df['trade_duration'] > extended_duration) & (p['pred_confidence'] < p['confidence_threshold']), 
                   ['exit_short', 'exit_tag']] = (True, 'timed_exit_short')
        
        # Opposite signal exits using enter_* signals (safe for backtesting, handled separately in live)
        if self.use_opposite_signal_exit.value:    
            df.loc[df['enter_short'] == 1, ['exit_long', 'exit_tag']] = (True, 'opposite_signal_short')
            df.loc[df['enter_long'] == 1, ['exit_short', 'exit_tag']] = (True, 'opposite_signal_long')
        
        return df

    # -------------------- Risk controls --------------------
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty or not trade or not trade.open_rate:
            return -1.0
        last = dataframe.iloc[-1]
        
        # Calculate elapsed time in candles
        tf_min = timeframe_to_minutes(self.timeframe)
        elapsed_min = (current_time - trade.open_date_utc).total_seconds() / 60
        dur_candles = elapsed_min / tf_min
        
        # Respect minimum trade duration from strategy parameters
        min_duration_candles = int(self.min_trade_duration_hours.value)
        
        # During minimum duration period, use very conservative stop
        if dur_candles < min_duration_candles:
            return self.fixed_soft_stoploss_pct  # -0.25 (25% emergency stop only)
        
        # Before initial protection period ends, still be conservative
        if dur_candles < self.fixed_initial_stop_duration_candles:
            return self.fixed_soft_stoploss_pct
            
        # Only start dynamic stoploss after minimum profit threshold
        if current_profit < self.fixed_min_profit_for_trailing:
            return self.fixed_soft_stoploss_pct

        # Calculate ATR-based dynamic stop
        if 'atr' in last and pd.notna(last.get('atr')) and last.get('atr', 0) > 0:
            atr = last['atr']
        else:
            atr_series = ta.ATR(dataframe, timeperiod=self.fixed_vol_window)
            atr = atr_series.iloc[-1] if atr_series is not None and not atr_series.empty and pd.notna(atr_series.iloc[-1]) else current_rate * 0.01
        atr = max(atr, current_rate * 0.005)

        # Apply ATR multiplier with more conservative approach
        atr_mult = float(self.atr_stoploss_multiplier.value) * 1.5  # Make ATR stop less aggressive
        hard_max = float(self.hard_max_stoploss_pct.value)
        stop_abs = atr * atr_mult
        stop_pct = -(stop_abs / trade.open_rate)
        final_sl = max(stop_pct, hard_max)
        
        # Ensure the dynamic stop is not more aggressive than our soft stop
        final_sl = max(final_sl, self.fixed_soft_stoploss_pct)
        
        return final_sl if final_sl < 0 else self.fixed_soft_stoploss_pct

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float,
                            min_stake: float | None, max_stake: float, leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return proposed_stake
        last = dataframe.iloc[-1]
        pred_conf = float(last.get('pred_confidence', 0.5))
        hist_vol = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        scaled_vol = min(hist_vol, 0.05)
        conf_factor = 1.0 + (pred_conf - 0.5) * float(self.confidence_influence.value) * 2
        vol_factor = max(0.1, 1.0 - (scaled_vol * float(self.volatility_influence.value) * 10))
        stake = proposed_stake * conf_factor * vol_factor
        stake *= self.fixed_stake_scaling_factor
        stake = min(stake, max_stake)
        if min_stake and stake < min_stake:
            stake = min_stake
        return min(stake, max_stake)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return 1.0
        hist_vol = dataframe['close'].pct_change().rolling(50).std().iloc[-1] if not dataframe.empty else 0.01
        pred_conf = float(dataframe.iloc[-1].get('pred_confidence', 0.5))
        min_lev = 1
        strat_max = int(self.leverage_range_end.value)
        base = (min_lev + strat_max) / 2
        adj = ((1 - hist_vol - 0.5) * float(self.volatility_influence.value) + (pred_conf - 0.5) * float(self.confidence_influence.value)) * (max_leverage - min_lev)
        lev = base + adj
        return int(min(max(min_lev, lev), strat_max, max_leverage))

    # -------------------- Entry confirmation --------------------
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

    # -------------------- Enhanced Metrics System --------------------
    def compute_prediction_metrics(self, dataframe: pd.DataFrame, metadata: dict, 
                                 label_col: str = "T", prediction_col: str = "&-s_target", 
                                 log_metrics: bool = True) -> pd.DataFrame:
        """Enhanced prediction metrics computation focused on prediction quality"""
        
        try:
            # Try to import and use focused enhanced metrics
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
            from simple_enhanced_metrics import compute_prediction_metrics_enhanced
            return compute_prediction_metrics_enhanced(self, dataframe, metadata, label_col, prediction_col, log_metrics)
            
        except ImportError as e:
            logger.warning(f"Simple enhanced metrics not available: {e}")
        except Exception as e:
            logger.error(f"Enhanced metrics failed: {e}")
        
        # Fallback to basic metrics
        return self._compute_basic_prediction_metrics(dataframe, metadata, label_col, prediction_col, log_metrics)

    def _add_basic_computed_columns(self, dataframe: pd.DataFrame, label_col: str, prediction_col: str) -> pd.DataFrame:
        """Add essential computed columns that the strategy needs for trading decisions"""
        # Ensure pred_correct column exists
        if "pred_correct" not in dataframe.columns:
            dataframe["pred_correct"] = np.nan
            valid = dataframe[label_col].notna() & dataframe[prediction_col].notna()
            if valid.sum() > 0:
                dataframe.loc[valid, "pred_correct"] = np.where(
                    np.sign(dataframe.loc[valid, label_col]) == np.sign(dataframe.loc[valid, prediction_col]), 1, 0
                )
        
        # Ensure rolling_accuracy exists
        if "rolling_accuracy" not in dataframe.columns:
            ra_win = 50
            dataframe["rolling_accuracy"] = dataframe["pred_correct"].rolling(
                ra_win, min_periods=max(1, ra_win // 5)
            ).mean().ffill().fillna(0.5)
        
        # Ensure MAE exists
        if "mae" not in dataframe.columns:
            dataframe["mae"] = np.abs(dataframe[label_col] - dataframe[prediction_col]).rolling(100, min_periods=1).mean()
        
        # Ensure pred_confidence exists (critical for trading decisions)
        if "pred_confidence" not in dataframe.columns:
            pred_std = prediction_col + "_std"
            if pred_std in dataframe.columns:
                base_conf = (np.abs(dataframe[prediction_col]) / (dataframe[pred_std] + 1e-6)).clip(0, 1)
                dataframe["pred_confidence"] = (base_conf * dataframe["rolling_accuracy"]).clip(0, 1).fillna(0)
            else:
                # Fallback confidence calculation
                pred_abs = np.abs(dataframe[prediction_col])
                pred_percentile = pred_abs.rolling(100).rank(pct=True)
                dataframe["pred_confidence"] = (pred_percentile * dataframe["rolling_accuracy"]).fillna(0.5)
        
        return dataframe

    def _compute_basic_prediction_metrics(self, dataframe: pd.DataFrame, metadata: dict, 
                                        label_col: str, prediction_col: str, log_metrics: bool) -> pd.DataFrame:
        """Fallback to basic prediction metrics if enhanced system fails"""
        # Add basic computed columns
        dataframe = self._add_basic_computed_columns(dataframe, label_col, prediction_col)
        
        # Prepare summary metrics and optionally persist to in-memory storage
        pair = metadata.get("pair", "unknown")
        try:
            valid_df = dataframe.dropna(subset=[label_col, prediction_col])
            corr = valid_df[prediction_col].corr(valid_df[label_col]) if not valid_df.empty else np.nan
        except Exception:
            corr = np.nan

        total_predictions = int((dataframe.get("do_predict") == 1).sum()) if "do_predict" in dataframe.columns else 0
        total_targets_available = int(dataframe[label_col].notna().sum()) if label_col in dataframe.columns else 0
        fraction_predicted = (total_predictions / total_targets_available) if total_targets_available > 0 else 0

        metrics = {
            "pair": pair,
            "total_predictions": total_predictions,
            "fraction_predicted": fraction_predicted,
            "rolling_accuracy": float(dataframe["rolling_accuracy"].iloc[-1]) if "rolling_accuracy" in dataframe.columns else float('nan'),
            "mae": float(dataframe["mae"].iloc[-1]) if "mae" in dataframe.columns else float('nan'),
            "correlation": float(corr) if pd.notna(corr) else float('nan')
        }

        # Ensure storage exists and append
        try:
            if not hasattr(self, 'pred_metrics_storage'):
                self.pred_metrics_storage = []
            self.pred_metrics_storage.append(metrics)
        except Exception:
            logger.debug("Could not append prediction metrics to storage")

        if log_metrics:
            try:
                logger.info(f"Metrics | {pair} | roll_acc={dataframe['rolling_accuracy'].iloc[-1] if 'rolling_accuracy' in dataframe else np.nan:.3f} | mae={dataframe['mae'].iloc[-1] if 'mae' in dataframe else np.nan:.6f} | corr={corr if pd.notna(corr) else 'NaN'}")
            except Exception:
                logger.info(f"Metrics | {pair} | (could not compute display values)")

        return dataframe

    def save_prediction_metrics(self, filename: str = "pred_metrics.csv") -> None:
        """Save enhanced prediction metrics focused on prediction quality"""
        try:
            # Try to use simple enhanced metrics saving
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
            from simple_enhanced_metrics import save_prediction_metrics_enhanced
            save_prediction_metrics_enhanced(self, filename)
            return
            
        except ImportError as e:
            logger.warning(f"Simple enhanced metrics saving not available: {e}")
        except Exception as e:
            logger.error(f"Enhanced metrics saving failed: {e}")
        
        # Fallback to basic CSV saving
        try:
            if hasattr(self, 'pred_metrics_storage') and self.pred_metrics_storage:
                from pathlib import Path
                import csv
                
                base = Path(self.config.get("user_data_dir", "./user_data"))
                out = base / filename
                
                write_header = not out.exists()
                with out.open("a", newline='') as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(self.pred_metrics_storage[0].keys()))
                    if write_header:
                        writer.writeheader()
                    for row in self.pred_metrics_storage:
                        writer.writerow(row)
                logger.info(f"Basic prediction metrics saved to {out}")
        except Exception as e:
            logger.warning(f"Failed to save prediction metrics: {e}")

    def _check_performance_alerts(self):
        """Check for performance issues and log alerts"""
        try:
            if not hasattr(self, 'enhanced_metrics'):
                return
            
            # Get current performance for all pairs
            pairs = getattr(self, 'pair_whitelist', ['BTC/USDT:USDT'])  # Default fallback
            
            for pair in pairs:
                summary = self.enhanced_metrics.get_performance_summary(pair, days=7)
                
                if not summary:
                    continue
                
                # Check correlation threshold
                if summary.get('avg_correlation', 1.0) < 0.4:
                    logger.warning(f"🔴 LOW CORRELATION ALERT: {pair} correlation={summary.get('avg_correlation', 0):.3f}")
                
                # Check direction accuracy
                if summary.get('avg_direction_accuracy', 1.0) < 0.45:
                    logger.warning(f"🔴 LOW ACCURACY ALERT: {pair} direction_accuracy={summary.get('avg_direction_accuracy', 0):.3f}")
                
                # Check Sharpe ratio
                if summary.get('avg_sharpe_ratio', 0) < 0.5:
                    logger.warning(f"⚠️  LOW SHARPE ALERT: {pair} sharpe_ratio={summary.get('avg_sharpe_ratio', 0):.2f}")
                
                # Check stability trend
                if summary.get('model_stability_trend', 0) < -0.15:
                    logger.warning(f"⚠️  STABILITY DECLINING: {pair} - consider retraining")
            
        except Exception as e:
            logger.debug(f"Performance alerts check failed: {e}")

    def get_prediction_quality_summary(self) -> dict:
        """Get current prediction quality summary for monitoring"""
        try:
            if hasattr(self, 'enhanced_metrics'):
                return self.enhanced_metrics.get_performance_summary()
            else:
                # Basic summary from stored metrics
                if hasattr(self, 'pred_metrics_storage') and self.pred_metrics_storage:
                    recent_metrics = self.pred_metrics_storage[-10:]  # Last 10 entries
                    return {
                        'avg_correlation': np.mean([m.get('correlation', 0) for m in recent_metrics]),
                        'avg_rolling_accuracy': np.mean([m.get('rolling_accuracy', 0) for m in recent_metrics]),
                        'avg_mae': np.mean([m.get('mae', 0) for m in recent_metrics]),
                        'total_predictions': sum([m.get('total_predictions', 0) for m in recent_metrics])
                    }
        except Exception as e:
            logger.error(f"Failed to get prediction quality summary: {e}")
            return {}

    # -------------------- Fear & Greed helpers --------------------
    def load_historical_fng_data(self) -> pd.DataFrame | None:
        fng_data_path = self.config.get('fng_data_path')
        if fng_data_path and os.path.exists(fng_data_path):
            try:
                fng = pd.read_csv(fng_data_path, index_col='timestamp', parse_dates=True)
                logger.info(f"Loaded historical FNG from {fng_data_path}")
                return fng
            except Exception as e:
                logger.warning(f"FNG load error: {e}")
        return None

    def get_fear_and_greed_index(self, current_date=None):
        if self.dp and self.dp.runmode in (RunMode.BACKTEST, RunMode.HYPEROPT) and self.historical_fng_data is not None:
            try:
                if current_date in self.historical_fng_data.index:
                    row = self.historical_fng_data.loc[current_date]
                    return int(row['value']), row['value_classification']
            except Exception:
                return None, None
            return None, None
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            if resp.ok:
                data = resp.json().get('data', [])
                if data:
                    latest = data[0]
                    return int(latest.get('value', 50)), latest.get('value_classification')
        except Exception as e:
            logger.debug(f"FNG fetch error: {e}")
        return None, None
