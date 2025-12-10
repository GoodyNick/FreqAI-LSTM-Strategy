import logging
import operator
from functools import reduce
from typing import Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from sympy import use
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.fftpack import fft
from scipy.stats import zscore
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.vendor.qtpylib.indicators import crossed_above, crossed_below
from freqtrade.enums import RunMode
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class HybridLSTMStrategy_v2(IStrategy):
    """
    HYBRID ML + Technical Analysis Strategy
    Combines LSTM predictions with classic technical indicators for entry/exit signals.
    """
    
    plot_config = {
        "main_plot": {
            "tema": {"color": "blue"},
            "bb_upperband": {"color": "red"},
            "bb_middleband": {"color": "gray"},
            "bb_lowerband": {"color": "red"}
        },
        "subplots": {
            "ML Predictions": {
                "Prediction": {"color": "purple", "plot_type": "line"},
                "True Label": {"color": "blue", "plot_type": "line"},
                "pred_ema_fast": {"color": "red", "plot_type": "line"},
                "pred_ema_slow": {"color": "green", "plot_type": "line"},
            },
            "Confidence & Filters": {
                "pred_confidence": {"color": "orange", "plot_type": "line"},
                "vol_rank": {"color": "blue", "plot_type": "line"},
                "vol_rank_dynamic_threshold": {"color": "lightblue", "plot_type": "line"},
                "do_predict": {"color": "purple", "plot_type": "scatter"},
            },
            "Technical Indicators": {
                "rsi": {"color": "green", "plot_type": "line"},
            },
        },
    }
    
    minimal_roi = {"0": 100}
    stoploss = -0.99
    timeframe = '1h'
    can_short = True
    startup_candle_count = 300
    process_only_new_candles = True
    use_custom_stoploss = True

    # === ML PREDICTION PARAMETERS ===
    # Method selection for ML signals
    # Whether to smooth ML predictions with EMA (allow optimizer to choose)
    use_ema_predictions = CategoricalParameter([False, True], default=True, space="buy", optimize=True)
    # Threshold for treating a prediction as a trade signal
    prediction_threshold = DecimalParameter(0.01, 0.50, default=0.05, decimals=2, space="buy", optimize=True)
    # Minimum confidence required for using ML signal (used with optional filters)
    confidence_threshold = DecimalParameter(0.01, 0.60, default=0.10, decimals=2, space="buy", optimize=True)
    
    # EMA smoothing for predictions (only used if use_ema_predictions=True)
    pred_ema_fast_period = IntParameter(2, 12, default=5, space="buy", optimize=True)
    pred_ema_slow_period = IntParameter(8, 48, default=12, space="buy", optimize=True)
    
    # === CLASSIC TECHNICAL INDICATOR PARAMETERS ===
    # RSI thresholds for entry/exit
    long_rsi = IntParameter(5, 45, default=30, space="buy", optimize=True)
    exit_long_rsi = IntParameter(45, 95, default=70, space="sell", optimize=True)
    short_rsi = IntParameter(55, 95, default=70, space="buy", optimize=True)
    exit_short_rsi = IntParameter(5, 50, default=30, space="sell", optimize=True)
    
    # Bollinger Bands
    bb_period = IntParameter(8, 40, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy", optimize=True)
    
    # TEMA trend indicator
    tema_period = IntParameter(5, 60, default=21, space="buy", optimize=True)
    
    # === SIGNAL COMBINATION & FILTERING ===
    # Volume filter
    vol_rank_threshold = DecimalParameter(0.1, 0.9, default=0.5, decimals=2, space="buy", optimize=True)
    
    # Signal filters (enable/disable)
    use_volume_confirmation = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    use_confidence_filter = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    
    # === RISK MANAGEMENT PARAMETERS ===
    # Dynamic stoploss
    atr_multiplier = DecimalParameter(1.0, 4.0, default=2.5, decimals=1, space="sell", optimize=True)
    max_stoploss_pct = DecimalParameter(0.01, 0.15, default=0.06, decimals=2, space="sell", optimize=True)
    
    # Position sizing & leverage
    use_leverage = False
    base_leverage = IntParameter(1, 3, default=1, space="buy", optimize=use_leverage)
    max_leverage_cap = IntParameter(2, 10, default=3, space="buy", optimize=use_leverage)

    def informative_pairs(self):
        return []

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, metadata: Dict, **kwargs):
        """
        Core period-based features - AUTO-EXPANDED by FreqAI across indicator_periods_candles [10, 20].
        Each feature here will be duplicated 2x (once for period=10, once for period=20).
        So 6 features × 2 periods = 12 base features before datasieve filtering.
        """
        # Compact, robust multi-horizon indicators (auto-expanded across indicator_periods_candles)
        # 1) RSI: momentum
        dataframe[f'%-rsi-{period}'] = ta.RSI(dataframe, timeperiod=period)

        # 2) EMA convergence: short vs long relative difference (trend)
        short_p = max(2, int(period // 2))
        long_p = max(short_p + 1, int(period))
        ema_short = ta.EMA(dataframe, timeperiod=short_p)
        ema_long = ta.EMA(dataframe, timeperiod=long_p)
        dataframe[f'%-ema_diff-{period}'] = (ema_short - ema_long) / (ema_long + 1e-9)

        # 3) Rate of change (momentum)
        dataframe[f'%-roc-{period}'] = ta.ROC(dataframe, timeperiod=period)

        # 4) Bollinger width percent (volatility around price)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2)
        dataframe[f'%-bb_width_pct-{period}'] = ((bb['upper'] - bb['lower']) / (bb['mid'] + 1e-9)) * 100

        # 5) MFI for volume+price momentum
        try:
            dataframe[f'%-mfi-{period}'] = ta.MFI(dataframe, timeperiod=period)
        except Exception:
            dataframe[f'%-mfi-{period}'] = 0

        # 6) OBV (volume flow) smoothed
        try:
            dataframe[f'%-obv_sma_{period}'] = pd.Series(ta.OBV(dataframe)).rolling(window=max(3, period//2)).mean()
        except Exception:
            dataframe[f'%-obv_sma_{period}'] = 0

        # Fill NaNs conservatively
        dataframe.fillna(0, inplace=False)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        """
        Statistical features - NOT auto-expanded across periods.
        These generate consistently regardless of training/prediction phase.
        """
        # Proven short-term statistical features useful for LSTMs
        # 1) Lagged returns (1, 3, 6)
        dataframe['%-ret_1'] = dataframe['close'].pct_change(1).fillna(0)
        dataframe['%-ret_3'] = dataframe['close'].pct_change(3).fillna(0)
        dataframe['%-ret_6'] = dataframe['close'].pct_change(6).fillna(0)

        # 2) Log return (smoothed)
        dataframe['%-logret_1'] = np.log(dataframe['close'] / dataframe['close'].shift(1)).fillna(0)

        # 3) Volume normalized by short SMA
        dataframe['%-volume_sma_21'] = dataframe['volume'].rolling(window=21, min_periods=1).mean().replace(0, 1e-9)
        dataframe['%-volume_norm'] = dataframe['volume'] / dataframe['%-volume_sma_21']

        # 4) OBV (raw) as an important volume-flow proxy
        try:
            dataframe['%-obv'] = ta.OBV(dataframe)
        except Exception:
            dataframe['%-obv'] = 0

        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, metadata: Dict, **kwargs):
        """
        Standard features - NOT auto-expanded.
        Day of week, hour of day, and regime detection.
        """
        # Time-based features (cyclical encoding not necessary here; simple scaled features work)
        dataframe['%-day_of_week'] = (dataframe['date'].dt.dayofweek).fillna(0)
        dataframe['%-hour_of_day'] = (dataframe['date'].dt.hour).fillna(0)

        # ATR percent for volatility normalization (used by target and stoploss)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['%-atr_pct'] = dataframe['atr'] / (dataframe['close'] + 1e-9)

        # Rolling volatility and trend
        dataframe['%-rolling_vol_20'] = dataframe['close'].pct_change().rolling(window=20, min_periods=1).std().fillna(0)
        ema_short = ta.EMA(dataframe, timeperiod=8)
        ema_long = ta.EMA(dataframe, timeperiod=21)
        dataframe['%-ema_trend'] = (ema_short - ema_long) / (ema_long + 1e-9)

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        Required FreqAI method for target generation.
        """
        dataframe['&-s_target'] = self.create_target_T(dataframe)
        return dataframe

    def create_target_T(self, dataframe: DataFrame) -> pd.Series:
        """
        ENHANCED MULTI-SIGNAL TARGET CREATION
        
        Creates a sophisticated target that combines:
        1. Multi-horizon future returns (short/medium term)
        2. Volatility-adjusted regime weighting
        3. Momentum persistence signals
        4. Risk-adjusted return expectations
        
        This approach aims to predict not just direction, but the quality and persistence
        of price movements, leading to better LSTM learning and higher accuracy.
        """
        # Simpler, robust target: ATR-normalized forward log-return, Z-scored and clipped.
        try:
            base_horizon = int(self.freqai_info.get("feature_parameters", {}).get("label_period_candles", 3))
        except Exception:
            base_horizon = 3

        K = max(1, base_horizon)
        df = dataframe.copy()

        # Ensure ATR for normalization
        if 'atr' not in df.columns:
            df['atr'] = ta.ATR(df, timeperiod=14)

        # Forward log return
        raw_forward = np.log((df['close'].shift(-K) + 1e-9) / (df['close'] + 1e-9))

        # Light smoothing
        smooth = raw_forward.ewm(span=3, adjust=False).mean()

        # Normalize by ATR percent to make returns comparable across volatility regimes
        atr_pct = df['atr'] / (df['close'] + 1e-9)
        normalized = smooth / (atr_pct + 1e-8)

        # Rolling z-score (stable) and clipping
        z_mean = normalized.rolling(window=50, min_periods=5).mean()
        z_std = normalized.rolling(window=50, min_periods=5).std()
        zscore_t = (normalized - z_mean) / (z_std + 1e-6)

        clipped = zscore_t.clip(-2.0, 2.0)

        # Optional scaling to match model output amplitude
        scaled = clipped * 0.7

        return scaled.fillna(0)

    # Note: indicator calculation moved into `populate_indicators` to ensure
    # all indicators are available after FreqAI target/prediction generation.

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates indicators and integrates with FreqAI for ML predictions.
        """
        # Initialize FreqAI info
        self.freqai_info = self.config.get("freqai", {})
        
        # Ensure clean data
        dataframe['close'] = pd.to_numeric(dataframe['close'], errors='coerce').replace(0, np.nan).ffill().bfill()
        dataframe['volume'] = pd.to_numeric(dataframe['volume'], errors='coerce').replace(0, 1e-9).ffill().bfill().fillna(1e-9)

        # Skip FreqAI processing in certain conditions
        fit_live_candles = self.freqai_info.get("fit_live_predictions_candles", 0)
        if fit_live_candles <= 0 and getattr(self.dp, "runmode", None) in [RunMode.BACKTEST, RunMode.HYPEROPT]:
            pass  # Skip live prediction fitting in backtest
        
        # Start FreqAI processing
        if hasattr(self, 'freqai') and self.freqai:
            dataframe = self.freqai.start(dataframe, metadata, self)
        
        # Ensure FreqAI columns exist
        freqai_cols = ["&-s_target", "&-s_target_mean", "&-s_target_std", "do_predict"]
        for col in freqai_cols:
            if col not in dataframe.columns:
                if col == "do_predict":
                    dataframe[col] = 1
                else:
                    dataframe[col] = 0
        
        # Add target and prediction columns for analysis
        dataframe["T"] = self.create_target_T(dataframe)
        dataframe["Prediction"] = dataframe["&-s_target"]
        dataframe["True Label"] = dataframe["T"]
        # --- Calculate all indicators used by trade logic (must be after FreqAI target/prediction) ---
        # RSI and TEMA
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.tema_period.value)

        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=self.bb_period.value, nbdevup=self.bb_std.value, nbdevdn=self.bb_std.value)
        dataframe['bb_upperband'] = bb['upperband']
        dataframe['bb_middleband'] = bb['middleband']
        dataframe['bb_lowerband'] = bb['lowerband']

        # EMA for regime detection (used by entry/exit)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)

        # Bollinger breakout / mean-reversion helpers already provided by bands and close

        # ML prediction processing with dynamic parameters
        if "&-s_target" in dataframe.columns:
            if hasattr(self, 'use_ema_predictions') and self.use_ema_predictions.value:
                dataframe['pred_ema_fast'] = ta.EMA(dataframe["&-s_target"], timeperiod=self.pred_ema_fast_period.value)
                dataframe['pred_ema_slow'] = ta.EMA(dataframe["&-s_target"], timeperiod=self.pred_ema_slow_period.value)
            else:
                dataframe['pred_ema_fast'] = dataframe["&-s_target"]
                dataframe['pred_ema_slow'] = 0

            dataframe["pred_confidence"] = np.abs(dataframe["&-s_target"])
            dataframe["vol_rank"] = dataframe["volume"].rolling(50).rank(pct=True)
            dataframe["vol_rank_dynamic_threshold"] = self.vol_rank_threshold.value
            dataframe["rolling_trend_scaled"] = dataframe["&-s_target"].rolling(20).mean()
            dataframe["rolling_trend_threshold"] = 0.0
        else:
            dataframe['pred_ema_fast'] = 0
            dataframe['pred_ema_slow'] = 0
            dataframe["pred_confidence"] = 0
            dataframe["vol_rank"] = 0.5
            dataframe["vol_rank_dynamic_threshold"] = self.vol_rank_threshold.value
            dataframe["rolling_trend_scaled"] = 0
            dataframe["rolling_trend_threshold"] = 0

        # Add do_predict column (FreqAI will update this)
        if "do_predict" not in dataframe.columns:
            dataframe["do_predict"] = 1
        
        # Calculate ATR for custom_stoploss and volatility features
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['%-atr_pct'] = dataframe['atr'] / (dataframe['close'] + 1e-9)

        # Extra rolling stats used elsewhere
        dataframe['%-rolling_vol_20'] = dataframe['close'].pct_change().rolling(window=20, min_periods=1).std().fillna(0)
        dataframe['%-ema_trend'] = (dataframe['ema_8'] - dataframe['ema_21']) / (dataframe['ema_21'] + 1e-9)
        
        # Compute and log enhanced prediction metrics (only if we have valid predictions)
        if "&-s_target" in dataframe.columns and "T" in dataframe.columns:
            valid_predictions = dataframe["&-s_target"].notna().sum()
            if valid_predictions > 10:  # Only compute metrics if we have enough data
                self.compute_enhanced_prediction_metrics(dataframe, metadata)
        
        return dataframe
    
    def compute_prediction_metrics(self, dataframe: pd.DataFrame, metadata: dict, 
                                 label_col: str = "T", prediction_col: str = "&-s_target") -> pd.DataFrame:
        """Backward compatibility wrapper for enhanced prediction metrics"""
        return self.compute_enhanced_prediction_metrics(dataframe, metadata, label_col, prediction_col)

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """Define entry conditions combining ML and technical signals"""
        
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = None

        # --- Regime detection (trend vs mean-reversion) using precomputed EMAs ---
        trend_strength = (df['ema_8'] - df['ema_21']) / (df['ema_21'] + 1e-9)
        trending_up = trend_strength > 0

        # --- ML Signals ---
        if self.use_ema_predictions.value:
            ml_long_signal = df['pred_ema_fast'] > df['pred_ema_slow']
            ml_short_signal = df['pred_ema_fast'] < df['pred_ema_slow']
        else:
            ml_long_signal = df.get("&-s_target", 0) > self.prediction_threshold.value
            ml_short_signal = df.get("&-s_target", 0) < -self.prediction_threshold.value

        # Prediction confidence (used to relax/ tighten classic requirements)
        pred_conf = df.get('pred_confidence', pd.Series(0, index=df.index))

        # --- Classic Technical Signals (regime aware) ---
        # Breakout (trend-following) and mean-reversion rules
        breakout_long = df['close'] > df['bb_upperband']
        breakout_short = df['close'] < df['bb_lowerband']

        meanrev_long = df['close'] < df['bb_lowerband']
        meanrev_short = df['close'] > df['bb_upperband']

        # Momentum confirmation for trend rules
        tema_rising = df['tema'] > df['tema'].shift(1)
        tema_falling = df['tema'] < df['tema'].shift(1)

        # Construct classic signals depending on detected regime
        classic_long_signal = (
            (trending_up & breakout_long & tema_rising) |
            (~trending_up & meanrev_long & (df['rsi'] < self.long_rsi.value))
        )

        classic_short_signal = (
            (~trending_up & meanrev_short & (df['rsi'] > self.short_rsi.value)) |
            (trending_up & breakout_short & tema_falling)
        )

        # --- Combination logic with confidence/volume filters ---
        def base_entry_condition_series(side: str = None):
            cond = (df.get("do_predict", 1) == 1)

            # Volume confirmation (optional)
            if self.use_volume_confirmation.value:
                cond &= (df.get("vol_rank", 0) > df.get("vol_rank_dynamic_threshold", 0.5))

            # Confidence filter (optional) - allow relaxed rules for very confident preds
            if self.use_confidence_filter.value:
                # If confidence is very high, relax the confidence requirement
                high_conf_relax = pred_conf > (self.confidence_threshold.value * 1.5)
                cond &= (pred_conf > self.confidence_threshold.value) | high_conf_relax

            return cond

        # Combine ML + Classic according to user preference
        hybrid_long = ml_long_signal & classic_long_signal
        hybrid_short = ml_short_signal & classic_short_signal

        # Tag entries with method for easier post-trade analysis
        df.loc[base_entry_condition_series("long") & hybrid_long, "enter_long"] = 1
        df.loc[base_entry_condition_series("long") & hybrid_long, "enter_tag"] = "hybrid_long"

        df.loc[base_entry_condition_series("short") & hybrid_short, "enter_short"] = 1
        df.loc[base_entry_condition_series("short") & hybrid_short, "enter_tag"] = "hybrid_short"
        
        # Allow ML to override classic if confidence is high
        high_conf = pred_conf > (self.confidence_threshold.value * 1.5)
        ml_long = (ml_long_signal & (classic_long_signal | high_conf)) | (classic_long_signal & ~self.use_confidence_filter.value)
        ml_short = (ml_short_signal & (classic_short_signal | high_conf)) | (classic_short_signal & ~self.use_confidence_filter.value)

        df.loc[ml_long & base_entry_condition_series("long"), "enter_long"] = 1
        df.loc[ml_long & base_entry_condition_series("long"), "enter_tag"] = "ml_override_long"

        df.loc[ml_short & base_entry_condition_series("short"), "enter_short"] = 1
        df.loc[ml_short & base_entry_condition_series("short"), "enter_tag"] = "ml_override_short"

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions with dynamic prediction methods"""
        
        df["exit_long"] = 0
        df["exit_short"] = 0
        df["exit_tag"] = None
        
        # --- ML Exit Signals ---
        if self.use_ema_predictions.value:
            ml_exit_long = crossed_below(df['pred_ema_fast'], df['pred_ema_slow'])
            ml_exit_short = crossed_above(df['pred_ema_fast'], df['pred_ema_slow'])
        else:
            ml_exit_long = df.get("&-s_target", 0) < -self.prediction_threshold.value
            ml_exit_short = df.get("&-s_target", 0) > self.prediction_threshold.value

        # --- Technical Exit Rules ---
        # RSI extremes, opposite band breakout, or momentum flip
        rsi_cross_up = crossed_above(df["rsi"], self.exit_long_rsi.value)
        rsi_cross_down = crossed_below(df["rsi"], self.exit_short_rsi.value)

        opposite_band_long = df['close'] > df['bb_middleband']
        opposite_band_short = df['close'] < df['bb_middleband']

        classic_exit_long = rsi_cross_up | opposite_band_long
        classic_exit_short = rsi_cross_down | opposite_band_short

        # Confidence decay: if prediction confidence falls sharply, consider exiting
        pred_conf = df.get('pred_confidence', pd.Series(0, index=df.index))
        low_conf_exit = pred_conf < (self.confidence_threshold.value * 0.5)

        # Combine exits conservatively: any strong ML exit OR technical exit OR low confidence
        long_exit_condition = ml_exit_long | classic_exit_long | low_conf_exit
        short_exit_condition = ml_exit_short | classic_exit_short | low_conf_exit

        df.loc[long_exit_condition, "exit_long"] = 1
        df.loc[long_exit_condition, "exit_tag"] = "hybrid_exit_long"

        df.loc[short_exit_condition, "exit_short"] = 1
        df.loc[short_exit_condition, "exit_tag"] = "hybrid_exit_short"

        return df
    
    def compute_enhanced_prediction_metrics(self, dataframe: pd.DataFrame, metadata: dict, 
                                          label_col: str = "T", prediction_col: str = "&-s_target") -> pd.DataFrame:
        """
        ENHANCED PREDICTION METRICS: Comprehensive analysis including regime-specific performance.
        Provides detailed insights into model quality across different market conditions.
        """
        if label_col not in dataframe.columns or prediction_col not in dataframe.columns:
            logger.warning(f"Missing columns for enhanced metrics: {label_col} or {prediction_col}")
            return dataframe
        
        # Prepare data with market regime indicators
        df_analysis = dataframe.copy()
        
        # Add market regime indicators if not present
        if 'atr' not in df_analysis.columns:
            df_analysis['atr'] = ta.ATR(df_analysis, timeperiod=14)
        
        # Volatility regime
        vol_short = df_analysis['close'].rolling(8).std()
        vol_long = df_analysis['close'].rolling(21).std()
        df_analysis['vol_regime'] = vol_short / (vol_long + 1e-8)
        
        # Trend regime
        ema_8 = ta.EMA(df_analysis, timeperiod=8)
        ema_21 = ta.EMA(df_analysis, timeperiod=21)
        df_analysis['trend_strength'] = np.abs((ema_8 - ema_21) / (ema_21 + 1e-8))
        
        # Clean data for analysis
        required_cols = [label_col, prediction_col, 'vol_regime', 'trend_strength']
        df_clean = df_analysis[required_cols].dropna()
        
        if len(df_clean) < 50:
            logger.warning(f"Insufficient data for enhanced metrics: {len(df_clean)} samples")
            return dataframe
        
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            labels = df_clean[label_col].values
            predictions = df_clean[prediction_col].values
            vol_regime = df_clean['vol_regime'].values
            trend_strength = df_clean['trend_strength'].values
            
            # === OVERALL METRICS ===
            mse = mean_squared_error(labels, predictions)
            mae = mean_absolute_error(labels, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(labels, predictions)
            correlation = np.corrcoef(labels, predictions)[0, 1] if len(labels) > 1 else 0
            
            # Direction accuracy
            label_direction = np.sign(labels)
            pred_direction = np.sign(predictions)
            direction_accuracy = np.mean(label_direction == pred_direction)
            
            # === REGIME-SPECIFIC ANALYSIS ===
            
            # Volatility regimes
            high_vol_mask = vol_regime > np.percentile(vol_regime, 66)
            low_vol_mask = vol_regime < np.percentile(vol_regime, 33)
            
            high_vol_acc = np.mean(label_direction[high_vol_mask] == pred_direction[high_vol_mask]) if np.any(high_vol_mask) else 0
            low_vol_acc = np.mean(label_direction[low_vol_mask] == pred_direction[low_vol_mask]) if np.any(low_vol_mask) else 0
            
            # Trend regimes
            strong_trend_mask = trend_strength > np.percentile(trend_strength, 66)
            weak_trend_mask = trend_strength < np.percentile(trend_strength, 33)
            
            strong_trend_acc = np.mean(label_direction[strong_trend_mask] == pred_direction[strong_trend_mask]) if np.any(strong_trend_mask) else 0
            weak_trend_acc = np.mean(label_direction[weak_trend_mask] == pred_direction[weak_trend_mask]) if np.any(weak_trend_mask) else 0
            
            # === CONFIDENCE ANALYSIS ===
            pred_confidence = np.abs(predictions)
            
            # Confidence-based accuracy
            high_conf_mask = pred_confidence > np.percentile(pred_confidence, 75)
            low_conf_mask = pred_confidence < np.percentile(pred_confidence, 25)
            
            high_conf_acc = np.mean(label_direction[high_conf_mask] == pred_direction[high_conf_mask]) if np.any(high_conf_mask) else 0
            low_conf_acc = np.mean(label_direction[low_conf_mask] == pred_direction[low_conf_mask]) if np.any(low_conf_mask) else 0
            
            # === SIGNAL QUALITY METRICS ===
            
            # Information Coefficient (IC) - correlation between predictions and future returns
            ic = correlation
            
            # Information Ratio - IC adjusted for consistency
            rolling_ic = pd.Series(predictions).rolling(50).corr(pd.Series(labels))
            ir = ic / (rolling_ic.std() + 1e-8) if not rolling_ic.isna().all() else 0
            
            # Hit rate by magnitude
            strong_signals = np.abs(predictions) > np.percentile(pred_confidence, 75)
            strong_signal_acc = np.mean(label_direction[strong_signals] == pred_direction[strong_signals]) if np.any(strong_signals) else 0
            
            # === TRADING SIMULATION METRICS ===
            
            # Simulated P&L based on predictions
            simulated_returns = labels * np.sign(predictions)
            
            # Sharpe-like ratio for predictions
            prediction_sharpe = np.mean(simulated_returns) / (np.std(simulated_returns) + 1e-8) if len(simulated_returns) > 1 else 0
            
            # Win rate and average win/loss
            winning_trades = simulated_returns > 0
            win_rate = np.mean(winning_trades) if len(simulated_returns) > 0 else 0
            avg_win = np.mean(simulated_returns[winning_trades]) if np.any(winning_trades) else 0
            avg_loss = np.mean(simulated_returns[~winning_trades]) if np.any(~winning_trades) else 0
            
            # === COMPREHENSIVE LOGGING ===
            pair_name = metadata.get('pair', 'Unknown')
            
            logger.info(f"🔬 ============= ENHANCED PREDICTION ANALYSIS: {pair_name} ==============")
            logger.info(f"📊 Dataset Overview:")
            logger.info(f"   • Total samples: {len(df_clean):,}")
            logger.info(f"   • Label range: [{labels.min():.3f}, {labels.max():.3f}]")
            logger.info(f"   • Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
            
            logger.info(f"📈 Core Performance:")
            logger.info(f"   • Direction Accuracy: {direction_accuracy:.1%}")
            logger.info(f"   • Correlation (IC): {correlation:.4f}")
            logger.info(f"   • Information Ratio: {ir:.4f}")
            logger.info(f"   • R²: {r2:.4f}")
            logger.info(f"   • RMSE: {rmse:.4f}")
            
            logger.info(f"🌡️ Regime-Specific Accuracy:")
            logger.info(f"   • High Volatility: {high_vol_acc:.1%}")
            logger.info(f"   • Low Volatility: {low_vol_acc:.1%}")
            logger.info(f"   • Strong Trend: {strong_trend_acc:.1%}")
            logger.info(f"   • Weak Trend: {weak_trend_acc:.1%}")
            
            logger.info(f"💪 Confidence Analysis:")
            logger.info(f"   • High Confidence Accuracy: {high_conf_acc:.1%}")
            logger.info(f"   • Low Confidence Accuracy: {low_conf_acc:.1%}")
            logger.info(f"   • Strong Signal Accuracy: {strong_signal_acc:.1%}")
            logger.info(f"   • Average Confidence: {np.mean(pred_confidence):.4f}")
            
            logger.info(f"💰 Trading Simulation:")
            logger.info(f"   • Prediction Sharpe: {prediction_sharpe:.4f}")
            logger.info(f"   • Simulated Win Rate: {win_rate:.1%}")
            logger.info(f"   • Average Win: {avg_win:.4f}")
            logger.info(f"   • Average Loss: {avg_loss:.4f}")
            logger.info(f"   • Win/Loss Ratio: {(avg_win/abs(avg_loss)):.2f}" if avg_loss < 0 else "N/A")
            
            logger.info(f"🔬 =================================================================")
            
        except Exception as e:
            logger.error(f"Error computing enhanced prediction metrics: {e}")
            
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                       current_profit: float, after_fill: bool, **kwargs) -> float | None:
        """
        Custom trailing stoploss based on ATR and prediction confidence.
        Returns the new stoploss value as a percentage relative to current_rate.
        
        :param pair: Pair currently being analyzed
        :param trade: Trade object with entry_side, open_rate, etc.
        :param current_time: Current datetime
        :param current_rate: Current rate (entry or exit pricing)
        :param current_profit: Current profit as ratio
        :param after_fill: True if called after an order fill
        :param **kwargs: Additional parameters
        :return: New stoploss value relative to current_rate, or None to keep current
        """
        # Fail-safe: ensure trade object is valid
        if not trade or not hasattr(trade, 'open_rate'):
            return None
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) == 0:
            return None  # Keep current stoploss
        
        # Get latest values
        latest = dataframe.iloc[-1]
        
        # Base stoploss parameters (hyperoptimizable)
        atr_multiplier = self.atr_multiplier.value
        max_stoploss_pct = self.max_stoploss_pct.value  # This should be positive (e.g., 0.06 for 6%)
        
        # Get ATR value - critical for stoploss calculation
        atr_value = latest.get('atr', 0)
        if atr_value <= 0 or pd.isna(atr_value):
            return None  # Keep current stoploss if no valid ATR
        
        # Calculate ATR-based stoploss distance from current_rate
        # This is the percentage distance we want the stoploss to be from current_rate
        try:
            atr_stoploss_distance = (atr_value * atr_multiplier) / current_rate
        except (ValueError, ZeroDivisionError):
            return None  # Safety check for division errors
        
        # Adjust based on prediction confidence
        pred_confidence = latest.get('pred_confidence', 0.5)
        if pred_confidence > 0:
            # Higher confidence = tighter stoploss (lower distance)
            # Confidence range [0, 1] -> factor range [1.5, 0.5]
            confidence_factor = 1.5 - pred_confidence
            atr_stoploss_distance *= confidence_factor
        
        # Apply maximum stoploss limit
        stoploss_distance = min(atr_stoploss_distance, max_stoploss_pct)
        
        # For short trades, we need to WIDEN the stoploss (positive distance above current rate)
        # For long trades, we need to TIGHTEN the stoploss (negative distance below current rate)
        if trade.is_short:
            # Short trade: stoploss is ABOVE current_rate (positive value)
            return stoploss_distance
        else:
            # Long trade: stoploss is BELOW current_rate (negative value)
            return -stoploss_distance

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, 
                           proposed_stake: float, min_stake: float | None, max_stake: float, 
                           leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        """
        Custom stake amount based on prediction confidence and volatility.
        Simplified version focusing on risk-adjusted position sizing.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) == 0:
            return proposed_stake
            
        # Get latest values
        latest = dataframe.iloc[-1]
        
        # Base scaling factor
        base_factor = 1.0
        
        # Adjust based on prediction confidence
        pred_confidence = latest.get('pred_confidence', 0)
        if pred_confidence > 0:
            # Higher confidence = larger position (but capped)
            confidence_multiplier = 0.5 + min(pred_confidence * 1.5, 1.5)  # Range: 0.5 to 2.0
            base_factor *= confidence_multiplier
        
        # Adjust based on volatility (ATR)
        atr_pct = latest.get('%-atr_pct', 0)
        if atr_pct > 0:
            # Higher volatility = smaller position
            volatility_factor = max(0.3, min(1.5, 1.0 / (1.0 + atr_pct * 0.1)))
            base_factor *= volatility_factor
        
        # Calculate final stake
        final_stake = proposed_stake * base_factor
        
        # Apply limits
        if min_stake is not None:
            final_stake = max(final_stake, min_stake)
        final_stake = min(final_stake, max_stake)
        
        return final_stake

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                **kwargs) -> float:
        """
        Custom leverage based on prediction confidence and volatility.
        Conservative approach prioritizing capital preservation.
        Always returns a positive integer between 1 and max_leverage.
        """
        if not self.use_leverage:
            return 1.0
            
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if dataframe is None or len(dataframe) == 0:
            return 1.0
            
        # Get latest values
        latest = dataframe.iloc[-1]
        
        # Base leverage (now hyperoptimizable)
        base_leverage = float(self.base_leverage.value)
        
        # Adjust based on prediction confidence
        pred_confidence = latest.get('pred_confidence', 0)
        if pred_confidence > 0:
            # Higher confidence allows higher leverage
            confidence_multiplier = max(0.5, min(2.0, pred_confidence * 2.0))
            base_leverage *= confidence_multiplier
        
        # Reduce leverage based on volatility
        atr_pct = latest.get('%-atr_pct', 0)
        if atr_pct > 2.0:  # High volatility
            volatility_factor = max(0.3, 1.0 / (1.0 + (atr_pct - 2.0) * 0.2))
            base_leverage *= volatility_factor
        
        # Apply limits with hyperoptimizable cap
        max_leverage_cap = float(self.max_leverage_cap.value)
        final_leverage = max(1.0, min(base_leverage, max_leverage, max_leverage_cap))
        
        # Ensure we return a positive integer
        return max(1, int(round(final_leverage)))