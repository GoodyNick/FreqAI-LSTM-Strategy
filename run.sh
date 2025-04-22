#!/usr/bin/env bash

# Define variables
STRATEGY="LSTMStrategy_v34"
MODEL="PyTorchLSTMRegressor_v3"
HYPEROPTMODEL="PyTorchLSTMRegressor_v2"
CONFIG="./user_data/configs/config-torch-lstm_v34_trade.json"
TIMERANGE="20240420-20250420"
PAIR="BTC/USDT:USDT"
HYPEROPTLOSS="MultiMetricHyperOptLoss"
# HYPEROPTLOSS="ShortTradeDurHyperOptLoss"
# HYPEROPTLOSS="SortinoHyperOptLoss"
# HYPEROPTLOSS="CalmarHyperOptLoss"

# Check input arguments
MODE="$1"
OPTION="$2"  # Captures the optional second argument (e.g., "clean")

# ✅ Optional Cleanup if "clean" argument is added
if [[ "$OPTION" == "clean" ]]; then
    echo "🧹 Cleaning old models and results..."
    rm -rf ./user_data/models/*
    rm -rf ./user_data/backtest_results/*
    rm -rf ./user_data/hyperopts/*
    rm -rf ./user_data/hyperopt_results/*
    rm -f ./user_data/ridge_model.pkl
    rm -f ./user_data/feature_importances.csv
    rm -rf ./user_data/strategies/*.json
    echo "✅ Cleanup complete!"
fi

# ✅ Backtesting
if [[ "$MODE" == "backtest" ]]; then
    echo "🔄 Running Backtest..."
    freqtrade backtesting -s "$STRATEGY" --freqaimodel "$MODEL" -c "$CONFIG" --timerange="$TIMERANGE" 2>&1 | tee ./user_data/backtest_results.txt
    # echo "📊 Plotting DataFrame..."
    # freqtrade plot-dataframe --strategy "$STRATEGY" --freqaimodel "$MODEL" --timerange="$TIMERANGE" --config "$CONFIG" --pair "$PAIR"

# ✅ Plotting
elif [[ "$MODE" == "plot" ]]; then
    echo "📊 Plotting DataFrame..."
    freqtrade plot-dataframe --strategy "$STRATEGY" --freqaimodel "$MODEL" --timerange="$TIMERANGE" --config "$CONFIG" --pair "$PAIR"

# ✅ Hyperopt
elif [[ "$MODE" == "hyperopt" ]]; then
    echo "🔍 Running Hyperopt..."
    freqtrade hyperopt -s "$STRATEGY" --freqaimodel "$HYPEROPTMODEL" -c "$CONFIG" --timerange="$TIMERANGE" --hyperopt-loss "$HYPEROPTLOSS" --spaces buy sell trades -e 1000 -j 16 --random-state 43

# ✅ Trade/Dry-run
elif [[ "$MODE" == "trade" ]]; then
    freqtrade trade -s "$STRATEGY" --freqaimodel "$MODEL" -c "$CONFIG"

# ✅ Download required market data
elif [[ "$MODE" == "download" ]]; then
    # freqtrade download-data -c "$CONFIG" --days 1460 -t 1h 2h 4h 1d --prepend
    freqtrade download-data -c "$CONFIG" --days 365 -t 5m 15m 30m 1h 2h 4h 1d 

# ✅ Invalid Argument Handling
else
    echo "❌ Invalid option! Use: backtest [clean], plot, hyperopt, or download."
fi