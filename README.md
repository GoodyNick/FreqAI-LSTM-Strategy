# FreqAI Strategy - AI-Driven Trading with LSTM

![FreqAI](https://www.freqtrade.io/en/stable/assets/freqai_algo.jpg)

## Overview
**FreqAI-Strategy** is an advanced **AI-driven trading strategy** built for [Freqtrade](https://www.freqtrade.io/en/stable/) using **Long Short-Term Memory (LSTM) neural networks**. This strategy is designed to:
- **Predict future price trends** in cryptocurrency markets using deep learning.
- **Execute trades based on AI-generated signals** to maximize profitability.
- **Adapt dynamically to market conditions** using engineered features.

🚨 **Work in Progress:** This strategy is still under active development and **is not meant for live trading with real money**. Use it for research and backtesting only.

## Features
✅ **Uses LSTM for time-series forecasting**  
✅ **Advanced feature engineering/filtering support**  
✅ **Dynamic target scaling and market regime filtering**  
✅ **Backtesting and hyperparameter tuning support**  
✅ **Automated model training and retraining**  
✅ **Optimized for training on GPU/CPU**   
✅ **Optimized for Futures market Trading**  

## Installation
### 1️⃣ Install Freqtrade(full installation with all dependencies)
```bash
# Clone and install Freqtrade
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
./setup.sh --install
```

### 2️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Clone This Repository
```bash
git clone https://github.com/GoodyNick/Freqai-Strategy.git
cd Freqai-Strategy
```

### 4️⃣ Copy Paths for Configuration and Model Files
After cloning, ensure the configuration, strategy, and model files are correctly placed in the Freqtrade directory structure. Use the following commands:
```bash
# Copy configuration file
cp config-torch-lstm_v2.json /freqtrade/user_data/configs/

# Copy strategy file
cp ExampleLSTMStrategy_v2.py /freqtrade/user_data/strategies/

# Copy model-related files
cp PyTorchLSTMModel_v2.py /freqtrade/freqtrade/freqai/torch/
cp PyTorchLSTMRegressor_v2.py /freqtrade/user_data/freqaimodels/
cp PyTorchModelTrainer_v2.py /freqtrade/freqtrade/freqai/torch/
cp freqai_interface.py /freqtrade/freqtrade/freqai/
cp BasePyTorchRegressor.py /freqtrade/freqtrade/freqai/base_models/
```
Modify them as needed before running Freqtrade.

## Configuration
Modify `config-torch-lstm_v2.json` to customize:
- **Train/Test periods** (`train_period_days`, `backtest_period_days`)
- **Feature Engineering Parameters** (DI threshold, scaling methods)
- **LSTM Model Parameters** (`hidden_dim`, `num_lstm_layers`, `dropout`, etc.)
- **Trading Settings** (Max trades, margin mode, stake size)
- **model training parameters**
- ** ... **

## Running Backtests
```bash
freqtrade backtesting --config config-torch-lstm_v2.json --strategy ExampleLSTMStrategy_v2 --freqaimodel PyTorchLSTMRegressor_v2 
```
you can also use run.sh script for backtesting, plotting, or hyperopt freqai strategy

## Contributions & Contact
🤝 **Contributions are welcome!** If you have suggestions or improvements, feel free to submit a **pull request** or open an **issue**.

📬 **Contact:** [GitHub Issues](https://github.com/GoodyNick/Freqai-Strategy/issues) or reach out on Telegram(@GoodyNick) or my email(a.goodarzinick@gmail.com)!

## Support the Project
Your support means a lot to me and helps keep this project alive! If you find this project useful or inspiring, consider making a donation(though absolutely not necessary!). Every contribution, no matter how small, is deeply meaningful and motivates me to continue improving and maintaining this work for everyone who uses it. Your donations will directly benefit the future development of this project, ensuring it remains free, open, and accessible to all. Thank you for being a part of this journey!

*Metamask wallet address:*

0x6AB1d47996f91f3Ad03071fBCE6Beb5187D2A48b

---
**License:** MIT

