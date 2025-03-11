import logging
import torch
import numpy as np
import random
import os
import pandas as pd
from typing import Dict, Any, Tuple

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor, DefaultPyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchLSTMModel_v2 import PyTorchLSTMModel
from freqtrade.freqai.torch.PyTorchModelTrainer_v2 import PyTorchLSTMTrainer
from datasieve.pipeline import Pipeline

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from freqtrade.freqai.freqai_interface import IFreqaiModel
from datasieve.transforms import SKLearnWrapper

class PyTorchLSTMRegressor_v2(BasePyTorchRegressor):

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, model_training_parameters=None, model_kwargs=None, config=None):

        # ✅ Ensure model_kwargs and model_training_parameters are dictionaries
        model_kwargs = model_kwargs or {}
        model_training_parameters = model_training_parameters or {}

        super().__init__(config=config)

        self.window_size = model_kwargs.get("window_size", config["freqai"]["model_kwargs"].get("window_size", 24))
        self.num_layers = model_kwargs.get("num_lstm_layers", config["freqai"]["model_kwargs"].get("num_lstm_layers", 3))
        self.dropout = model_kwargs.get("dropout_percent", config["freqai"]["model_kwargs"].get("dropout_percent", 0.2))

        self.lr = model_training_parameters.get("learning_rate", config["freqai"]["model_training_parameters"].get("learning_rate", 0.0005))
        self.weight_decay = model_training_parameters.get("weight_decay", config["freqai"]["model_training_parameters"].get("weight_decay", 0.00005))
        self.num_epochs = model_training_parameters.get("num_epochs", config["freqai"]["model_training_parameters"].get("num_epochs", 50))

        self.batch_size = model_training_parameters.get("trainer_kwargs", {}).get("batch_size", config["freqai"]["model_training_parameters"]["trainer_kwargs"].get("batch_size", 64))
        self.trainer_kwargs = model_training_parameters.get("trainer_kwargs", config["freqai"]["model_training_parameters"].get("trainer_kwargs", {}))

        self.trained_feature_count = None
        self.model = None

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        use_gpu = dk.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Using {'GPU' if self.device.type == 'cuda' else 'CPU'} for training!")

        train_features_np = data_dictionary["train_features"].values
        train_labels_np = data_dictionary["train_labels"].values

        # Detect feature count dynamically
        n_features = train_features_np.shape[1]
        
        # Reinitialize model if feature count changes
        if self.trained_feature_count != n_features:
            logger.warning(f"⚠ Feature count changed! Reinitializing model with {n_features} features.")
            self.trained_feature_count = n_features
            self.model = PyTorchLSTMModel(
                input_dim=n_features,
                output_dim=1,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

        # ✅ Correct `num_batches` calculation
        num_batches = train_features_np.shape[0] // self.window_size

        # ✅ Ensure we do not attempt to reshape more data than available
        trimmed_size = num_batches * self.window_size
        train_features_np = train_features_np[:trimmed_size]
        train_labels_np = train_labels_np[:trimmed_size]

        # ✅ Proper reshaping
        train_features_np = train_features_np.reshape(num_batches, self.window_size, n_features)
        train_labels_np = train_labels_np.reshape(num_batches, self.window_size, 1)

        logger.info(f"✅ Feature dimensions after reshaping: {train_features_np.shape}")
        logger.info(f"✅ Label dimensions after reshaping: {train_labels_np.shape}")

        # train_features_tensor = torch.tensor(train_features_np, dtype=torch.float32).to(self.device)
        # train_labels_tensor = torch.tensor(train_labels_np, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.SmoothL1Loss()
        trainer = PyTorchLSTMTrainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            data_convertor=self.data_convertor,
            tb_logger=self.tb_logger,
            window_size=self.window_size,
            **self.trainer_kwargs,
        )

        trainer.fit(data_dictionary, self.splits)
        
        # ✅ Save trained model
        model_path = os.path.join(dk.config["user_data_dir"], "models", "pytorch_lstm_v2.pth")
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"💾 Model saved at: {model_path}")

        return self.model