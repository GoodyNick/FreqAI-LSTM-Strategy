import logging
import torch
import numpy as np
import random
import os
import pandas as pd
from typing import Dict, Any

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
    torch.backends.cudnn.benchmark = False  # We'll optionally enable it for GPU below

set_seed(42)

from freqtrade.freqai.freqai_interface import IFreqaiModel
from datasieve.transforms import SKLearnWrapper

class PyTorchLSTMRegressor_v3(BasePyTorchRegressor):

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, model_training_parameters=None, model_kwargs=None, config=None):

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

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # Enable auto-tuning on GPU for faster convs

        train_features_np = data_dictionary["train_features"].values
        train_labels_np = data_dictionary["train_labels"].values
        n_features = train_features_np.shape[1]

        logger.info(f"📐 Feature count for training: {n_features}")

        # ✅ Local-only model variable (not attached to self)
        model = PyTorchLSTMModel(
            input_dim=n_features,
            output_dim=1,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # 🔧 Dynamic batch size scaling on GPU
        if self.device.type == "cuda":
            self.trainer_kwargs["batch_size"] = max(96, self.batch_size * 2)  # e.g., 128–256
        else:
            self.trainer_kwargs["batch_size"] = self.batch_size

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.SmoothL1Loss()

        trainer = PyTorchLSTMTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            data_convertor=self.data_convertor,
            tb_logger=self.tb_logger,
            window_size=self.window_size,
            use_amp=(self.device.type == "cuda"),
            **self.trainer_kwargs,
        )

        trainer.fit(data_dictionary, self.splits)

        model_path = os.path.join(dk.config["user_data_dir"], "models", "pytorch_lstm_v2.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"💾 Model saved at: {model_path}")

        return model
