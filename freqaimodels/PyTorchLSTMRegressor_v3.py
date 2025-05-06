import logging
import torch
import numpy as np
import random
import os
import pandas as pd
import json
from pathlib import Path
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

class ModelContainer:
    def __init__(self, pytorch_model_instance):
        self.model = pytorch_model_instance  # The nn.Module instance

    def save(self, path: str):
        """
        Delegates saving to the underlying PyTorchLSTMModel's save method
        or saves the state_dict directly.
        """
        if hasattr(self.model, 'save') and callable(self.model.save):
            # This will call PyTorchLSTMModel.save() which saves the state_dict
            self.model.save(path)
        else:
            # Fallback: should not be strictly needed if PyTorchLSTMModel.save exists
            logger.warning(f"ModelContainer: Underlying model has no 'save' method. Saving state_dict directly to {path}.")
            torch.save(self.model.state_dict(), path)

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

        self.loss_function_name = model_training_parameters.get(
            "loss_function", config["freqai"]["model_training_parameters"].get("loss_function", "smoothl1")
        ).lower()

        self.trained_feature_count = None
        # self.model is initialized by BasePyTorchRegressor or set by fit/load_model
        # No self.device here, which is good for pickling self.

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        logger.info("🔥 fit() called, starting training.")
        use_gpu = dk.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)
        # device is a local variable, not self.device
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Using {'GPU' if device.type == 'cuda' else 'CPU'} for training!")

        train_features_np = data_dictionary["train_features"].values
        n_features = train_features_np.shape[1]
        self.trained_feature_count = n_features

        logger.info(f"📐 Feature count for training: {n_features}")

        pytorch_model_instance = PyTorchLSTMModel(
            input_dim=n_features,
            output_dim=1,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        current_batch_size_fit = self.batch_size
        if device.type == "cuda":
            current_batch_size_fit = max(96, self.batch_size * 2)
        
        trainer_kwargs_fit = self.trainer_kwargs.copy()
        trainer_kwargs_fit["batch_size"] = current_batch_size_fit

        optimizer = torch.optim.AdamW(pytorch_model_instance.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        criterion: torch.nn.Module
        if self.loss_function_name == 'mae' or self.loss_function_name == 'l1':
            criterion = torch.nn.L1Loss()
            logger.info("Using MAE (L1Loss) loss function.")
        elif self.loss_function_name == 'mse':
            criterion = torch.nn.MSELoss()
            logger.info("Using MSE loss function.")
        elif self.loss_function_name == 'smoothl1':
            criterion = torch.nn.SmoothL1Loss()
            logger.info("Using SmoothL1Loss loss function.")
        else:
            logger.warning(f"Unsupported loss function '{self.loss_function_name}'. Defaulting to SmoothL1Loss.")
            criterion = torch.nn.SmoothL1Loss()

        trainer = PyTorchLSTMTrainer(
            model=pytorch_model_instance,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            data_convertor=self.data_convertor,
            tb_logger=self.tb_logger,
            window_size=self.window_size,
            use_amp=(device.type == "cuda"),
            **trainer_kwargs_fit,
        )

        trainer.fit(data_dictionary, getattr(self, 'splits', ['train', 'test']))

        container = ModelContainer(pytorch_model_instance)
        
        del pytorch_model_instance, optimizer, criterion, trainer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return container

    def load_model(self, path: str) -> Any:
        logger.info(f"🔄 PyTorchLSTMRegressor_v3 loading model from path: {str(path)}")
        
        model_path = Path(path)
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        use_gpu = self.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)
        # device is a local variable
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Using {'GPU' if device.type == 'cuda' else 'CPU'} for loading model!")

        if model_path.name.endswith(".zip"):
            meta_filename = model_path.stem + "_meta.json"
        else:
            meta_filename = model_path.name.replace(model_path.suffix, "") + "_meta.json"
        
        metadata_path = model_path.parent / meta_filename
        input_dim = None

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                input_dim = metadata.get("n_features_train")
                if input_dim is not None:
                    logger.info(f"Loaded input_dim (n_features_train): {input_dim} from metadata {metadata_path}")
                else:
                    logger.warning(f"Could not find 'n_features_train' in metadata {metadata_path}.")
            except Exception as e:
                logger.exception(f"Error loading or parsing metadata file {metadata_path}: {e}")
        else:
            logger.warning(f"Metadata file not found at {metadata_path}. Cannot determine input_dim for model reconstruction.")

        if input_dim is None:
            if self.trained_feature_count is not None:
                input_dim = self.trained_feature_count
                logger.warning(f"Using self.trained_feature_count as input_dim: {input_dim}. This might be unreliable if not from current training session's metadata.")
            else:
                logger.error("Failed to determine input_dim for model loading. Check metadata or model saving process.")
                raise ValueError("Cannot load model: input_dim is unknown.")
        
        try:
            state_dict = super()._load_torch_model(model_path, device=device)
        except Exception as e:
            logger.exception(f"Error loading state_dict using super()._load_torch_model from {model_path}: {e}")
            raise

        loaded_pytorch_model = PyTorchLSTMModel(
            input_dim=input_dim,
            output_dim=1,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        loaded_pytorch_model.load_state_dict(state_dict)
        loaded_pytorch_model.eval()

        logger.info(f"✅ PyTorchLSTMModel successfully loaded and state_dict applied from {model_path}")

        container = ModelContainer(loaded_pytorch_model)
        return container
