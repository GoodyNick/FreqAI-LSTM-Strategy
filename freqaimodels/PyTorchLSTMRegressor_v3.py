import logging
import torch
import numpy as np
import random
import os
import json
from pathlib import Path
from typing import Dict, Any

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor, DefaultPyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchLSTMModel_v3 import PyTorchLSTMModel
from freqtrade.freqai.torch.PyTorchModelTrainer_v2 import PyTorchLSTMTrainer
from datasieve.pipeline import Pipeline

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Keep this False for reproducibility

set_seed(42)

class ModelContainer:
    def __init__(self, model_instance):
        self.model = model_instance
        
    def eval(self):
        """Delegate to the model's eval method"""
        return self.model.eval()
        
    def train(self, mode=True):
        """Delegate to the model's train method"""
        return self.model.train(mode)
        
    def __call__(self, *args, **kwargs):
        """Delegate to the model's __call__ method"""
        return self.model(*args, **kwargs)
    
    def save(self, path):
        """
        Save the model properly structured for FreqAI.
        FreqAI expects a zipfile with a "pytrainer" key.
        """
        # Create a dictionary with the key "pytrainer" that points to your model
        save_dict = {"pytrainer": self.model}
        
        # Save it to the path
        torch.save(save_dict, path)
        logger.info(f"💾 Model saved to: {path} with FreqAI format (includes 'pytrainer' key)")

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

        # ✅ Read loss function name from config
        self.loss_function_name = model_training_parameters.get(
            "loss_function", config["freqai"]["model_training_parameters"].get("loss_function", "smoothl1")
        ).lower()

        self.trained_feature_count = None
        self.model = None

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        logger.info("🔥 fit() called, starting training.")
        use_gpu = dk.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Using {'GPU' if device.type == 'cuda' else 'CPU'} for training!")

        # Comment out this line for hyperopt compatibility
        # if device.type == "cuda":
        #     torch.backends.cudnn.benchmark = True

        metadata = kwargs.get("metadata", {})

        train_features_np = data_dictionary["train_features"].values
        train_labels_np = data_dictionary["train_labels"].values
        n_features = train_features_np.shape[1]
        self.trained_feature_count = n_features  # Store for later use in load_model

        logger.info(f"📐 Feature count for training: {n_features}")

        model = PyTorchLSTMModel(
            input_dim=n_features,
            output_dim=1,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        if device.type == "cuda":
            self.trainer_kwargs["batch_size"] = max(96, self.batch_size * 2)
        else:
            self.trainer_kwargs["batch_size"] = self.batch_size

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # ✅ SELECT CRITERION BASED ON CONFIG
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
            # Default or raise error if unsupported
            logger.warning(f"Unsupported loss function '{self.loss_function_name}'. Defaulting to SmoothL1Loss.")
            criterion = torch.nn.SmoothL1Loss()

        trainer = PyTorchLSTMTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            data_convertor=self.data_convertor,
            tb_logger=self.tb_logger,
            window_size=self.window_size,
            use_amp=(device.type == "cuda"),
            **self.trainer_kwargs,
        )

        trainer.fit(data_dictionary, self.splits)

        # Wrap the model in our container
        container = ModelContainer(model)

        # No need to save the model separately - FreqAI will call container.save()
        # which will save in the {"pytrainer": model} format

        # Clean up any tensors 
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Return the container
        return container

    def load_model(self, path: str) -> Any:
        """
        Load model from a state_dict file. This is called by FreqAI when a model is being loaded.
        """
        logger.info(f"🔄 Loading model from path: {str(path)}")
        
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Determine device
        use_gpu = self.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Get input_dim from metadata
        # ... existing metadata loading code ...
        
        try:
            # Load the zipfile which contains a dict with "pytrainer" key
            zipfile_dict = torch.load(model_path, map_location=device)
            
            if "pytrainer" in zipfile_dict:
                # FreqAI format - contains a "pytrainer" key
                model_loaded = zipfile_dict["pytrainer"]
                logger.info("Successfully loaded model from FreqAI format with 'pytrainer' key")
                container = ModelContainer(model_loaded)
                return container
            else:
                # Old format - direct state_dict or model
                logger.warning("Model not in FreqAI format (missing 'pytrainer' key). "
                              "Falling back to legacy loading...")
                
                # Load state dict using BasePyTorchRegressor's helper
                state_dict = super()._load_torch_model(model_path, device=device)
                
                # Create model instance
                model = PyTorchLSTMModel(
                    input_dim=self.input_dim,
                    output_dim=1,
                    num_layers=self.num_layers,
                    dropout=self.dropout
                ).to(device)
                
                # Load state dict into model
                model.load_state_dict(state_dict)
                model.eval()  # Set to evaluation mode
                
                # Wrap in container and return
                container = ModelContainer(model)
                return container
                
        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            raise

    # predict method is inherited from BasePyTorchRegressor
    # The container approach ensures its calls to self.model.model.eval() will work
