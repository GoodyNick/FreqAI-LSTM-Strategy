import logging
import torch
import torch.nn as nn
import numpy as np
import random
import os
import pandas as pd
from typing import Dict, Any, Tuple
from tqdm import tqdm # For progress bar during training

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor, DefaultPyTorchDataConvertor
# REMOVED: from freqtrade.freqai.torch.PyTorchLSTMModel_v2 import PyTorchLSTMModel
# REMOVED: from freqtrade.freqai.torch.PyTorchModelTrainer_v2 import PyTorchLSTMTrainer
# REMOVED: from datasieve.pipeline import Pipeline # Assuming this was specific to old structure

logger = logging.getLogger(__name__)

# --- Seed setting (Good practice) ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Consider removing deterministic for potential speedup if exact reproducibility isn't critical
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Define the Model Architecture INSIDE this file ---
class PyTorchLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super(PyTorchLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # We need to pass hidden and cell state through LSTM
        # LSTM outputs: output, (hidden state, cell state)
        out, _ = self.lstm(x, (h0.detach(), c0.detach())) # Use detach() to prevent backprop through init states

        # Index hidden state of the last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> corresponds to hidden state at last time step
        out = self.fc(out[:, -1, :])
        return out

# --- Define the Main Regressor Class ---
class PyTorchLSTMRegressor_v4(BasePyTorchRegressor):

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        # Use the default convertor, assuming input data is suitable
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, model_training_parameters=None, model_kwargs=None, config=None):
        super().__init__(config=config)
        self.tb_logger = None # Disable Tensorboard for hyperopt compatibility

        freqai_config = config.get('freqai', {})
        model_kwargs_config = freqai_config.get('model_kwargs', {})
        model_training_parameters_config = freqai_config.get('model_training_parameters', {})
        trainer_kwargs_config = model_training_parameters_config.get('trainer_kwargs', {})

        # Model Architecture Params
        self.window_size = model_kwargs_config.get("window_size", 24)
        self.hidden_dim = model_kwargs_config.get("hidden_dim", 64) # Added hidden_dim
        self.num_layers = model_kwargs_config.get("num_lstm_layers", 3)
        self.dropout = model_kwargs_config.get("dropout_percent", 0.2)

        # Training Params
        self.lr = model_training_parameters_config.get("learning_rate", 0.0005)
        self.weight_decay = model_training_parameters_config.get("weight_decay", 0.00005)
        self.num_epochs = model_training_parameters_config.get("num_epochs", 50)
        self.batch_size = trainer_kwargs_config.get("batch_size", 64)
        self.loss_function_name = model_training_parameters_config.get("loss_function", "smoothl1").lower()
        # Add any other trainer_kwargs you might need from the config

        self.trained_feature_count = None
        # No self.model initialization needed here

        logger.info(f"Initialized PyTorchLSTMRegressor_v3 with window_size: {self.window_size}, "
                    f"hidden_dim: {self.hidden_dim}, num_layers: {self.num_layers}, dropout: {self.dropout}, "
                    f"lr: {self.lr}, epochs: {self.num_epochs}, batch_size: {self.batch_size}, "
                    f"loss: {self.loss_function_name}")

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        logger.info("🔥 fit() called, starting training.")
        # --- Determine device LOCALLY ---
        use_gpu = dk.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Using {'GPU' if device.type == 'cuda' else 'CPU'} for training!")

        # --- Disable cudnn.benchmark temporarily ---
        # if device.type == "cuda":
        #     torch.backends.cudnn.benchmark = True # Temporarily disabled

        # --- Prepare Data ---
        (train_features, train_labels) = self.data_convertor.convert_data(
            data_dictionary["train_features"],
            dk.training_features_list,
            self.window_size,
            device,
            labels=data_dictionary["train_labels"]
        )
        n_features = train_features.shape[-1]
        logger.info(f"📐 Feature count for training: {n_features}")
        self.trained_feature_count = n_features

        # --- Instantiate LOCAL Model ---
        model = PyTorchLSTMModel(
            input_dim=n_features,
            output_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        # --- Setup Optimizer and Criterion ---
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # ... (criterion selection logic remains the same) ...
        if self.loss_function_name == 'mae' or self.loss_function_name == 'l1':
            criterion = nn.L1Loss()
        elif self.loss_function_name == 'mse':
            criterion = nn.MSELoss()
        elif self.loss_function_name == 'smoothl1':
            criterion = nn.SmoothL1Loss()
        else:
            logger.warning(f"Unsupported loss function '{self.loss_function_name}'. Defaulting to SmoothL1Loss.")
            criterion = nn.SmoothL1Loss()


        # --- Training Loop ---
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_batch_size = self.batch_size * 2 if device.type == 'cuda' else self.batch_size
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            batch_count = 0

            # --- Remove tqdm wrapper ---
            # pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)
            # for i, (batch_features, batch_labels) in enumerate(pbar):
            for i, (batch_features, batch_labels) in enumerate(train_loader): # Iterate directly
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1
                # pbar.set_postfix({'loss': running_loss / batch_count}) # Remove tqdm postfix

            avg_epoch_loss = running_loss / batch_count
            # Log loss less frequently if needed without tqdm
            if (epoch + 1) % 10 == 0 or epoch == self.num_epochs - 1: # Log every 10 epochs or last epoch
                 logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_epoch_loss:.6f}")

        logger.info("✅ Training finished.")

        # --- Save the model state dictionary ---
        model_path = os.path.join(dk.config["user_data_dir"], "models", "pytorch_lstm_v3_consolidated.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"💾 Model state_dict saved at: {model_path}")

        # --- Clean up GPU memory ---
        del model, optimizer, criterion, train_features, train_labels, dataset, train_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # --- Return None ---
        return None

    # --- predict method remains the same (but also remove tqdm if used there) ---
    def predict(self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
        logger.info(" Preditct() called")
        # --- Determine device LOCALLY ---
        use_gpu = dk.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Using {'GPU' if device.type == 'cuda' else 'CPU'} for prediction!")

        # --- Get expected feature count ---
        if not hasattr(self, 'trained_feature_count') or self.trained_feature_count is None:
             raise ValueError("Trained feature count not available. Was fit() called?")

        # --- Instantiate LOCAL Model Architecture ---
        model = PyTorchLSTMModel(
            input_dim=self.trained_feature_count,
            output_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        # --- Load state dict ---
        model_path = os.path.join(dk.config["user_data_dir"], "models", "pytorch_lstm_v3_consolidated.pth")
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                logger.info(f"✅ Model loaded successfully from {model_path}")
            except Exception as e:
                logger.exception(f"❌ Failed to load model from {model_path}: {e}")
                default_preds = np.zeros(len(unfiltered_df))
                return unfiltered_df, default_preds
        else:
            logger.error(f"❌ Model file not found at {model_path}. Cannot predict.")
            default_preds = np.zeros(len(unfiltered_df))
            return unfiltered_df, default_preds

        # --- Move model to device and set to eval mode ---
        model.to(device)
        model.eval()

        # --- Prepare data for prediction ---
        (pred_features, _) = self.data_convertor.convert_data(
            unfiltered_df, dk.training_features_list, self.window_size, device, labels=None
        )

        # --- Make predictions ---
        predictions = np.array([])
        pred_batch_size = self.batch_size * 4
        dataset = torch.utils.data.TensorDataset(pred_features)
        pred_loader = torch.utils.data.DataLoader(dataset, batch_size=pred_batch_size, shuffle=False)

        logger.info(f"Starting prediction with batch size {pred_batch_size}...")
        with torch.no_grad():
            # --- Remove tqdm wrapper ---
            # for (batch_features,) in tqdm(pred_loader, desc="Predicting", leave=False):
            for (batch_features,) in pred_loader: # Iterate directly
                 batch_features = batch_features.to(device)
                 outputs = model(batch_features)
                 batch_preds = outputs.cpu().numpy().squeeze()
                 if batch_preds.ndim == 0:
                     batch_preds = np.expand_dims(batch_preds, axis=0)
                 predictions = np.append(predictions, batch_preds)

        logger.info("✅ Prediction finished.")

        # --- Align predictions ---
        # ... (alignment logic remains the same) ...
        if len(predictions) != len(unfiltered_df):
             if len(predictions) > len(unfiltered_df):
                  logger.warning("More predictions than dataframe rows, truncating predictions.")
                  predictions = predictions[:len(unfiltered_df)]
             elif len(predictions) < len(unfiltered_df):
                  logger.warning("Fewer predictions than dataframe rows, aligning to tail.")
                  full_preds = np.full(len(unfiltered_df), np.nan)
                  full_preds[-len(predictions):] = predictions
                  predictions = full_preds

        pred_df = unfiltered_df.copy()
        assign_length = min(len(pred_df), len(predictions))
        pred_df[dk.label_list[0]] = predictions[-assign_length:]


        # --- Clean up GPU memory ---
        del model, pred_features, dataset, pred_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return pred_df, predictions
