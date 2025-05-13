import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)

class PyTorchLSTMModel(nn.Module):
    """
    A Long Short-Term Memory (LSTM) model implemented using PyTorch.

    This class serves as a complex example for the integration of PyTorch models.
    It is designed to handle sequential data and capture long-term dependencies.

    :param input_dim: The number of input features.
    :param output_dim: The number of output classes.
    :param num_layers: The number of LSTM layers.
    :param dropout: Dropout rate for regularization.

    :returns: The output of the LSTM, with shape (batch_size, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int, dropout: float):
        super(PyTorchLSTMModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Initialize LSTM layers dynamically based on input feature count
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_dim if i == 0 else 128  # First layer uses input_dim, others use 128
            self.lstm_layers.append(
                nn.LSTM(input_size=layer_input_size, hidden_size=128, num_layers=1, batch_first=True)
            )

        # Batch Normalization & Dropout layers
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(128, affine=True) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.alpha_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        x_res = None  # Initialize residual connection variable

        for i in range(self.num_layers):
            self.lstm_layers[i].flatten_parameters()
            x, _ = self.lstm_layers[i](x)  # LSTM Forward Pass

            # Apply BatchNorm1d correctly: (batch_size, features, seq_len) → normalize → revert shape
            x = self.batch_norms[i](x.transpose(1, 2)).transpose(1, 2)

            x = self.dropouts[i](x)  # Apply dropout after batch norm

            # Residual Connection
            if x_res is not None:
                x = x + x_res  # Skip connection
            x_res = x  # Store for next layer

        # Fully Connected Layers
        x = self.relu(self.fc1(x[:, -1, :]))  # Use last LSTM output only
        x = self.alpha_dropout(x)
        x = self.fc2(x)
        
        return x
    
    def save(self, save_path: str):
        """
        Saves the PyTorch model state dictionary.
        
        :param save_path: Path to save the model.
        """
        logger.info(f"💾 Saving model to {save_path}")
        torch.save(self.state_dict(), save_path)

    @classmethod
    def load(cls, load_path: str, input_dim: int, output_dim: int, num_layers: int, dropout: float):
        """
        Loads a saved model and initializes the class.
        
        :param load_path: Path to the saved model.
        :param input_dim: Number of input features.
        :param output_dim: Number of output features.
        :param num_layers: Number of LSTM layers.
        :param dropout: Dropout rate.
        :return: Loaded PyTorchLSTMModel instance.
        """
        logger.info(f"🔄 Loading model from {load_path}")
        model = cls(input_dim, output_dim, num_layers, dropout)
        model.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
        model.eval()  # Ensure model is in evaluation mode
        return model

    @classmethod
    def load_from_checkpoint(cls, checkpoint):
        """
        Load model from checkpoint for FreqAI compatibility.
        This method creates a wrapper object with a .model attribute
        that meets BasePyTorchRegressor.predict() expectations.
        
        Args:
            checkpoint: Dictionary containing the model (under 'pytrainer' key)
        
        Returns:
            A wrapper object with a .model attribute pointing to the actual model
        """
        if isinstance(checkpoint, dict) and "pytrainer" in checkpoint:
            # Create a wrapper that will satisfy BasePyTorchRegressor.predict()
            # which calls self.model.model.eval()
            class PyTorchModelWrapper:
                def __init__(self, inner_model):
                    self.model = inner_model
                    
                def eval(self):
                    """Required for compatibility with FreqAI's check_if_model_exists"""
                    self.model.eval()
                    return self
                    
                def train(self, mode=True):
                    """Required for PyTorch model interface"""
                    self.model.train(mode)
                    return self
            
            # Get the actual PyTorchLSTMModel from the checkpoint
            model = checkpoint["pytrainer"]
            
            # Return a wrapper that has the actual model in its .model attribute
            return PyTorchModelWrapper(model)
        else:
            raise ValueError("Invalid checkpoint format. Expected dict with 'pytrainer' key.")