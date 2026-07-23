"""
LSTM workload forecasting model.
Architecture: 2-layer stacked LSTM with dropout, trained to predict
request_rate_rps (or any target) n steps ahead.
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Two-layer stacked LSTM with a fully-connected output head.

    Args:
        input_size:  number of input features per timestep
        hidden_size: LSTM hidden units per layer
        num_layers:  number of stacked LSTM layers
        dropout:     dropout probability between LSTM layers
        output_size: number of values to predict (1 for single-step)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last timestep output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)  # (batch,)

    def init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0
