import torch
import torch.nn as nn
import numpy as np

# ── Attention Layer ───────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        weights = torch.softmax(self.attention(lstm_out), dim=1)
        # Weighted sum of all timesteps
        context = (weights * lstm_out).sum(dim=1)
        return context

# ── Attention + LSTM Model ────────────────────
class StockLSTMAttention(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, 
                 num_layers=3, dropout=0.3):
        super(StockLSTMAttention, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)       # all timesteps
        context = self.attention(lstm_out) # attention weighted
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
