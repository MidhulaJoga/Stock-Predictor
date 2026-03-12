import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from data.fetch import fetch_stock_data

# ── 1. Data Load ──────────────────────────────
df = fetch_stock_data("AAPL", period="2y")
prices = df['Close'].values.reshape(-1, 1)

# ── 2. Normalize (0 to 1) ─────────────────────
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# ── 3. Sequences Banao (60 days → next day) ───
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(prices_scaled, SEQ_LEN)

# ── 4. Train/Test Split ───────────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Torch tensors
X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test  = torch.FloatTensor(y_test)

# ── 5. LSTM Model Define ──────────────────────
class StockLSTM(nn.Module):
    def __init__(self):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = StockLSTM()

# ── 6. Train ──────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
print("Training shuru ho raha hai...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

# ── 7. Model Save ─────────────────────────────
torch.save(model.state_dict(), 'model/model.pth')
print("Model saved!")

# ── 8. Plot Predictions ───────────────────────
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.numpy())

plt.figure(figsize=(12,5))
plt.plot(actual, label='Actual Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.savefig('model/prediction_plot.png')
plt.show()
print("Plot saved!")
