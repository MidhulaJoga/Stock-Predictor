import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from data.fetch import fetch_stock_data
from model.model_v2 import StockLSTMAttention

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df = fetch_stock_data("AAPL", period="2y")
df['RSI'] = compute_rsi(df['Close'])
df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
df.dropna(inplace=True)

features = df[['Close', 'Volume', 'High', 'Low', 'Open']].values

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

close_scaler = MinMaxScaler()
close_scaler.fit(df[['Close']].values)

SEQ_LEN = 60

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i][0])
    return np.array(X), np.array(y)

X, y = create_sequences(features_scaled, SEQ_LEN)

split = int(len(X) * 0.8)
X_train = torch.FloatTensor(X[:split])
X_test  = torch.FloatTensor(X[split:])
y_train = torch.FloatTensor(y[:split]).unsqueeze(1)
y_test  = torch.FloatTensor(y[split:]).unsqueeze(1)

model = StockLSTMAttention(input_size=5, hidden_size=128, num_layers=3, dropout=0.3)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

print("Training V2 shuru...")
best_loss = float('inf')
train_losses = []

for epoch in range(150):
    model.train()
    batch_losses = []
    for i in range(0, len(X_train), 32):
        xb = X_train[i:i+32]
        yb = y_train[i:i+32]
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_losses.append(loss.item())

    epoch_loss = np.mean(batch_losses)
    train_losses.append(epoch_loss)
    scheduler.step(epoch_loss)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'model/model_v2_best.pth')

    if (epoch+1) % 25 == 0:
        print(f"Epoch {epoch+1}/150 | Loss: {epoch_loss:.6f} | Best: {best_loss:.6f}")

model.load_state_dict(torch.load('model/model_v2_best.pth'))
model.eval()

with torch.no_grad():
    preds = model(X_test).numpy()
    preds_actual = close_scaler.inverse_transform(preds)
    actual = close_scaler.inverse_transform(y_test.numpy())

mae  = mean_absolute_error(actual, preds_actual)
rmse = np.sqrt(mean_squared_error(actual, preds_actual))
mape = np.mean(np.abs((actual - preds_actual) / actual)) * 100

# Emojis hata diye - Windows ke liye plain text
print("\n=== Model Performance ===")
print(f"MAE      : ${mae:.2f}")
print(f"RMSE     : ${rmse:.2f}")
print(f"MAPE     : {mape:.2f}%")
print(f"Accuracy : {100-mape:.2f}%")
print("=========================")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
ax1.plot(actual, label='Actual', color='blue', linewidth=2)
ax1.plot(preds_actual, label='Predicted', color='red', linewidth=2, linestyle='--')
ax1.set_title(f'AAPL - Attention LSTM | Accuracy: {100-mape:.2f}%')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_losses, color='green', linewidth=2)
ax2.set_title('Training Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model/prediction_v2.png', dpi=150)
plt.show()
print("Plot saved -> model/prediction_v2.png")
