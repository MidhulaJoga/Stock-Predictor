import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import sys
sys.path.append('.')
from model.model_v2 import StockLSTMAttention

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

STOCKS = {
    "🇺🇸 US Technology": {
        "Apple Inc. (AAPL)":   "AAPL",
        "Microsoft (MSFT)":    "MSFT",
        "NVIDIA (NVDA)":       "NVDA",
        "Amazon (AMZN)":       "AMZN",
        "Tesla (TSLA)":        "TSLA",
    },
    "🇮🇳 Indian IT & Banking": {
        "Reliance Industries": "RELIANCE.NS",
        "TCS":                 "TCS.NS",
        "Infosys":             "INFY.NS",
        "HDFC Bank":           "HDFCBANK.NS",
        "ICICI Bank":          "ICICIBANK.NS",
    },
    "🥇 Commodities": {
        "Gold":   "GC=F",
        "Silver": "SI=F",
    },
}

USD_TO_INR   = 83.5
TROY_OZ_TO_G = 31.1035

def get_symbol(ticker):
    if ticker in ["GC=F", "SI=F"] or ticker.endswith(".NS"):
        return "Rs."
    return "$"

def convert_to_display(price, ticker):
    if ticker in ["GC=F", "SI=F"]:
        return (price * USD_TO_INR) / TROY_OZ_TO_G
    return price

st.sidebar.image("https://img.icons8.com/fluency/96/stocks.png", width=60)
st.sidebar.title("AI Stock Predictor")
st.sidebar.markdown("---")

category     = st.sidebar.selectbox("Select Market", list(STOCKS.keys()))
stock_name   = st.sidebar.selectbox("Select Stock", list(STOCKS[category].keys()))
ticker       = STOCKS[category][stock_name]
symbol       = get_symbol(ticker)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
period       = st.sidebar.selectbox("Training Data Period", ["1y", "2y", "5y"], index=1)
predict_days = st.sidebar.slider("Forecast Days", 1, 30, 7)
epochs       = st.sidebar.slider("Training Epochs", 50, 200, 150, step=25)
st.sidebar.markdown("---")
st.sidebar.info("Uses LSTM + Attention neural network trained on live market data via Yahoo Finance API.")

st.title("📈 AI Stock Price Predictor")
st.markdown(f"### {stock_name} — `{ticker}`")
if ticker in ["GC=F", "SI=F"]:
    st.caption("Prices shown in Rs. per gram (converted from USD/troy oz)")
elif ticker.endswith(".NS"):
    st.caption("Prices shown in Indian Rupees (Rs.)")
else:
    st.caption("Prices shown in USD ($)")
st.markdown("---")

try:
    t    = yf.Ticker(ticker)
    info = t.info

    if ticker in ["GC=F", "SI=F"]:
        raw   = info.get("regularMarketPrice") or info.get("previousClose") or t.history(period="1d")["Close"].iloc[-1]
        high  = info.get("fiftyTwoWeekHigh", 0)
        low   = info.get("fiftyTwoWeekLow", 0)
        curr  = convert_to_display(raw, ticker)
        h52   = convert_to_display(high, ticker)
        l52   = convert_to_display(low, ticker)
        label = "Gold" if ticker == "GC=F" else "Silver"
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Current {label} Price (per gram)", f"{symbol}{curr:,.2f}")
        col2.metric("52W High (per gram)",               f"{symbol}{h52:,.2f}")
        col3.metric("52W Low (per gram)",                f"{symbol}{l52:,.2f}")
    elif ticker.endswith(".NS"):
        raw  = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        high = info.get("fiftyTwoWeekHigh", 0)
        low  = info.get("fiftyTwoWeekLow", 0)
        cap  = info.get("marketCap", 0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"{symbol}{raw:,.2f}")
        col2.metric("52W High",      f"{symbol}{high:,.2f}")
        col3.metric("52W Low",       f"{symbol}{low:,.2f}")
        col4.metric("Market Cap",    f"{symbol}{cap/1e9:.1f}B")
    else:
        raw  = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        high = info.get("fiftyTwoWeekHigh", 0)
        low  = info.get("fiftyTwoWeekLow", 0)
        cap  = info.get("marketCap", 0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"{symbol}{raw:,.2f}")
        col2.metric("52W High",      f"{symbol}{high:,.2f}")
        col3.metric("52W Low",       f"{symbol}{low:,.2f}")
        col4.metric("Market Cap",    f"{symbol}{cap/1e9:.1f}B")
except Exception as e:
    st.warning(f"Live info unavailable: {e}")

st.markdown("---")

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = -delta.clip(upper=0).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i][0])
    return np.array(X), np.array(y)

if st.button("Run Prediction", type="primary", use_container_width=True):

    with st.spinner("Fetching live market data..."):
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            st.error("No data found. Please try another stock.")
            st.stop()
        for col in ['Close', 'High', 'Low', 'Open']:
            df[col] = df[col].apply(lambda x: convert_to_display(x, ticker))
        df['RSI']  = compute_rsi(df['Close'])
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df.dropna(inplace=True)

    st.success(f"Loaded {len(df)} trading days of data!")

    features        = df[['Close', 'Volume', 'High', 'Low', 'Open']].values
    scaler          = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    close_scaler    = MinMaxScaler()
    close_scaler.fit(df[['Close']].values)

    X, y  = create_sequences(features_scaled, 60)
    split = int(len(X) * 0.8)

    X_train = torch.FloatTensor(X[:split])
    X_test  = torch.FloatTensor(X[split:])
    y_train = torch.FloatTensor(y[:split]).unsqueeze(1)
    y_test  = torch.FloatTensor(y[split:]).unsqueeze(1)

    model     = StockLSTMAttention(input_size=5, hidden_size=128, num_layers=3, dropout=0.3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    progress_bar = st.progress(0)
    status_text  = st.empty()
    best_loss    = float('inf')

    for epoch in range(epochs):
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
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'model/model_v2_best.pth')

        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Training... Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    status_text.text("Training complete!")

    model.load_state_dict(torch.load('model/model_v2_best.pth'))
    model.eval()

    with torch.no_grad():
        preds_actual = close_scaler.inverse_transform(model(X_test).numpy())
        actual       = close_scaler.inverse_transform(y_test.numpy())

        last_seq     = torch.FloatTensor(features_scaled[-60:]).unsqueeze(0)
        future_preds = []
        for _ in range(predict_days):
            pred = model(last_seq)
            future_preds.append(pred.item())
            new_row = last_seq[:, -1:, :].clone()
            new_row[:, :, 0] = pred
            last_seq = torch.cat([last_seq[:, 1:, :], new_row], dim=1)
        future_preds = close_scaler.inverse_transform(
            np.array(future_preds).reshape(-1, 1))

    current_price = df['Close'].values[-1]

    # ── Detailed Forecast Table ───────────────
    st.markdown("---")
    st.subheader("Detailed Forecast Table")

    last_date    = df.index[-1]
    future_dates = []
    next_date    = last_date + timedelta(days=1)
    while len(future_dates) < predict_days:
        if next_date.weekday() < 5:
            future_dates.append(next_date)
        next_date += timedelta(days=1)

    table_data = []
    for i, (date, price) in enumerate(zip(future_dates, future_preds.flatten())):
        change     = price - current_price
        change_pct = (change / current_price) * 100
        signal     = "🟢 BUY" if change_pct > 1 else ("🔴 SELL" if change_pct < -1 else "🟡 HOLD")
        table_data.append({
            "Day":             f"Day {i+1}",
            "Date":            date.strftime("%d %b %Y"),
            "Predicted Price": f"{symbol}{price:,.2f}",
            "Change":          f"{symbol}{change:+,.2f}",
            "Change %":        f"{change_pct:+.2f}%",
            "Signal":          signal,
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── Chart ─────────────────────────────────
    st.markdown("---")
    st.subheader("Prediction Chart")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=actual.flatten(), name="Actual",
        line=dict(color="#00ffe1", width=2)))
    fig.add_trace(go.Scatter(
        y=preds_actual.flatten(), name="Predicted",
        line=dict(color="#ff4466", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=list(range(len(actual), len(actual) + predict_days)),
        y=future_preds.flatten(), name="Forecast",
        line=dict(color="#ffaa00", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(255,170,0,0.05)"))

    yaxis_label = f"Price ({symbol}/gram)" if ticker in ["GC=F", "SI=F"] else f"Price ({symbol})"
    fig.update_layout(
        template="plotly_dark", height=500,
        xaxis_title="Days",
        yaxis_title=yaxis_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Raw Historical Data"):
        st.dataframe(df[['Close', 'High', 'Low', 'Open', 'Volume']].tail(30), use_container_width=True)

    st.success("Prediction complete!")
