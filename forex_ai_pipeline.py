from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import numpy as np

from ta.trend import EMAIndicator
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

df = pd.read_csv("forex_price_data.csv")

df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_20"] =df["Close"].rolling(window=20).mean()

ema10 = EMAIndicator(close=df["Close"], window=10)
df["EMA_10"] = ema10.ema_indicator()

rsi = RSIIndicator(close=df["Close"], window=14)
df["RSI"] = rsi.rsi()

macd = MACD(close=df["Close"])
df["MACD"] = macd.macd()
df["MACD_Signal"] = macd.macd_signal()

bb = BollingerBands(close=df["Close"], window=20)
df["BB_High"] = bb.bollinger_hband()
df["BB_Low"] = bb.bollinger_lband()
df["BB_Middle"] = bb.bollinger_mavg()

df["Next_Close"] = df["Close"].shift(-1)
df["Target"] = np.where(df["Next_Close"] > df["Close"], 1, 0)

features = [
     "SMA_10",
     "EMA_10",
     "RSI",
     "MACD",
     "MACD_Signal",
     "BB_High",
     "BB_Low",
     "BB_Middle"
]

X = df[features]
Y = df["Target"]


df["Signal"] = np.where(
    df["SMA_20"].isna(),
     "WAIT",
      np.where(df["SMA_10"] > df["SMA_20"], "BUY", "SELL"))

data = df.dropna()

X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

print(X.shape)
print(y.shape)


print(df[["Close", "SMA_10","EMA_10",
          "RSI",
          "MACD",
          "MACD_Signal",
          "BB_High",
          "BB_Low",
          "BB_Middle",
          "Close",
          "Next_Close",
          "Target"]].head(15))


predictions = model.predict(X_test)
print(predictions[:20])

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nclassification Report:")
print(classification_report(y_test, predictions))


joblib.dump(model, "forex_model.pkl")
print("Model saved successfully!")

# Predict on the full dataset

data["Prediction"] = model.predict(X)

# Convert predicitions into trading signals
data["AI_Signal"] = data["Prediction"].map({
    1: "BUY",
    0: "SELL"
})

# Save predictions
data.to_csv("trading_signals.csv", index=False)

print("Trading signals generated successfully!")


# ==============================================
# Trading Performance Calculation
# ==============================================

# Calculate return for next candle
data["Return"] = (data["Next_Close"] - data["Close"]) / data["Close"]

# Profit depends on BUY or SELL prediction
data["Trade_Return"] = np.where(
    data["AI_Signal"] == "BUY",
    data["Return"],
    -data["Return"]
)

# Running cumulative return
data["Cumulative_Return"] = (1 + data["Trade_Return"]).cumprod()

# Save updated file
data.to_csv("trading_signals.csv", index=False)

print("Trading performance calculated successfully!")