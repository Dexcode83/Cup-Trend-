import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy import stats
from tvDatafeed import TvDatafeed, Interval
from scipy.signal import argrelextrema

tv = TvDatafeed()

# =========================================
# 📊 BIST Tüm Hisseler Fonksiyonu
# =========================================
def bist_tum_hisseler():
    url = "https://scanner.tradingview.com/turkey/scan"
    payload = {
        "filter": [{"left": "exchange", "operation": "equal", "right": "BIST"}],
        "options": {"lang": "tr"},
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name"]
    }
    r = requests.post(url, json=payload, timeout=15)
    data = r.json()
    hisseler = [item["d"][0].replace("BIST:", "") for item in data["data"]]
    hisseler = [h.strip().upper() for h in hisseler if len(h) <= 5 and h.strip()]
    return list(set(hisseler))

# =========================================
# 📈 Veri Çekme Fonksiyonu
# =========================================
def Stock_Prices(Hisse):
    df = tv.get_hist(symbol=Hisse, exchange="BIST", interval=Interval.in_daily, n_bars=200)
    if df is None or len(df) < 50:
        return None
    df.reset_index(inplace=True)
    df.rename(columns={'close':'Close','volume':'Volume','open':'Open','high':'High','low':'Low'}, inplace=True)
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================================
# 📉 Dip Tespiti
# =========================================
def detect_dips(prices, order=5):
    minima = argrelextrema(prices.values, np.less, order=order)[0]
    dips = prices.iloc[minima]
    return dips

# =========================================
# 📌 Formasyon Tespiti
# =========================================
def detect_pattern(df):
    recent = df.tail(100).copy()
    prices, highs, lows = recent['Close'].values, recent['High'].values, recent['Low'].values
    sma20_val = recent['SMA_20'].iloc[-1]
    sma50_val = recent['SMA_50'].iloc[-1]
    price = prices[-1]

    # Çanak formasyonu
    left_top = highs[:30].max()
    dip = lows[30:70].min()
    right_top = highs[70:].max()
    is_cup = abs(left_top - right_top) / left_top < 0.08 and dip < (left_top * 0.85)
    is_handle = prices[-15:].mean() < right_top and prices[-15:].min() > dip
    if is_cup and is_handle:
        return "Kulplu Çanak (Cup with Handle)", 90
    elif is_cup:
        return "Çanak Formasyonu (Cup)", 85

    # Çift Dip
    minima = argrelextrema(prices, np.less, order=5)[0]
    if len(minima) >= 2 and abs(prices[minima[-1]] - prices[minima[-2]])/prices[minima[-1]] < 0.05:
        return "Çift Dip (Double Bottom)", 85

    # Çift Tepe
    maxima = argrelextrema(prices, np.greater, order=5)[0]
    if len(maxima) >= 2 and abs(prices[maxima[-1]] - prices[maxima[-2]])/prices[maxima[-1]] < 0.05:
        return "Çift Tepe (Double Top)", 85

    # Omuz-Baş-Omuz
    if len(maxima) >= 3:
        left, head, right = prices[maxima[0]], prices[maxima[1]], prices[maxima[2]]
        if head > left and head > right and abs(left - right)/left < 0.1:
            return "Omuz-Baş-Omuz (Head & Shoulders)", 80

    return "Belirsiz / Kararsız", 50

# =========================================
# 🚀 Streamlit Arayüzü
# =========================================
st.title("📈 BIST Pattern Scanner")
st.write("TradingView verileri ile dip ve formasyon analizi")

hisseler = bist_tum_hisseler()
secilen_hisse = st.selectbox("Hisse seçiniz:", sorted(hisseler))

if st.button("Analiz Et"):
    data = Stock_Prices(secilen_hisse)
    if data is not None:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(data['Close'], label="Kapanış")
        ax.plot(data['SMA_20'], label="SMA 20")
        ax.plot(data['SMA_50'], label="SMA 50")

        # Dip noktalarını işaretle
        dips = detect_dips(data['Close'])
        ax.scatter(dips.index, dips.values, color='red', marker='v', label="Dip Noktaları")

        ax.set_title(f"{secilen_hisse} Fiyat ve Dip Noktaları")
        ax.legend()
        st.pyplot(fig)

        # Formasyon tespiti
        pattern, confidence = detect_pattern(data)
        st.subheader("📊 Formasyon Analizi")
        st.write(f"**Formasyon:** {pattern}")
        st.write(f"**Güven:** %{confidence}")
    else:
        st.error("Veri alınamadı.")
