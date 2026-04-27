import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy import stats
from tvDatafeed import TvDatafeed, Interval
from tqdm import tqdm
from scipy.signal import argrelextrema

# TradingView bağlantısı
tv = TvDatafeed()

# =========================================
# 📊 BIST Tüm Hisseler Fonksiyonu
# =========================================
def bist_tum_hisseler():
    print("📊 BIST hisseleri yükleniyor...")
    try:
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
    except Exception as e:
        print(f"❌ Hisse listesi hatası: {e}")
        return []

# =========================================
# 📈 Veri Çekme Fonksiyonu
# =========================================
def Stock_Prices(Hisse):
    try:
        df = tv.get_hist(symbol=Hisse, exchange="BIST", interval=Interval.in_daily, n_bars=200)
        if df is None or len(df) < 50:
            return None
        df.reset_index(inplace=True)
        df.rename(columns={'close':'Close','volume':'Volume','open':'Open','high':'High','low':'Low'}, inplace=True)
        # Basit indikatörler ekleyelim
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['RSI'] = compute_rsi(df['Close'], 14)
        return df
    except Exception as e:
        print(f"{Hisse} için veri hatası: {e}")
        return None

# =========================================
# 📊 RSI Hesaplama
# =========================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================================
# 📊 Trend Kanalı Hesaplama
# =========================================
def Trend_Channel(df):
    best_period = None
    best_r_value = 0
    periods = [50, 100, 150]
    for period in periods:
        close_data = df['Close'].tail(period)
        x = np.arange(len(close_data))
        slope, intercept, r_value, _, _ = stats.linregress(x, close_data)
        if abs(r_value) > abs(best_r_value):
            best_r_value = abs(r_value)
            best_period = period
    return best_period, best_r_value

# =========================================
# 📉 Trend Çizgisi ve Kanal Grafiği
# =========================================
def Plot_Trendlines(Hisse,data,best_period,rval=0.85):
    plt.close()
    close_data = data['Close'].tail(best_period)
    x_best_period = np.arange(len(close_data))
    slope_best_period, intercept_best_period, r_value_best_period, _, _ = stats.linregress(x_best_period, close_data)
    trendline = slope_best_period * x_best_period + intercept_best_period
    upper_channel = trendline + (trendline.std() * 1.1)
    lower_channel = trendline - (trendline.std() * 1.1)

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Kapanış Fiyatı')
    plt.plot(data.index[-best_period:], trendline, 'g-', label=f'Trend Çizgisi (R={r_value_best_period:.2f})')
    plt.fill_between(data.index[-best_period:], upper_channel, trendline, color='lightgreen', alpha=0.3, label='Üst Kanal')
    plt.fill_between(data.index[-best_period:], trendline, lower_channel, color='lightcoral', alpha=0.3, label='Alt Kanal')
    plt.title(str(Hisse)+' Kapanış Fiyatı ve Trend Çizgisi')
    plt.xlabel('Tarih Endeksi')
    plt.ylabel('Kapanış Fiyatı')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    upper_diff = upper_channel - close_data
    lower_diff = close_data - lower_channel
    last_upper_diff = upper_diff.iloc[-1]
    last_lower_diff = lower_diff.iloc[-1]
    
    if abs(r_value_best_period) > rval and (last_upper_diff < 0):
        print(str(Hisse)+' Yukarı kırılım!')
        plt.savefig(f'{Hisse}_Yukarı_Kırılım.png', bbox_inches='tight', dpi=200)

    if abs(r_value_best_period) > rval and (last_lower_diff < 0):
        print(str(Hisse)+' Aşağı kırılım!')
        plt.savefig(f'{Hisse}_Aşağı_Kırılım.png', bbox_inches='tight', dpi=200)
  
    return

# =========================================
# 📌 Formasyon Tespiti (Çanak dahil)
# =========================================
def detect_pattern(df):
    recent = df.tail(100).copy()
    prices, highs, lows, volumes = recent['Close'].values, recent['High'].values, recent['Low'].values, recent['Volume'].values
    rsi_vals = recent['RSI'].values
    atr = recent['ATR'].mean()
    sma20_val = recent['SMA_20'].iloc[-1]
    sma50_val = recent['SMA_50'].iloc[-1]
    price = prices[-1]

    # Çanak formasyonu kontrolü
    left_top = highs[:30].max()
    dip = lows[30:70].min()
    right_top = highs[70:].max()
    is_cup = abs(left_top - right_top) / left_top < 0.08 and dip < (left_top * 0.85)
    is_handle = prices[-15:].mean() < right_top and prices[-15:].min() > dip
    if is_cup and is_handle:
        return "Kulplu Çanak (Cup with Handle)", 90
    elif is_cup:
        return "Çanak Formasyonu (Cup)", 85

    # Basit diğer formasyonlar
    body1 = recent['Close'].iloc[-2] - recent['Open'].iloc[-2]
    body2 = recent['Close'].iloc[-1] - recent['Open'].iloc[-1]
    if (body1 < 0) and (body2 > 0) and (body2 > abs(body1) * 1.2):
        return "Bullish Engulfing", 75
    if (body1 > 0) and (body2 < 0) and (abs(body2) > abs(body1) * 1.2):
        return "Bearish Engulfing", 75

    return "Belirsiz / Kararsız", 50

# =========================================
# 🚀 Ana Çalışma
# =========================================
Hisseler = bist_tum_hisseler()
print(f"Toplam {len(Hisseler)} hisse bulundu.")

for h in tqdm(Hisseler, desc="📈 Hisseler işleniyor", unit="hisse"):
    try:
        data = Stock_Prices(h)
        if data is not None:
            best_period, best_r_value = Trend_Channel(data)
            Plot_Trendlines(h, data, best_period)
            pattern, confidence = detect_pattern(data)
            print(f"{h}: {pattern} (Güven: {confidence}%)")
    except Exception as e:
        print(f"{h} için hata: {e}")
