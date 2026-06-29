# -*- coding: utf-8 -*-
"""
src/strategy/smc_utils.py — Smart Money Concepts (SMC) Yardımcı Fonksiyonları.
"""
import pandas as pd
import numpy as np
import datetime

def detect_displacement_candle(df, idx, atr_col="atr", multiplier=1.5, min_body_ratio=0.55):
    """
    Belirli bir indeksteki mumun 'displacement' (hacimli/gözle görülür hareket) mumu olup olmadığını kontrol eder.
    
    Koşullar:
    1. Mum gövdesi (body) > multiplier * ATR
    2. Mum gövdesinin toplam mum aralığına oranı (body / (high - low)) >= min_body_ratio
    """
    if idx < 0 or idx >= len(df):
        return False, None
        
    h = df["high"].iloc[idx]
    l = df["low"].iloc[idx]
    o = df["open"].iloc[idx]
    c = df["close"].iloc[idx]
    
    # ATR değerini al (varsayılan yoksa hesaplanmalıdır ama df içinde olması beklenir)
    atr_val = df[atr_col].iloc[idx] if atr_col in df.columns else (h - l)
    
    candle_range = h - l
    if candle_range == 0:
        return False, None
        
    body = abs(c - o)
    body_ratio = body / candle_range
    
    is_large = body > (multiplier * atr_val)
    is_solid = body_ratio >= min_body_ratio
    
    if is_large and is_solid:
        direction = "BULL" if c > o else "BEAR"
        return True, direction
    return False, None

def get_premium_discount_zone(price, swing_high, swing_low):
    """
    Fiyatın mevcut swing aralığı içindeki Premium/Discount durumunu belirler.
    SMC Felsefesi:
    - Fiyat %50 (Equilibrium) seviyesinin altındaysa: DISCOUNT (Long için ucuz/uygun)
    - Fiyat %50 (Equilibrium) seviyesinin üstündeyse: PREMIUM (Short için pahalı/uygun)
    """
    if swing_high == swing_low:
        return "EQUILIBRIUM"
    equilibrium = (swing_high + swing_low) / 2.0
    if price < equilibrium:
        return "DISCOUNT"
    elif price > equilibrium:
        return "PREMIUM"
    return "EQUILIBRIUM"

def detect_order_block(df, sweep_idx, mss_idx, direction):
    """
    Sweep ile Yapı Kırılması (MSS) arasındaki bölgede en son oluşan karşıt yönlü mumu (Order Block) tespit eder.
    
    - BULLISH Setup için: MSS öncesi son ayı mumu (close < open)
    - BEARISH Setup için: MSS öncesi son boğa mumu (close > open)
    
    Döner:
        (ob_top, ob_bottom, ob_idx) veya (None, None, None)
    """
    if sweep_idx >= mss_idx or sweep_idx < 0:
        return None, None, None
        
    # Tersten arama yapıyoruz (MSS'e en yakın olandan sweep'e doğru)
    for idx in range(mss_idx - 1, sweep_idx - 1, -1):
        o = df["open"].iloc[idx]
        c = df["close"].iloc[idx]
        h = df["high"].iloc[idx]
        l = df["low"].iloc[idx]
        
        if direction == "BULLISH":
            # Ayı mumu arıyoruz
            if c < o:
                return h, l, idx
        elif direction == "BEARISH":
            # Boğa mumu arıyoruz
            if c > o:
                return h, l, idx
                
    # Bulunamazsa sweep mumunun kendisini veya hemen öncesini alalım
    o_sweep = df["open"].iloc[sweep_idx]
    c_sweep = df["close"].iloc[sweep_idx]
    h_sweep = df["high"].iloc[sweep_idx]
    l_sweep = df["low"].iloc[sweep_idx]
    return h_sweep, l_sweep, sweep_idx

def is_kill_zone(timestamp):
    """
    İşlemin yapıldığı saat diliminin London veya New York Kill Zone olup olmadığını belirler (Crypto için hacim teyidi).
    timestamp: Saniye veya Milisaniye cinsinden epoch.
    
    London KZ: 07:00 - 10:00 UTC
    NY KZ: 12:00 - 15:00 UTC
    """
    if not timestamp or np.isnan(timestamp):
        return False
        
    # CCXT genellikle milisaniye döndürür, saniyeye çevir
    if timestamp > 1e11:
        timestamp = timestamp / 1000.0
        
    try:
        dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        hour = dt.hour
        # London (07:00 - 10:00 UTC) veya NY (12:00 - 15:00 UTC)
        return (7 <= hour < 10) or (12 <= hour < 15)
    except Exception:
        return False
