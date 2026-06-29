# -*- coding: utf-8 -*-
"""
src/strategy/v7_pa_strategy.py — V7 Price Action Stratejisi
Backtest sonucuna göre optimize edilmiş: 100 coin, %114 ROI, %26.6 win rate.
"""
import pandas as pd
import numpy as np
from src.strategy.smc_utils import get_premium_discount_zone, detect_displacement_candle


class V7PriceActionStrategy:
    """
    ANTIGRAVITI V7 — Backtest Optimized Price Action Stratejisi
    
    V5'ten Farklar:
      - Volatilite filtresi (> %0.3)
      - Hacim filtresi (ortalamanın x0.5'i)
      - Sabit RR hedefi: 5.5
      - Konservatif risk: %1/trade
    """

    def __init__(
        self,
        sweep_window: int = 30,
        max_hold_sweep: int = 20,
        target_rr: float = 5.5,
        trend_ema: int = 100,
        atr_multiplier: float = 0.6,
        use_volume_filter: bool = True,
        volume_threshold: float = 0.3,
        min_volatility_pct: float = 0.3,
        use_premium_discount: bool = False,
        max_tp_pct: float = 0.50,
        min_sl_pct: float = 0.05,
        max_sl_pct: float = 0.08,
    ):
        self.sweep_window = sweep_window
        self.max_hold_sweep = max_hold_sweep
        self.target_rr = target_rr
        self.trend_ema = trend_ema
        self.atr_multiplier = atr_multiplier
        self.use_volume_filter = use_volume_filter
        self.volume_threshold = volume_threshold
        self.min_volatility_pct = min_volatility_pct
        self.use_premium_discount = use_premium_discount
        self.max_tp_pct = max_tp_pct
        self.min_sl_pct = min_sl_pct
        self.max_sl_pct = max_sl_pct

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """V7 sinyallerini hesaplar."""
        df = df.copy()

        # İndikatörler
        df["swing_high"] = df["high"].shift(1).rolling(window=self.sweep_window).max()
        df["swing_low"] = df["low"].shift(1).rolling(window=self.sweep_window).min()
        df["trend_ema_val"] = df["close"].ewm(span=self.trend_ema, adjust=False).mean()

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        # Hacim ortalaması
        df["volume_ma"] = df["volume"].rolling(window=50).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Volatilite
        df["volatility"] = df["close"].pct_change(periods=20).abs() * 100

        # Sinyal kolonları
        df["signal"] = "HOLD"
        df["entry_price"] = 0.0
        df["sl_price"] = 0.0
        df["tp_price"] = 0.0

        start_idx = int(max(self.sweep_window, self.trend_ema) + 5)
        if start_idx >= len(df):
            return df

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        swing_highs = df["swing_high"].values
        swing_lows = df["swing_low"].values
        emas = df["trend_ema_val"].values
        atrs = df["atr"].values
        vol_ratios = df["volume_ratio"].values
        volatilities = df["volatility"].values

        signals = ["HOLD"] * len(df)
        entry_prices = [0.0] * len(df)
        sl_prices = [0.0] * len(df)
        tp_prices = [0.0] * len(df)

        last_bull_idx = -1
        last_bull_low = 0.0
        last_bear_idx = -1
        last_bear_high = 0.0

        for i in range(start_idx, len(df)):
            h, l, c, o = highs[i], lows[i], closes[i], opens[i]
            ema_val = emas[i]
            atr_val = atrs[i]
            vol_ratio = vol_ratios[i] if not pd.isna(vol_ratios[i]) else 1.0
            volatility = volatilities[i] if not pd.isna(volatilities[i]) else 0.0

            if pd.isna(ema_val) or pd.isna(atr_val) or atr_val == 0:
                continue

            candle_range = h - l
            if candle_range == 0:
                continue

            # Volatilite filtresi
            if volatility < self.min_volatility_pct:
                continue

            # Sweep algılama
            is_bull_sweep = (l < swing_lows[i]) and (c > swing_lows[i])
            if is_bull_sweep:
                lower_wick = min(c, o) - l
                if candle_range > 0 and (lower_wick / candle_range >= 0.35 or (c > o and closes[i-1] < opens[i-1])):
                    last_bull_idx = i
                    last_bull_low = l
                    last_bear_idx = -1

            is_bear_sweep = (h > swing_highs[i]) and (c < swing_highs[i])
            if is_bear_sweep:
                upper_wick = h - max(c, o)
                if candle_range > 0 and (upper_wick / candle_range >= 0.35 or (c < o and closes[i-1] > opens[i-1])):
                    last_bear_idx = i
                    last_bear_high = h
                    last_bull_idx = -1

            # BULLISH giriş
            if last_bull_idx != -1 and i - last_bull_idx <= self.max_hold_sweep:
                if c > max(closes[i-1], closes[i-2], closes[i-3]):
                    if c > ema_val:
                        # Hacim filtresi
                        if self.use_volume_filter and vol_ratio < self.volume_threshold:
                            continue

                        # Premium/Discount
                        if self.use_premium_discount:
                            s_h = swing_highs[last_bull_idx]
                            s_l = swing_lows[last_bull_idx]
                            zone = get_premium_discount_zone(c, s_h, s_l)
                            if zone != "DISCOUNT":
                                continue

                        sl = last_bull_low - (atr_val * self.atr_multiplier)
                        risk = c - sl
                        if risk > 0:
                            sl_pct = risk / c
                            if sl_pct < self.min_sl_pct:
                                continue
                            if sl_pct > self.max_sl_pct:
                                sl = c * (1 - self.max_sl_pct)
                                risk = c - sl
                                if risk <= 0:
                                    continue
                            tp = c + (self.target_rr * risk)
                            # TP çok uzak mı kontrol et
                            tp_pct = (tp - c) / c
                            if tp_pct > self.max_tp_pct:
                                tp = c * (1 + self.max_tp_pct)
                            if tp <= 0: continue
                            signals[i] = "BUY"
                            entry_prices[i] = c
                            sl_prices[i] = sl
                            tp_prices[i] = tp
                            last_bull_idx = -1

            # BEARISH giriş
            elif last_bear_idx != -1 and i - last_bear_idx <= self.max_hold_sweep:
                if c < min(closes[i-1], closes[i-2], closes[i-3]):
                    if c < ema_val:
                        if self.use_volume_filter and vol_ratio < self.volume_threshold:
                            continue

                        if self.use_premium_discount:
                            s_h = swing_highs[last_bear_idx]
                            s_l = swing_lows[last_bear_idx]
                            zone = get_premium_discount_zone(c, s_h, s_l)
                            if zone != "PREMIUM":
                                continue

                        sl = last_bear_high + (atr_val * self.atr_multiplier)
                        risk = sl - c
                        if risk > 0:
                            sl_pct = risk / c
                            if sl_pct < self.min_sl_pct:
                                continue
                            if sl_pct > self.max_sl_pct:
                                sl = c * (1 + self.max_sl_pct)
                                risk = sl - c
                                if risk <= 0:
                                    continue
                            tp = c - (self.target_rr * risk)
                            # TP çok uzak mı kontrol et
                            tp_pct = (c - tp) / c
                            if tp_pct > self.max_tp_pct:
                                tp = c * (1 - self.max_tp_pct)
                            if tp <= 0: continue
                            signals[i] = "SELL"
                            entry_prices[i] = c
                            sl_prices[i] = sl
                            tp_prices[i] = tp
                            last_bear_idx = -1

        df["signal"] = signals
        df["entry_price"] = entry_prices
        df["sl_price"] = sl_prices
        df["tp_price"] = tp_prices

        return df
