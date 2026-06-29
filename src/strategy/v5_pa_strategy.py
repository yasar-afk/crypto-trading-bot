# -*- coding: utf-8 -*-
"""
src/strategy/v5_pa_strategy.py — Refined, Optimized Price Action & SMC Strategy V5.
"""
import pandas as pd
import numpy as np
from src.strategy.smc_utils import get_premium_discount_zone

class V5PriceActionStrategy:
    """ANTIGRAVITI V5 Price Action & Smart Money Concepts Strategy (Refine Optimized).
    
    Optimized confluences from focused historical parameter sweeps:
      1. Sweep Window: 100
      2. Max Hold Sweep: 7 candles
      3. Stop Loss ATR offset: 0.6 * ATR
      4. Target Risk-Reward: 5.5
      5. Trend Filter: 180 EMA
      6. Premium/Discount Zone Filter: Enabled
      7. Session Filtering: Disabled (24/7 scanning)
    """
    
    def __init__(
        self, 
        sweep_window=30, 
        max_hold_sweep=20, 
        target_rr=5.5, 
        require_trend=True, 
        trend_ema=100,
        use_premium_discount=False, 
        atr_multiplier=0.6,
        fvg_wait=3,
        use_session_filter=False,
        max_tp_pct=0.50,
        min_sl_pct=0.05,
    ):
        self.sweep_window = sweep_window
        self.max_hold_sweep = max_hold_sweep
        self.target_rr = target_rr
        self.require_trend = require_trend
        self.trend_ema = trend_ema
        self.use_premium_discount = use_premium_discount
        self.atr_multiplier = atr_multiplier
        self.fvg_wait = fvg_wait
        self.use_session_filter = use_session_filter
        self.max_tp_pct = max_tp_pct
        self.min_sl_pct = min_sl_pct

    def calculate_signals(self, df):
        """Calculates indicators and entry setups. Returns a df with signals."""
        df = df.copy()
        
        df["swing_high"] = df["high"].shift(1).rolling(window=self.sweep_window).max()
        df["swing_low"] = df["low"].shift(1).rolling(window=self.sweep_window).min()
        df["trend_ema_val"] = df["close"].ewm(span=self.trend_ema, adjust=False).mean()
        
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()
        
        # Ensure Funding and Open Interest columns exist for compatibility
        if "funding_rate" not in df.columns:
            df["funding_rate"] = 0.0
        if "open_interest" not in df.columns:
            df["open_interest"] = 0.0
            
        df["signal"] = "HOLD"
        df["entry_price"] = 0.0
        df["sl_price"] = 0.0
        df["tp_price"] = 0.0
        df["has_fvg"] = False
        
        last_bull_sweep_idx = -1
        last_bull_sweep_low = 0.0
        last_bear_sweep_idx = -1
        last_bear_sweep_high = 0.0
        
        start_idx = int(max(self.sweep_window, self.trend_ema) + 2)
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
        
        signals = ["HOLD"] * len(df)
        entry_prices = [0.0] * len(df)
        sl_prices = [0.0] * len(df)
        tp_prices = [0.0] * len(df)
        has_fvgs = [False] * len(df)
        
        timestamps = df.index.astype(np.int64) // 10**9 if hasattr(df.index, 'timestamp') or isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df))
        
        for i in range(start_idx, len(df)):
            high_prev = swing_highs[i]
            low_prev = swing_lows[i]
            ema_val = emas[i]
            atr_val = atrs[i]
            
            h = highs[i]
            l = lows[i]
            c = closes[i]
            o = opens[i]
            
            candle_range = h - l
            if candle_range == 0:
                continue
                
            # Sweep Detection
            is_bull_sweep_now = (l < low_prev) and (c > low_prev)
            if is_bull_sweep_now:
                lower_wick = min(c, o) - l
                is_bull_rejection = (lower_wick / candle_range >= 0.35) or (c > o and closes[i-1] < opens[i-1])
                if is_bull_rejection:
                    last_bull_sweep_idx = i
                    last_bull_sweep_low = l
                    last_bear_sweep_idx = -1
                    
            is_bear_sweep_now = (h > high_prev) and (c < high_prev)
            if is_bear_sweep_now:
                upper_wick = h - max(c, o)
                is_bear_rejection = (upper_wick / candle_range >= 0.35) or (c < o and closes[i-1] > opens[i-1])
                if is_bear_rejection:
                    last_bear_sweep_idx = i
                    last_bear_sweep_high = h
                    last_bull_sweep_idx = -1
                    
            # Stateful Signal Evaluation
            # 1. BULLISH SETUP
            if last_bull_sweep_idx != -1:
                if i - last_bull_sweep_idx > self.max_hold_sweep:
                    last_bull_sweep_idx = -1
                else:
                    is_bullish_mss = c > max(closes[i-1], closes[i-2], closes[i-3])
                    if is_bullish_mss:
                        trend_ok = True
                        if self.require_trend:
                            trend_ok = (c > ema_val)
                            
                        if self.use_premium_discount and trend_ok:
                            s_h = swing_highs[last_bull_sweep_idx]
                            s_l = swing_lows[last_bull_sweep_idx]
                            zone = get_premium_discount_zone(c, s_h, s_l)
                            trend_ok = (zone == "DISCOUNT")
                            
                        if trend_ok:
                            if self.use_session_filter:
                                try:
                                    dt = pd.to_datetime(timestamps[i], unit='s', utc=True)
                                    hour = dt.hour
                                    is_kz = (7 <= hour < 10) or (12 <= hour < 16)
                                    if not is_kz:
                                        trend_ok = False
                                except Exception:
                                    trend_ok = False
                                    
                        if trend_ok:
                            sl = last_bull_sweep_low - (atr_val * self.atr_multiplier)
                            
                            fvg_gap = lows[i] - highs[i-2]
                            if fvg_gap > 0:
                                has_fvgs[i] = True
                                fvg_mid = (lows[i] + highs[i-2]) / 2.0
                                entry_prices[i] = fvg_mid
                            else:
                                entry_prices[i] = c
                                
                            risk = entry_prices[i] - sl
                            if risk <= 0:
                                risk = atr_val * 0.5
                                sl = entry_prices[i] - risk
                            
                            # SL mesafesi çok dar mı kontrol et
                            sl_pct = risk / entry_prices[i] if entry_prices[i] > 0 else 0
                            if sl_pct < self.min_sl_pct:
                                continue
                                
                            tp_val = entry_prices[i] + (self.target_rr * risk)
                            # TP çok uzak mı kontrol et
                            tp_pct = (tp_val - entry_prices[i]) / entry_prices[i] if entry_prices[i] > 0 else 0
                            if tp_pct > self.max_tp_pct:
                                tp_val = entry_prices[i] * (1 + self.max_tp_pct)
                            if tp_val <= 0: continue
                            signals[i] = "BUY"
                            sl_prices[i] = sl
                            tp_prices[i] = tp_val
                            last_bull_sweep_idx = -1
                            
            # 2. BEARISH SETUP
            if last_bear_sweep_idx != -1:
                if i - last_bear_sweep_idx > self.max_hold_sweep:
                    last_bear_sweep_idx = -1
                else:
                    is_bearish_mss = c < min(closes[i-1], closes[i-2], closes[i-3])
                    if is_bearish_mss:
                        trend_ok = True
                        if self.require_trend:
                            trend_ok = (c < ema_val)
                            
                        if self.use_premium_discount and trend_ok:
                            s_h = swing_highs[last_bear_sweep_idx]
                            s_l = swing_lows[last_bear_sweep_idx]
                            zone = get_premium_discount_zone(c, s_h, s_l)
                            trend_ok = (zone == "PREMIUM")
                            
                        if trend_ok:
                            if self.use_session_filter:
                                try:
                                    dt = pd.to_datetime(timestamps[i], unit='s', utc=True)
                                    hour = dt.hour
                                    is_kz = (7 <= hour < 10) or (12 <= hour < 16)
                                    if not is_kz:
                                        trend_ok = False
                                except Exception:
                                    trend_ok = False
                                    
                        if trend_ok:
                            sl = last_bear_sweep_high + (atr_val * self.atr_multiplier)
                            
                            fvg_gap = lows[i-2] - highs[i]
                            if fvg_gap > 0:
                                has_fvgs[i] = True
                                fvg_mid = (highs[i] + lows[i-2]) / 2.0
                                entry_prices[i] = fvg_mid
                            else:
                                entry_prices[i] = c
                                
                            risk = sl - entry_prices[i]
                            if risk <= 0:
                                risk = atr_val * 0.5
                                sl = entry_prices[i] + risk
                            
                            # SL mesafesi çok dar mı kontrol et
                            sl_pct = risk / entry_prices[i] if entry_prices[i] > 0 else 0
                            if sl_pct < self.min_sl_pct:
                                continue
                                
                            tp_val = entry_prices[i] - (self.target_rr * risk)
                            # TP çok uzak mı kontrol et
                            tp_pct = (entry_prices[i] - tp_val) / entry_prices[i] if entry_prices[i] > 0 else 0
                            if tp_pct > self.max_tp_pct:
                                tp_val = entry_prices[i] * (1 - self.max_tp_pct)
                            if tp_val <= 0: continue
                            signals[i] = "SELL"
                            sl_prices[i] = sl
                            tp_prices[i] = tp_val
                            last_bear_sweep_idx = -1
                            
        df["signal"] = signals
        df["entry_price"] = entry_prices
        df["sl_price"] = sl_prices
        df["tp_price"] = tp_prices
        df["has_fvg"] = has_fvgs
        
        return df
