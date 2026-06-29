# ============================================================
# src/strategy/regime_detector.py — Piyasa Rejimi Tespiti
#
# AMAÇ:
#   Piyasanın trend mi yoksa range mi olduğunu tespit ederek
#   hangi stratejinin kullanılacağına karar verir.
#
# REJİMLER:
#   TREND_UP   → Trend Following stratejisi
#   TREND_DOWN → Trend Following (kısa pozisyon)
#   RANGE      → Mean Reversion + Grid stratejisi
#   VOLATILE   → Dikkatli ol, pozisyon küçült
# ============================================================

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Piyasa rejimi türleri."""
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"


class RegimeDetector:
    """ADX ve Bollinger Bands ile piyasa rejimi tespiti."""

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_spike_multiplier: float = 2.0,
    ) -> None:
        """RegimeDetector başlatır.

        Args:
            adx_period: ADX periyodu.
            adx_trend_threshold: Trend eşiği (ADX > bu değer → trend).
            adx_range_threshold: Range eşiği (ADX < bu değer → range).
            bb_period: Bollinger Bands periyodu.
            bb_std: BB standart sapması.
            atr_period: ATR periyodu.
            atr_spike_multiplier: ATR sıçrama çarpanı.
        """
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_spike_multiplier = atr_spike_multiplier

    def detect(self, df: pd.DataFrame, idx: int) -> MarketRegime:
        """Piyasa rejimini tespit eder.

        Args:
            df: İndikatörlü DataFrame.
            idx: Mevcut bardaki indeks.

        Returns:
            MarketRegime enum değeri.
        """
        if idx < max(self.adx_period, self.bb_period, self.atr_period):
            return MarketRegime.RANGE

        row = df.iloc[idx]

        # ADX ile trend gücü
        adx = float(row.get("adx", 20))
        plus_di = float(row.get("plus_di", 0))
        minus_di = float(row.get("minus_di", 0))

        # Bollinger Bands genişliği
        bb_width = float(row.get("bb_width", 0))
        bb_width_ma = float(row.get("bb_width_ma", bb_width))
        is_squeeze = bb_width < bb_width_ma * 0.8 if bb_width_ma > 0 else False

        # ATR volatilite
        atr = float(row.get("atr", 0))
        atr_ma = float(row.get("atr_ma", atr))
        is_volatile = atr > atr_ma * self.atr_spike_multiplier if atr_ma > 0 else False

        # Rejim belirleme
        if is_volatile:
            return MarketRegime.VOLATILE

        if adx >= self.adx_trend_threshold:
            if plus_di > minus_di:
                return MarketRegime.TREND_UP
            else:
                return MarketRegime.TREND_DOWN

        if adx <= self.adx_range_threshold:
            return MarketRegime.RANGE

        # Orta zone — mevcut trend'e göre
        if plus_di > minus_di:
            return MarketRegime.TREND_UP
        else:
            return MarketRegime.TREND_DOWN

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """İndikatörleri hesaplar.

        Args:
            df: Ham OHLCV DataFrame.

        Returns:
            İndikatörler eklenmiş DataFrame.
        """
        df = df.copy()

        # ADX hesaplama
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = tr.rolling(window=self.atr_period).mean()
        smooth_plus = plus_dm.rolling(window=self.adx_period).mean()
        smooth_minus = minus_dm.rolling(window=self.adx_period).mean()

        plus_di = 100 * smooth_plus / atr
        minus_di = 100 * smooth_minus / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df["adx"] = dx.rolling(window=self.adx_period).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_mid"] + self.bb_std * bb_std
        df["bb_lower"] = df["bb_mid"] - self.bb_std * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_width_ma"] = df["bb_width"].rolling(window=self.bb_period).mean()

        # ATR
        df["atr"] = tr.rolling(window=self.atr_period).mean()
        df["atr_ma"] = df["atr"].rolling(window=self.atr_period).mean()

        return df

    def get_strategy_for_regime(self, regime: MarketRegime) -> str:
        """Rejime göre strateji adını döndürür.

        Args:
            regime: Piyasa rejimi.

        Returns:
            Strateji adı.
        """
        mapping = {
            MarketRegime.TREND_UP: "trend_following",
            MarketRegime.TREND_DOWN: "trend_following",
            MarketRegime.RANGE: "mean_reversion",
            MarketRegime.VOLATILE: "mean_reversion",
        }
        return mapping.get(regime, "mean_reversion")
