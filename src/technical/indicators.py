# ============================================================
# src/technical/indicators.py — İndikatör Modelleri & Yeni İndikatörler
#
# AMAÇ:
#   TechnicalEngine tarafından üretilen indikatör yorumlarının
#   veri modellerini (dataclass) ve enum'larını tanımlar.
#   Ayrıca VWAP, Ichimoku, OBV, Stochastic RSI, Volume Profile
#   gibi ek indikatörleri hesaplar.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Enum'lar ──────────────────────────────────────────────

class SignalDirection(Enum):
    """Sinyal yönü."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

    @property
    def score(self) -> float:
        """Skor karşılığı."""
        scores = {
            "STRONG_BUY": 1.00,
            "BUY": 0.75,
            "NEUTRAL": 0.50,
            "SELL": 0.25,
            "STRONG_SELL": 0.00,
        }
        return scores[self.value]


class RSIZone(Enum):
    """RSI bölge tanımları."""
    OVERSOLD = "OVERSOLD"
    NEAR_OVERSOLD = "NEAR_OVERSOLD"
    NEUTRAL = "NEUTRAL"
    NEAR_OVERBOUGHT = "NEAR_OVERBOUGHT"
    OVERBOUGHT = "OVERBOUGHT"


class MACDCrossType(Enum):
    """MACD kesişim türleri."""
    BULLISH_CROSS = "BULLISH_CROSS"
    BEARISH_CROSS = "BEARISH_CROSS"
    NO_CROSS = "NO_CROSS"


class BBPosition(Enum):
    """Bollinger Bands pozisyonu."""
    ABOVE_UPPER = "ABOVE_UPPER"
    NEAR_UPPER = "NEAR_UPPER"
    MIDDLE = "MIDDLE"
    NEAR_LOWER = "NEAR_LOWER"
    BELOW_LOWER = "BELOW_LOWER"


class EMAAlignment(Enum):
    """EMA hizalama durumu."""
    FULL_BULLISH = "FULL_BULLISH"
    PARTIAL_BULL = "PARTIAL_BULL"
    NEUTRAL = "NEUTRAL"
    PARTIAL_BEAR = "PARTIAL_BEAR"
    FULL_BEARISH = "FULL_BEARISH"


# ─── İndikatör Dataclass'ları ──────────────────────────────

@dataclass(frozen=True)
class RSIResult:
    """RSI analiz sonucu."""
    value: float
    zone: RSIZone
    signal: SignalDirection
    signal_strength: float
    is_bullish_divergence: bool = False
    is_bearish_divergence: bool = False
    divergence_note: str = ""

    @property
    def has_divergence(self) -> bool:
        return self.is_bullish_divergence or self.is_bearish_divergence

    @property
    def effective_weight(self) -> float:
        return 0.85 if self.has_divergence else 0.70


@dataclass(frozen=True)
class MACDResult:
    """MACD analiz sonucu."""
    macd_line: float
    signal_line: float
    histogram: float
    cross_type: MACDCrossType
    signal: SignalDirection
    signal_strength: float
    histogram_trend: bool


@dataclass(frozen=True)
class EMAResult:
    """EMA analiz sonucu."""
    ema20: float
    ema50: float
    ema200: float
    current_price: float
    alignment: EMAAlignment
    signal: SignalDirection
    signal_strength: float
    golden_cross: bool
    death_cross: bool

    @property
    def price_vs_ema200(self) -> str:
        if self.current_price > self.ema200:
            return "Fiyat EMA200 üstünde"
        elif self.current_price < self.ema200:
            return "Fiyat EMA200 altında"
        return "Fiyat EMA200 seviyesinde"


@dataclass(frozen=True)
class ATRResult:
    """ATR analiz sonucu."""
    value: float
    current_price: float
    atr_pct: float
    stop_loss_long: float
    stop_loss_short: float
    take_profit_long: float
    take_profit_short: float
    volatility_label: str


@dataclass(frozen=True)
class BollingerResult:
    """Bollinger Bands analiz sonucu."""
    upper: float
    middle: float
    lower: float
    current_price: float
    bandwidth: float
    percent_b: float
    position: BBPosition
    is_squeeze: bool
    signal: SignalDirection
    signal_strength: float


@dataclass(frozen=True)
class VolumeResult:
    """Hacim analiz sonucu."""
    current_volume: float
    avg_volume: float
    volume_ratio: float
    is_above_average: bool
    signal: SignalDirection
    signal_strength: float


@dataclass(frozen=True)
class ADXResult:
    """ADX analiz sonucu."""
    value: float
    di_plus: float
    di_minus: float
    signal: SignalDirection
    signal_strength: float


@dataclass(frozen=True)
class FibonacciResult:
    """Fibonacci analiz sonucu."""
    swing_high: float
    swing_low: float
    fib_236: float
    fib_382: float
    fib_500: float
    fib_618: float
    fib_786: float


@dataclass(frozen=True)
class PatternResult:
    """Candlestick Pattern analiz sonucu."""
    hammer: bool
    shooting_star: bool
    bullish_engulfing: bool
    bearish_engulfing: bool
    double_bottom: bool
    double_top: bool
    active_patterns: List[str]


# ─── IndicatorSet ──────────────────────────────────────────

@dataclass
class IndicatorSet:
    """Tüm indikatör yorumlarının bir arada tutulduğu kapsamlı set."""
    symbol: str
    timeframe: str
    timestamp: int
    current_price: float
    rsi: Optional[RSIResult] = None
    macd: Optional[MACDResult] = None
    ema: Optional[EMAResult] = None
    atr: Optional[ATRResult] = None
    bollinger: Optional[BollingerResult] = None
    volume: Optional[VolumeResult] = None
    adx: Optional[ADXResult] = None
    fib: Optional[FibonacciResult] = None
    patterns: Optional[PatternResult] = None
    pa_signal: str = "HOLD"
    pa_entry_price: float = 0.0
    pa_sl_price: float = 0.0
    pa_tp_price: float = 0.0
    pa_has_fvg: bool = False

    weighted_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def calculate_weighted_score(self) -> float:
        """Ağırlıklı skor hesaplar."""
        from src.strategy.feature_weights import SCORING_CONFIG
        total = 0.0
        breakdown = {}

        if self.rsi:
            w = self.rsi.effective_weight
            total += self.rsi.signal.score * w
            breakdown["rsi"] = self.rsi.signal.score * w

        if self.macd:
            w = SCORING_CONFIG.get("macd", {}).get("weight", 0.60) if "macd" in SCORING_CONFIG else 0.60
            total += self.macd.signal.score * w
            breakdown["macd"] = self.macd.signal.score * w

        if self.ema:
            w = SCORING_CONFIG.get("ema", {}).get("weight", 0.55) if "ema" in SCORING_CONFIG else 0.55
            total += self.ema.signal.score * w
            breakdown["ema"] = self.ema.signal.score * w

        if self.atr:
            w = SCORING_CONFIG.get("atr", {}).get("weight", 0.65) if "atr" in SCORING_CONFIG else 0.65
            total += 0.5 * w
            breakdown["atr"] = 0.5 * w

        if self.bollinger:
            w = SCORING_CONFIG.get("bollinger_bands", {}).get("weight", 0.50) if "bollinger_bands" in SCORING_CONFIG else 0.50
            total += self.bollinger.signal.score * w
            breakdown["bollinger"] = self.bollinger.signal.score * w

        if self.volume:
            w = SCORING_CONFIG.get("volume", {}).get("weight", 0.85) if "volume" in SCORING_CONFIG else 0.85
            total += self.volume.signal.score * w
            breakdown["volume"] = self.volume.signal.score * w

        if self.adx:
            w = SCORING_CONFIG.get("adx", {}).get("weight", 0.65) if "adx" in SCORING_CONFIG else 0.65
            total += self.adx.signal.score * w
            breakdown["adx"] = self.adx.signal.score * w

        self.weighted_score = total
        self.score_breakdown = breakdown
        return total

    def summary(self) -> str:
        """İnsan okunabilir özet."""
        parts = [f"[{self.symbol}@{self.timeframe}] Fiyat: ${self.current_price:,.4f}"]
        if self.rsi:
            parts.append(f"RSI: {self.rsi.value:.1f} ({self.rsi.zone.value})")
        if self.macd:
            parts.append(f"MACD: {self.macd.cross_type.value}")
        if self.ema:
            parts.append(f"EMA: {self.ema.alignment.value}")
        if self.atr:
            parts.append(f"ATR: {self.atr.value:.4f} ({self.atr.volatility_label})")
        if self.volume:
            parts.append(f"Vol: {self.volume.volume_ratio:.2f}x")
        if self.adx:
            parts.append(f"ADX: {self.adx.value:.1f}")
        parts.append(f"Skor: {self.weighted_score:.3f}")
        return " | ".join(parts)


# ─── Yeni İndikatör Fonksiyonları (V6) ─────────────────────

def calculate_vwap(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """Volume Weighted Average Price hesaplar."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume_tp = typical_price * df["volume"]
    vwap = volume_tp.rolling(window=period).sum() / df["volume"].rolling(window=period).sum()
    return vwap


def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> pd.DataFrame:
    """Ichimoku Cloud hesaplar."""
    df = df.copy()
    high_tenkan = df["high"].rolling(window=tenkan_period).max()
    low_tenkan = df["low"].rolling(window=tenkan_period).min()
    df["tenkan_sen"] = (high_tenkan + low_tenkan) / 2

    high_kijun = df["high"].rolling(window=kijun_period).max()
    low_kijun = df["low"].rolling(window=kijun_period).min()
    df["kijun_sen"] = (high_kijun + low_kijun) / 2

    df["senkou_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(kijun_period)

    high_senkou = df["high"].rolling(window=senkou_b_period).max()
    low_senkou = df["low"].rolling(window=senkou_b_period).min()
    df["senkou_b"] = ((high_senkou + low_senkou) / 2).shift(kijun_period)

    df["chikou_span"] = df["close"].shift(-kijun_period)
    df["cloud_green"] = df["senkou_a"] > df["senkou_b"]

    return df


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume hesaplar."""
    obv = pd.Series(0, index=df.index, dtype=float)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    return obv


def calculate_stochastic_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> pd.DataFrame:
    """Stochastic RSI hesaplar."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)

    df["stoch_rsi_k"] = stoch_rsi.rolling(window=k_smooth).mean() * 100
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(window=d_smooth).mean()

    return df


def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
) -> dict:
    """Volume Profile hesaplar."""
    price_min = float(df["low"].min())
    price_max = float(df["high"].max())

    if price_max == price_min:
        return {"poc": price_min, "vah": price_max, "val": price_min, "bins": {}}

    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    volume_at_price = np.zeros(num_bins)

    for _, row in df.iterrows():
        low, high, vol = float(row["low"]), float(row["high"]), float(row["volume"])
        for j in range(num_bins):
            if bin_edges[j] <= low and high <= bin_edges[j + 1]:
                volume_at_price[j] += vol
            elif low < bin_edges[j + 1] and high > bin_edges[j]:
                overlap = min(high, bin_edges[j + 1]) - max(low, bin_edges[j])
                range_size = high - low if high > low else 1
                volume_at_price[j] += vol * (overlap / range_size)

    poc_idx = np.argmax(volume_at_price)
    poc = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

    total_vol = volume_at_price.sum()
    if total_vol > 0:
        cum_vol = np.cumsum(volume_at_price)
        va_low_idx = np.searchsorted(cum_vol, total_vol * 0.16)
        va_high_idx = np.searchsorted(cum_vol, total_vol * 0.84)
        val = float(bin_edges[min(va_low_idx, num_bins - 1)])
        vah = float(bin_edges[min(va_high_idx, num_bins - 1)])
    else:
        val, vah = price_min, price_max

    bins = {}
    for j in range(num_bins):
        level = float((bin_edges[j] + bin_edges[j + 1]) / 2)
        bins[level] = float(volume_at_price[j])

    return {"poc": poc, "vah": vah, "val": val, "bins": bins}


def enrich_with_new_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame'e tüm yeni indikatörleri ekler."""
    df = df.copy()
    df["vwap"] = calculate_vwap(df)
    df = calculate_ichimoku(df)
    df["obv"] = calculate_obv(df)
    df["obv_ma"] = df["obv"].rolling(window=20).mean()
    df = calculate_stochastic_rsi(df)
    return df
