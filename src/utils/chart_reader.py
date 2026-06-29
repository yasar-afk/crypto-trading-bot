# ============================================================
# src/utils/chart_reader.py — Grafik Okuma Modülü
#
# AMAÇ:
#   Otomatik grafik analizi yapar, support/resistance
#   seviyeleri tespit eder ve piyasa durumu hakkında
#   yorum üretir.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SupportResistance:
    """Support/Resistance seviyesi."""
    price: float
    strength: int  # Kaç kez test edildi
    level_type: str  # "support" veya "resistance"
    last_touch_idx: int


@dataclass
class ChartAnalysis:
    """Grafik analiz sonucu."""
    trend_direction: str  # "UP", "DOWN", "SIDEWAYS"
    trend_strength: str  # "STRONG", "MODERATE", "WEAK"
    support_levels: List[SupportResistance]
    resistance_levels: List[SupportResistance]
    key_levels: List[float]
    pattern_detected: Optional[str]
    volume_analysis: str
    summary: str


class ChartReader:
    """Otomatik grafik analizi yapan sınıf."""

    def __init__(
        self,
        lookback: int = 100,
        swing_window: int = 5,
        level_tolerance: float = 0.02,
        min_level_touches: int = 2,
    ) -> None:
        """ChartReader başlatır.

        Args:
            lookback: Geriye dönük bakış penceresi.
            swing_window: Swing high/low tespit penceresi.
            level_tolerance: Seviye eşleşme toleransı (%2).
            min_level_touches: Minimum seviye dokunma sayısı.
        """
        self.lookback = lookback
        self.swing_window = swing_window
        self.level_tolerance = level_tolerance
        self.min_level_touches = min_level_touches

    def analyze(self, df: pd.DataFrame) -> ChartAnalysis:
        """Grafik analizi yapar.

        Args:
            df: OHLCV DataFrame.

        Returns:
            ChartAnalysis nesnesi.
        """
        if len(df) < self.lookback:
            df = df.tail(max(self.lookback, 50))

        # Trend tespiti
        trend_dir, trend_str = self._detect_trend(df)

        # Support/Resistance
        sr_levels = self._find_support_resistance(df)

        # Desen tespiti
        pattern = self._detect_pattern(df)

        # Hacim analizi
        vol_analysis = self._analyze_volume(df)

        # Özet rapor
        summary = self._generate_summary(
            trend_dir, trend_str, sr_levels, pattern, vol_analysis
        )

        all_levels = [s.price for s in sr_levels]
        supports = [s for s in sr_levels if s.level_type == "support"]
        resistances = [s for s in sr_levels if s.level_type == "resistance"]

        return ChartAnalysis(
            trend_direction=trend_dir,
            trend_strength=trend_str,
            support_levels=supports,
            resistance_levels=resistances,
            key_levels=all_levels,
            pattern_detected=pattern,
            volume_analysis=vol_analysis,
            summary=summary,
        )

    def _detect_trend(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Trend yönünü ve gücünü tespit eder.

        Args:
            df: DataFrame.

        Returns:
            (yön, güç) tuple'ı.
        """
        close = df["close"].values
        ema20 = pd.Series(close).ewm(span=20).mean().values
        ema50 = pd.Series(close).ewm(span=50).mean().values

        current_price = close[-1]
        ema20_now = ema20[-1]
        ema50_now = ema50[-1]

        # ADX benzeri güç ölçümü
        highs = df["high"].values
        lows = df["low"].values
        avg_range = np.mean(highs[-20:] - lows[-20:])
        price_range = highs[-20:].max() - lows[-20:].min()
        strength_ratio = price_range / avg_range if avg_range > 0 else 1

        # Trend yönü
        if current_price > ema20_now > ema50_now:
            direction = "UP"
        elif current_price < ema20_now < ema50_now:
            direction = "DOWN"
        else:
            direction = "SIDEWAYS"

        # Trend gücü
        if strength_ratio > 3:
            strength = "STRONG"
        elif strength_ratio > 1.5:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return direction, strength

    def _find_support_resistance(
        self,
        df: pd.DataFrame,
    ) -> List[SupportResistance]:
        """Support ve resistance seviyelerini bulur.

        Args:
            df: DataFrame.

        Returns:
            SupportResistance listesi.
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Swing high/low tespiti
        swing_highs = []
        swing_lows = []

        for i in range(self.swing_window, len(closes) - self.swing_window):
            if highs[i] == max(highs[i - self.swing_window:i + self.swing_window + 1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i - self.swing_window:i + self.swing_window + 1]):
                swing_lows.append((i, lows[i]))

        # Seviyeleri birleştir ve güçlerini say
        levels = {}
        tolerance = self.level_tolerance

        for idx, price in swing_highs:
            found = False
            for key in levels:
                if abs(key - price) / price < tolerance:
                    levels[key]["count"] += 1
                    levels[key]["last_idx"] = idx
                    found = True
                    break
            if not found:
                levels[price] = {"count": 1, "type": "resistance", "last_idx": idx}

        for idx, price in swing_lows:
            found = False
            for key in levels:
                if abs(key - price) / price < tolerance:
                    levels[key]["count"] += 1
                    levels[key]["last_idx"] = idx
                    found = True
                    break
            if not found:
                levels[price] = {"count": 1, "type": "support", "last_idx": idx}

        # Güçlü seviyeleri filtrele
        current_price = closes[-1]
        result = []
        for price, info in levels.items():
            if info["count"] >= self.min_level_touches:
                # Fiyata göre support/resistance belirle
                if price < current_price:
                    level_type = "support"
                else:
                    level_type = "resistance"

                result.append(SupportResistance(
                    price=price,
                    strength=info["count"],
                    level_type=level_type,
                    last_touch_idx=info["last_idx"],
                ))

        # Fiyata göre sırala
        result.sort(key=lambda x: abs(x.price - current_price))
        return result[:10]

    def _detect_pattern(self, df: pd.DataFrame) -> Optional[str]:
        """Candlestick deseni tespit eder.

        Args:
            df: DataFrame.

        Returns:
            Desen adı veya None.
        """
        if len(df) < 3:
            return None

        last3 = df.tail(3)
        o1, h1, l1, c1 = float(last3.iloc[0]["open"]), float(last3.iloc[0]["high"]), float(last3.iloc[0]["low"]), float(last3.iloc[0]["close"])
        o2, h2, l2, c2 = float(last3.iloc[1]["open"]), float(last3.iloc[1]["high"]), float(last3.iloc[1]["low"]), float(last3.iloc[1]["close"])
        o3, h3, l3, c3 = float(last3.iloc[2]["open"]), float(last3.iloc[2]["high"]), float(last3.iloc[2]["low"]), float(last3.iloc[2]["close"])

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        range2 = h2 - l2

        # Hammer (Çekiç)
        if body2 < range2 * 0.3 and l2 < min(o2, c2) - body2 * 2:
            return "HAMMER"

        # Shooting Star
        if body2 < range2 * 0.3 and h2 > max(o2, c2) + body2 * 2:
            return "SHOOTING_STAR"

        # Engulfing
        if c1 > o1 and c2 < o2 and o2 > c2 and c2 > o1 and o2 < c1:
            return "BULLISH_ENGULFING"
        if c1 < o1 and c2 > o2 and o2 < c2 and c2 < o1 and o2 > c1:
            return "BEARISH_ENGULFING"

        # Doji
        if body2 < range2 * 0.1:
            return "DOJI"

        return None

    def _analyze_volume(self, df: pd.DataFrame) -> str:
        """Hacim analizi yapar.

        Args:
            df: DataFrame.

        Returns:
            Hacim analizi açıklaması.
        """
        if len(df) < 20:
            return "Yeterli hacim verisi yok"

        vol = df["volume"].values
        vol_ma = np.mean(vol[-20:])
        current_vol = vol[-1]
        vol_ratio = current_vol / vol_ma if vol_ma > 0 else 1

        # Son mum yönü
        last_close = float(df.iloc[-1]["close"])
        last_open = float(df.iloc[-1]["open"])
        is_bullish = last_close > last_open

        if vol_ratio > 2:
            direction = "yükseliş" if is_bullish else "düşüş"
            return f"ÇOK YÜKSEK Hacim ({vol_ratio:.1f}x ortalama) → Güçlü {direction} baskısı"
        elif vol_ratio > 1.5:
            direction = "yükseliş" if is_bullish else "düşüş"
            return f"Yüksek Hacim ({vol_ratio:.1f}x ortalama) → {direction.title()} devamı muhtemel"
        elif vol_ratio < 0.5:
            return f"Düşük Hacim ({vol_ratio:.1f}x ortalama) → Dikkatli ol, sahte sinyaller mümkün"
        else:
            return f"Normal Hacim ({vol_ratio:.1f}x ortalama)"

    def _generate_summary(
        self,
        trend_dir: str,
        trend_str: str,
        levels: List[SupportResistance],
        pattern: Optional[str],
        vol_analysis: str,
    ) -> str:
        """Grafik analiz özeti üretir.

        Args:
            trend_dir: Trend yönü.
            trend_str: Trend gücü.
            levels: S/R seviyeleri.
            pattern: Tespit edilen desen.
            vol_analysis: Hacim analizi.

        Returns:
            Türkçe özet rapor.
        """
        trend_tr = {"UP": "Yükseliş", "DOWN": "Düşüş", "SIDEWAYS": "Yatay"}
        strength_tr = {"STRONG": "Güçlü", "MODERATE": "Orta", "WEAK": "Zayıf"}

        parts = []
        parts.append(f"Trend: {strength_tr.get(trend_str, trend_str)} {trend_tr.get(trend_dir, trend_dir)}")

        supports = [l for l in levels if l.level_type == "support"]
        resistances = [l for l in levels if l.level_type == "resistance"]

        if supports:
            parts.append(f"En yakın destek: ${supports[0].price:,.2f} ({supports[0].strength} test)")
        if resistances:
            parts.append(f"En yakın direnç: ${resistances[0].price:,.2f} ({resistances[0].strength} test)")

        if pattern:
            pattern_tr = {
                "HAMMER": "Çekiç (potansiyel yükseliş)",
                "SHOOTING_STAR": "Yıldız Kayması (potansiyel düşüş)",
                "BULLISH_ENGULFING": "Yükseliş Yutması (güçlü yükseliş sinyali)",
                "BEARISH_ENGULFING": "Düşüş Yutması (güçlü düşüş sinyali)",
                "DOJI": "Doji (kararsızlık)",
            }
            parts.append(f"Desen: {pattern_tr.get(pattern, pattern)}")

        parts.append(vol_analysis)

        return " | ".join(parts)
