# ============================================================
# src/bot/scheduler.py — Trading Bot Trading Bot
#
# AMAÇ:
#   Her timeframe için bir sonraki mum kapanış zamanını hesaplar
#   ve ne kadar beklenmesi gerektiğini söyler.
#   BotEngine bu modülü kullanarak gereksiz API çağrısı yapmaz:
#   sadece yeni mum kapandığında veri çeker.
#
# MİMARİ NOT:
#   Timeframe'e göre interval hesaplanır:
#     4h → 4 * 3600 saniye
#     1d → 86400 saniye
#   Her mum tam kapanış anında veri çekilir (±30sn tolerans).
#   Bu yaklaşım API rate-limit tasarrufu sağlar.
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v1.0 | İlk Scheduler implementasyonu
# ============================================================

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Desteklenen timeframe → saniye dönüşüm tablosu
TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
    "1d":  86400,
    "1w":  604800,
}


class CandleScheduler:
    """Her timeframe için mum kapanış zamanını yönetir.

    BotEngine bu sınıfı kullanarak:
      - Bir sonraki mum kapanışına kaç saniye kaldığını öğrenir
      - Gereksiz API çağrısından kaçınır
      - Çok timeframe polling'i koordine eder

    Attributes:
        timeframes: Takip edilen timeframe listesi.
        _last_fetch: Her timeframe için son veri çekme zamanı.

    Example:
        >>> scheduler = CandleScheduler(["4h", "1d"])
        >>> due = scheduler.get_due_timeframes()
        >>> secs = scheduler.seconds_until_next_candle("4h")
    """

    def __init__(self, timeframes: list[str]) -> None:
        """CandleScheduler başlatır.

        Args:
            timeframes: Takip edilecek timeframe listesi (ör. ['4h', '1d']).
        """
        self.timeframes = timeframes
        self._last_fetch: Dict[str, float] = {}   # timeframe → unix timestamp

        for tf in timeframes:
            if tf not in TIMEFRAME_SECONDS:
                logger.warning(
                    f"Bilinmeyen timeframe: '{tf}'. "
                    f"Desteklenenler: {list(TIMEFRAME_SECONDS.keys())}"
                )

        logger.info(
            f"CandleScheduler başlatıldı | "
            f"Timeframes: {timeframes}"
        )

    def is_due(self, timeframe: str) -> bool:
        """Bu timeframe için yeni veri çekme zamanı geldi mi?

        İlk çağrıda her zaman True döner (başlangıç verisi için).
        Sonrasında interval süresi geçtiyse True döner.

        Args:
            timeframe: Kontrol edilecek timeframe.

        Returns:
            True: Veri çekme zamanı geldi.
            False: Henüz bekleme süresi dolmadı.
        """
        interval = TIMEFRAME_SECONDS.get(timeframe, 3600)
        last     = self._last_fetch.get(timeframe, 0.0)
        now      = time.time()

        if now - last >= interval:
            return True
        return False

    def mark_fetched(self, timeframe: str) -> None:
        """Bir timeframe için veri çekme zamanını günceller.

        Args:
            timeframe: Güncellenecek timeframe.
        """
        self._last_fetch[timeframe] = time.time()

    def get_due_timeframes(self) -> list[str]:
        """Şu an veri çekilmesi gereken timeframe'leri döner.

        Returns:
            is_due() == True olan timeframe listesi.
        """
        return [tf for tf in self.timeframes if self.is_due(tf)]

    def seconds_until_next_candle(self, timeframe: str) -> float:
        """Bir sonraki mum kapanışına kaç saniye kaldığını hesaplar.

        Args:
            timeframe: Hesaplanacak timeframe.

        Returns:
            Saniye cinsinden bekleme süresi (0 = hemen çek).
        """
        interval = TIMEFRAME_SECONDS.get(timeframe, 3600)
        last     = self._last_fetch.get(timeframe, 0.0)
        elapsed  = time.time() - last
        remaining = max(0.0, interval - elapsed)
        return remaining

    def next_candle_time(self, timeframe: str) -> Optional[datetime]:
        """Bir sonraki mum kapanış zamanını UTC datetime olarak döner.

        Args:
            timeframe: Sorgulanacak timeframe.

        Returns:
            UTC datetime veya None (bilinmeyen timeframe).
        """
        interval = TIMEFRAME_SECONDS.get(timeframe)
        if interval is None:
            return None
        next_ts = self._last_fetch.get(timeframe, time.time()) + interval
        return datetime.fromtimestamp(next_ts, tz=timezone.utc)

    def status_report(self) -> str:
        """Tüm timeframe'lerin durum özeti."""
        lines = ["Zamanlayıcı Durumu:"]
        for tf in self.timeframes:
            remaining = self.seconds_until_next_candle(tf)
            next_time = self.next_candle_time(tf)
            time_str  = (
                next_time.strftime("%H:%M:%S UTC") if next_time else "belirsiz"
            )
            lines.append(
                f"  {tf:<5} | Sonraki çekim: {time_str} "
                f"({remaining:.0f}sn sonra)"
            )
        return "\n".join(lines)
