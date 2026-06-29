# ============================================================
# src/data/validator.py — Trading Bot Trading Bot
# Amaç : DataCollector'dan gelen ham OHLCV verisini sisteme
#         girmeden önce kontrol eder. "Garbage in, garbage out"
#         prensibini engelleyen kalite kapısıdır.
# Tarih: 2026-06-03
#
# KONTROL EDİLEN ANOMALILER:
#   1. Eksik/None değerler (open, high, low, close, volume)
#   2. Mantıksal tutarsızlık (high < low, close < 0 vb.)
#   3. Fiyat sıçraması anomalisi (%10+ ani değişim)
#   4. Tekrarlayan zaman damgaları
#   5. Kronolojik sıralama bozukluğu
#   6. Sıfır veya negatif hacim
#
# GENİŞLETME:
#   Yeni doğrulama kuralı eklemek için _check_* metodunu yaz ve
#   validate() içinde çağır. Her kural bağımsız, izole edilmiştir.
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Doğrulama Sonuç Nesnesi ─────────────────────────────────

@dataclass
class ValidationResult:
    """Doğrulama işleminin sonucunu tutar.

    Attributes:
        is_valid: Veri kullanılabilir mi?
        warnings: Kritik olmayan uyarılar listesi.
        errors: Kritik hatalar listesi (is_valid=False nedeni).
        original_count: Gelen toplam mum sayısı.
        cleaned_count: Temizleme sonrası kalan mum sayısı.
    """

    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    original_count: int = 0
    cleaned_count: int = 0

    def add_error(self, msg: str) -> None:
        """Kritik hata ekler ve is_valid'i False yapar."""
        self.errors.append(msg)
        self.is_valid = False
        logger.error(f"[Validator] HATA: {msg}")

    def add_warning(self, msg: str) -> None:
        """Kritik olmayan uyarı ekler."""
        self.warnings.append(msg)
        logger.warning(f"[Validator] UYARI: {msg}")

    @property
    def summary(self) -> str:
        """Kısa özet string'i döner."""
        status = "✅ GEÇERLİ" if self.is_valid else "❌ GEÇERSİZ"
        return (
            f"{status} | Orijinal: {self.original_count} mum | "
            f"Temiz: {self.cleaned_count} mum | "
            f"Hata: {len(self.errors)} | Uyarı: {len(self.warnings)}"
        )


# ─── Ana Validator Sınıfı ────────────────────────────────────

class OHLCVValidator:
    """OHLCV veri dizilerini doğrular ve temizler.

    Bu sınıf DataCollector tarafından veri sisteme girmeden önce
    çağrılır. Hem doğrulama hem de mümkünse otomatik temizleme yapar.

    Attributes:
        max_missing_candles: İzin verilen maksimum eksik mum sayısı.
        min_volume_threshold: Minimum işlem hacmi eşiği.
        price_spike_factor: Anomali sayılan fiyat değişim oranı.
    """

    def __init__(
        self,
        max_missing_candles: int = 5,
        min_volume_threshold: float = 0.0,
        price_spike_factor: float = 0.10,
    ) -> None:
        """OHLCVValidator başlatır.

        Args:
            max_missing_candles: Bu sayıdan fazla eksik mum varsa hata.
            min_volume_threshold: Bu değerin altındaki mumlar uyarı üretir.
            price_spike_factor: Bu oran (%lik) üzerinde değişim anomalidir.
        """
        self.max_missing_candles = max_missing_candles
        self.min_volume_threshold = min_volume_threshold
        self.price_spike_factor = price_spike_factor

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "",
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """Ana doğrulama metodu — tüm kontrolleri sırayla çalıştırır.

        Args:
            df: Ham OHLCV DataFrame (ccxt çıktısı formatında).
                Beklenen sütunlar: timestamp, open, high, low, close, volume
            symbol: İşlem çifti (loglama için).
            timeframe: Zaman dilimi (loglama için).

        Returns:
            Tuple[temizlenmiş DataFrame, ValidationResult nesnesi]
        """
        tag = f"[{symbol}@{timeframe}]"
        result = ValidationResult(original_count=len(df))
        logger.info(f"{tag} Doğrulama başladı: {len(df)} mum")

        # Boş DataFrame kontrolü
        if df.empty:
            result.add_error(f"{tag} DataFrame tamamen boş.")
            return df, result

        # Sütun kontrolü
        df, result = self._check_columns(df, result, tag)
        if not result.is_valid:
            return df, result

        # Tip dönüşümü
        df = self._convert_types(df)

        # Sıralanmış zaman damgası
        df, result = self._check_timestamps(df, result, tag)

        # Temel değer doğrulama
        df, result = self._check_ohlc_values(df, result, tag)

        # Fiyat sıçraması anomalisi
        result = self._check_price_spikes(df, result, tag)

        # Hacim kontrolü
        result = self._check_volume(df, result, tag)

        # Son mum sayısı
        result.cleaned_count = len(df)
        logger.info(f"{tag} Doğrulama tamamlandı: {result.summary}")

        return df, result

    # ── Özel Kontrol Metodları ───────────────────────────────

    def _check_columns(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        tag: str,
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """Gerekli sütunların varlığını kontrol eder."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            result.add_error(f"{tag} Eksik sütunlar: {missing}")
        return df, result

    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tüm OHLCV sütunlarını float'a dönüştürür."""
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _check_timestamps(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        tag: str,
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """Zaman damgalarını doğrular: tekrar ve sıra bozukluğu."""

        # Kronolojik sıralama
        if "timestamp" in df.columns:
            if not df["timestamp"].is_monotonic_increasing:
                result.add_warning(f"{tag} Kronolojik sıra bozuk — otomatik sıralandı.")
                df = df.sort_values("timestamp").reset_index(drop=True)

            # Tekrarlayan timestamp
            dupes = df["timestamp"].duplicated().sum()
            if dupes > 0:
                result.add_warning(f"{tag} {dupes} tekrarlayan timestamp kaldırıldı.")
                df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        return df, result

    def _check_ohlc_values(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        tag: str,
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """OHLC mantıksal tutarlılık kontrolü."""

        # None / NaN değerler
        nan_counts = df[["open", "high", "low", "close", "volume"]].isna().sum()
        total_nans = nan_counts.sum()

        if total_nans > 0:
            if total_nans > self.max_missing_candles:
                result.add_error(
                    f"{tag} {total_nans} NaN değer var (limit: {self.max_missing_candles})."
                    " Veri kalitesi yetersiz."
                )
                return df, result
            else:
                result.add_warning(
                    f"{tag} {total_nans} NaN değer var — forward-fill ile dolduruldu."
                )
                df = df.ffill()

        # Mantıksal kontroller
        invalid_mask = (
            (df["high"] < df["low"]) |          # high < low
            (df["open"] <= 0) |                  # negatif/sıfır open
            (df["close"] <= 0)                   # negatif/sıfır close
        )
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            result.add_warning(
                f"{tag} {invalid_count} mantıksal olarak tutarsız mum kaldırıldı "
                "(high < low veya negatif fiyat)."
            )
            df = df[~invalid_mask].reset_index(drop=True)

        return df, result

    def _check_price_spikes(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        tag: str,
    ) -> ValidationResult:
        """Ani fiyat sıçramalarını (anomali) tespit eder."""
        if len(df) < 2:
            return result

        close_pct_change = df["close"].pct_change().abs()
        spikes = close_pct_change[close_pct_change > self.price_spike_factor]

        if len(spikes) > 0:
            result.add_warning(
                f"{tag} {len(spikes)} fiyat sıçraması tespit edildi "
                f"(>{self.price_spike_factor*100:.0f}% değişim): "
                f"index={spikes.index.tolist()[:5]}"
            )
        return result

    def _check_volume(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        tag: str,
    ) -> ValidationResult:
        """Sıfır hacimli mumları tespit eder."""
        if self.min_volume_threshold > 0:
            zero_vol = (df["volume"] <= self.min_volume_threshold).sum()
            if zero_vol > 0:
                result.add_warning(
                    f"{tag} {zero_vol} mum hacim eşiğinin altında "
                    f"(< {self.min_volume_threshold})."
                )
        return result
