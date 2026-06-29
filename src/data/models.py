# ============================================================
# src/data/models.py — ANTIGRAVITI Trading Bot
# Amaç : Sistemin tüm veri yapılarını tek yerde tanımlar.
#         DataCollector bu modelleri üretir; diğer tüm modüller
#         (TechnicalEngine, SignalGenerator vb.) bu şemaları tüketir.
# Tarih: 2026-06-03
#
# MİMARİ NOT:
#   Pydantic modelleri kullandık çünkü:
#   1. Otomatik tip doğrulaması → hatalı veri modüle giremez
#   2. .model_dump() ile kolayca dict/JSON'a dönüşür
#   3. Her alan için belge (description) eklenebilir
#   4. Downstream modüllere güvenli arayüz sağlar
#
# GENİŞLETME:
#   Yeni indikatör eklemek istersen CandleData'ya alan ekle ya da
#   yeni bir model (e.g. SignalData) bu dosyaya ekle.
# ============================================================

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ─── Enum Tanımları ──────────────────────────────────────────

class Timeframe(str, Enum):
    """Desteklenen zaman dilimleri."""
    M1  = "1m"
    M5  = "5m"
    M15 = "15m"
    H1  = "1h"
    H4  = "4h"
    D1  = "1d"
    W1  = "1w"


class DataSource(str, Enum):
    """Verinin nereden geldiği."""
    REST      = "rest"
    WEBSOCKET = "websocket"
    CACHE     = "cache"
    BACKTEST  = "backtest"


class MarketStatus(str, Enum):
    """Piyasa durumu."""
    OPEN    = "open"
    CLOSED  = "closed"
    UNKNOWN = "unknown"


# ─── Temel Veri Modelleri ────────────────────────────────────

class OHLCVCandle(BaseModel):
    """Tek bir mum (candlestick) verisi.

    Attributes:
        timestamp: Mumun açılış zamanı (Unix ms).
        open: Açılış fiyatı.
        high: En yüksek fiyat.
        low: En düşük fiyat.
        close: Kapanış fiyatı.
        volume: İşlem hacmi (base currency).
        candle_datetime: İnsan-okunur zaman damgası.
    """

    timestamp: int = Field(..., description="Unix timestamp (milisaniye)")
    open: float   = Field(..., gt=0, description="Açılış fiyatı")
    high: float   = Field(..., gt=0, description="En yüksek fiyat")
    low: float    = Field(..., gt=0, description="En düşük fiyat")
    close: float  = Field(..., gt=0, description="Kapanış fiyatı")
    volume: float = Field(..., ge=0, description="İşlem hacmi")
    # Python 3.10 + Pydantic v2 uyumu: Optional alanı None default ile
    candle_datetime: Optional[datetime] = None

    model_config = {"frozen": True}   # Immutable — değer değişemez

    @model_validator(mode="after")
    def check_ohlc_logic(self) -> "OHLCVCandle":
        """OHLC mantıksal tutarlılık: high >= low olmalı.

        Pydantic v2'de çapraz alan (cross-field) doğrulaması için
        field_validator yerine model_validator(mode='after') kullanılır.
        Bu sayede tüm alanlar set edildikten sonra kontrol yapılır.
        """
        if self.high < self.low:
            raise ValueError(
                f"high ({self.high}) < low ({self.low}): Geçersiz mum verisi."
            )
        return self

    @property
    def body_size(self) -> float:
        """Mum gövde boyutu (|close - open|)."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Yükselişçi mum mu?"""
        return self.close >= self.open

    @property
    def range_size(self) -> float:
        """Toplam mum aralığı (high - low)."""
        return self.high - self.low


class OHLCVSeries(BaseModel):
    """Belirli bir sembol ve timeframe için mum verisi koleksiyonu.

    Attributes:
        symbol: İşlem çifti (ör. 'BTC/USDT').
        timeframe: Zaman dilimi.
        candles: Kronolojik sıralanmış mum listesi.
        source: Verinin kaynağı (REST/WS/cache).
        fetched_at: Bu verinin ne zaman çekildiği.
    """

    symbol: str
    timeframe: str
    candles: List[OHLCVCandle] = Field(default_factory=list)
    source: DataSource = DataSource.REST
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def latest_candle(self) -> Optional[OHLCVCandle]:
        """En son (en güncel) mum."""
        return self.candles[-1] if self.candles else None

    @property
    def oldest_candle(self) -> Optional[OHLCVCandle]:
        """En eski mum."""
        return self.candles[0] if self.candles else None

    @property
    def candle_count(self) -> int:
        """Toplam mum sayısı."""
        return len(self.candles)

    @property
    def latest_close(self) -> Optional[float]:
        """En son kapanış fiyatı."""
        return self.latest_candle.close if self.latest_candle else None

    def to_dict_list(self) -> List[Dict]:
        """Pandas DataFrame'e dönüştürmek için dict listesi döner."""
        return [c.model_dump() for c in self.candles]


class MarketInfo(BaseModel):
    """Borsa piyasa bilgisi (sembol meta verisi).

    Attributes:
        symbol: İşlem çifti.
        base: Temel para birimi (ör. 'BTC').
        quote: Karşı para birimi (ör. 'USDT').
        min_order_size: Minimum emir miktarı.
        price_precision: Fiyat hassasiyeti (ondalık basamak).
        amount_precision: Miktar hassasiyeti.
        status: Piyasa durumu.
    """

    symbol: str
    base: str
    quote: str
    min_order_size: float = 0.0
    price_precision: int = 8
    amount_precision: int = 8
    status: MarketStatus = MarketStatus.OPEN


class DataFetchResult(BaseModel):
    """DataCollector.fetch() metodunun dönüş nesnesi.

    Hem başarılı hem başarısız veri çekimlerini temsil eder.

    Attributes:
        success: İşlem başarılı mı?
        symbol: Hedef sembol.
        timeframe: Hedef timeframe.
        data: Başarılı ise OHLCVSeries, değilse None.
        error_message: Hata varsa açıklama.
        fetch_duration_ms: API çağrısının sürdüğü süre (ms).
    """

    success: bool
    symbol: str
    timeframe: str
    data: Optional[OHLCVSeries] = None
    error_message: Optional[str] = None
    fetch_duration_ms: float = 0.0

    @property
    def candle_count(self) -> int:
        """Dönen mum sayısı (başarısızsa 0)."""
        return self.data.candle_count if self.data else 0
