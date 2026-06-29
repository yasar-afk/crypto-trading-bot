# ============================================================
# src/data/collector.py — ANTIGRAVITI Trading Bot
# Amaç : Binance'tan OHLCV verisi çeken ana DataCollector sınıfı.
#         REST polling ile 4h ve 1d mum verisini alır, doğrular,
#         önbellekler ve sisteme temiz DataFrame olarak sunar.
# Tarih: 2026-06-03
#
# MİMARİ KARARLAR:
#   - CCXT: Binance API'sini soyutlar; ileride borsa değişirse
#     sadece exchange adı değişir.
#   - Önbellek: aynı veriyi gereksiz yere tekrar çekmez
#     (API limit tasarrufu).
#   - Validator: her veri çekiminde otomatik kalite kontrolü.
#   - is_paper_trade flag: canlı vs kağıt işlem ayrımı net tutulur.
#
# GELECEK GENİŞLEME (Faz 2):
#   WebSocket desteği için _connect_websocket() metodu eklenecek.
#   Mevcut REST metodları değişmeyecek — backward-compatible.
# ============================================================

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ccxt
import pandas as pd

from src.config.settings import Settings, get_settings
from src.data.models import (
    DataFetchResult,
    DataSource,
    MarketInfo,
    MarketStatus,
    OHLCVCandle,
    OHLCVSeries,
)
from src.data.validator import OHLCVValidator, ValidationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Önbellek Girdisi ─────────────────────────────────────────

class _CacheEntry:
    """Önbellek girdisi: veri + geçerlilik süresi."""

    def __init__(self, data: OHLCVSeries, ttl_seconds: int) -> None:
        self.data = data
        self.expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

    @property
    def is_expired(self) -> bool:
        """Önbellek süresi dolmuş mu?"""
        return datetime.utcnow() > self.expires_at


# ─── Ana DataCollector Sınıfı ────────────────────────────────

class DataCollector:
    """Binance'tan REST API ile OHLCV verisi toplar.

    Tüm veri akışının tek giriş noktasıdır. TechnicalEngine,
    SignalGenerator ve diğer tüm modüller veriyi buradan alır.

    Attributes:
        settings: Sistem konfigürasyonu.
        is_paper_trade: True ise canlı emir gönderilmez (Faz 2'de önemli).
        exchange: CCXT Binance bağlantı nesnesi.
        validator: OHLCV kalite doğrulayıcı.
        _cache: Sembol+timeframe bazlı önbellek.

    Example:
        >>> collector = DataCollector()
        >>> collector.connect()
        >>> result = collector.fetch("BTC/USDT", "4h", limit=200)
        >>> if result.success:
        ...     df = collector.to_dataframe(result.data)
        ...     print(df.tail())
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        is_paper_trade: bool = True,
    ) -> None:
        """DataCollector başlatır.

        Args:
            settings: Konfigürasyon nesnesi. None ise get_settings() kullanılır.
            is_paper_trade: True = kağıt işlem modu (güvenli).
                            False = canlı mod (Faz 2 sonrası).
        """
        self.settings = settings or get_settings()
        self.is_paper_trade = is_paper_trade
        self.exchange: Optional[ccxt.binance] = None
        self.validator = OHLCVValidator(
            max_missing_candles=self.settings.data.validation.max_missing_candles,
            min_volume_threshold=self.settings.data.validation.min_volume_threshold,
            price_spike_factor=self.settings.data.validation.price_spike_factor,
        )
        self._cache: Dict[str, _CacheEntry] = {}
        self._is_connected: bool = False

        logger.info(
            f"DataCollector oluşturuldu | "
            f"Mod: {'📄 PAPER TRADE' if is_paper_trade else '💰 CANLI'} | "
            f"Borsa: {self.settings.exchange.name.upper()}"
        )

    # ── Bağlantı Metodları ───────────────────────────────────

    def connect(self) -> bool:
        """Binance API'sine bağlanır ve bağlantıyı doğrular.

        Returns:
            True: Bağlantı başarılı.
            False: Bağlantı başarısız (hata loglanır).

        Raises:
            ccxt.AuthenticationError: API key geçersizse.
        """
        cfg = self.settings.exchange

        try:
            logger.info(
                f"Binance'a baglaniliyor... "
                f"[Sandbox: {'EVET' if cfg.sandbox else 'HAYIR'}] "
                f"[Mod: {'PAPER' if self.is_paper_trade else 'CANLI'}]"
            )

            exchange_params: dict = {
                "enableRateLimit": cfg.rate_limit,
                "timeout": cfg.request_timeout * 1000,
                "options": {
                    "recvWindow": cfg.recv_window,
                    "defaultType": getattr(cfg, "default_type", "future"),
                },
            }

            # Paper trade modunda API key olmadan da baglanti kurulabilir
            has_creds = self.settings.has_api_credentials
            if has_creds:
                exchange_params["apiKey"] = self.settings.binance_api_key
                exchange_params["secret"] = self.settings.binance_api_secret
                logger.info("API kimlik bilgileri yuklendi.")
            else:
                logger.warning(
                    "API key bulunamadi -- yalnizca public endpoint'ler kullanilabilir."
                )

            self.exchange = ccxt.binance(exchange_params)

            # Sandbox (testnet) modunu etkinlestir
            if cfg.sandbox:
                self.exchange.set_sandbox_mode(True)
                logger.info("[SANDBOX] Testnet modu ETKIN")

            # Piyasalari yukle (sembol dogrulama icin gerekli)
            logger.info("Piyasa bilgileri yukleniyor...")
            try:
                self.exchange.load_markets()
            except ccxt.AuthenticationError as auth_err:
                if self.is_paper_trade and has_creds:
                    # Paper trade modunda API key gecersizse key'siz tekrar dene
                    logger.warning(
                        f"API key gecersiz (paper trade modu) -- key'siz baglanti deneniyor. "
                        f"Hata: {auth_err}"
                    )
                    exchange_params.pop("apiKey", None)
                    exchange_params.pop("secret", None)
                    self.exchange = ccxt.binance(exchange_params)
                    self.exchange.load_markets()
                    logger.info(
                        "[UYARI] Paper trade: API key'siz baglandi (sadece public veri). "
                        "Gercek key olmadan canli islem YAPILAMAZ."
                    )
                else:
                    raise

            logger.info(
                f"[OK] Binance baglantisi basarili | "
                f"{len(self.exchange.markets)} piyasa yuklendi"
            )
            self._is_connected = True
            return True

        except ccxt.AuthenticationError as e:
            logger.error(f"[HATA] Kimlik dogrulama hatasi: {e}")
            logger.error("API key ve secret'i .env dosyasinda kontrol et.")
            return False

        except ccxt.NetworkError as e:
            logger.error(f"[HATA] Ag hatasi: {e}")
            logger.error("Internet baglantini ve Binance'in erisilebilegini kontrol et.")
            return False

        except ccxt.ExchangeError as e:
            logger.error(f"[HATA] Borsa hatasi: {e}")
            return False

        except Exception as e:
            logger.error(f"[HATA] Beklenmeyen baglanti hatasi: {e}", exc_info=True)
            return False

    def disconnect(self) -> None:
        """Bağlantıyı temizler ve kaynakları serbest bırakır."""
        self.exchange = None
        self._is_connected = False
        self._cache.clear()
        logger.info("DataCollector bağlantısı kapatıldı.")

    @property
    def is_connected(self) -> bool:
        """Bağlantı durumunu döner."""
        return self._is_connected and self.exchange is not None

    # ── Veri Çekme Metodları ─────────────────────────────────

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None,
        since: Optional[int] = None,
        use_cache: bool = True,
    ) -> DataFetchResult:
        """Belirtilen sembol ve timeframe için OHLCV verisi çeker.

        Önce önbelleği kontrol eder; geçerli veri varsa API'ye gitmez.
        Veri çekildikten sonra otomatik olarak doğrulanır.

        Args:
            symbol: İşlem çifti (ör. 'BTC/USDT', 'ETH/USDT').
            timeframe: Zaman dilimi ('4h', '1d', vb.).
            limit: Alınacak mum sayısı (None = config'den).
            since: Başlangıç zamanı (Unix ms). None = son N mum.
            use_cache: False ise önbellek atlanır.

        Returns:
            DataFetchResult nesnesi.
        """
        if not self.is_connected:
            return DataFetchResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                error_message="Borsa bağlantısı yok. Önce connect() çağır.",
            )

        limit = limit or self.settings.data.limit
        cache_key = f"{symbol}:{timeframe}"

        # ── Önbellek Kontrolü ────────────────────────────────
        if use_cache and self.settings.data.cache_enabled:
            cached = self._cache.get(cache_key)
            if cached and not cached.is_expired:
                logger.debug(
                    f"[Cache HIT] {symbol}@{timeframe} — "
                    f"{cached.data.candle_count} mum önbellekten döndü"
                )
                return DataFetchResult(
                    success=True,
                    symbol=symbol,
                    timeframe=timeframe,
                    data=cached.data,
                )

        # ── REST API Çağrısı ──────────────────────────────────
        start_time = time.monotonic()

        try:
            logger.info(f"Veri çekiliyor: {symbol} @ {timeframe} | limit={limit}")

            raw_ohlcv: List[List] = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

            fetch_duration_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                f"Ham veri alındı: {len(raw_ohlcv)} mum "
                f"({fetch_duration_ms:.0f}ms)"
            )

            # CCXT formatını DataFrame'e dönüştür
            df = self._raw_to_dataframe(raw_ohlcv)

            # Doğrulama
            df, validation = self.validator.validate(df, symbol, timeframe)

            if not validation.is_valid:
                return DataFetchResult(
                    success=False,
                    symbol=symbol,
                    timeframe=timeframe,
                    error_message=(
                        f"Veri doğrulama başarısız: {validation.errors}"
                    ),
                    fetch_duration_ms=fetch_duration_ms,
                )

            # OHLCVSeries oluştur
            series = self._dataframe_to_series(df, symbol, timeframe)

            # Önbelleğe yaz
            if self.settings.data.cache_enabled:
                self._cache[cache_key] = _CacheEntry(
                    data=series,
                    ttl_seconds=self.settings.data.cache_ttl_seconds,
                )
                logger.debug(f"[Cache SET] {cache_key}")

            return DataFetchResult(
                success=True,
                symbol=symbol,
                timeframe=timeframe,
                data=series,
                fetch_duration_ms=fetch_duration_ms,
            )

        except ccxt.BadSymbol as e:
            logger.error(f"Geçersiz sembol '{symbol}': {e}")
            return DataFetchResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                error_message=f"Geçersiz sembol: {symbol}",
            )

        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit aşıldı, 30sn bekleniyor: {e}")
            time.sleep(30)
            return DataFetchResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                error_message="Rate limit aşıldı — lütfen tekrar dene.",
            )

        except ccxt.NetworkError as e:
            logger.error(f"Ağ hatası: {e}")
            return DataFetchResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                error_message=f"Ağ hatası: {e}",
            )

        except Exception as e:
            logger.error(f"Beklenmeyen hata ({symbol}@{timeframe}): {e}", exc_info=True)
            return DataFetchResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                error_message=str(e),
            )

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, DataFetchResult]:
        """Birden fazla timeframe için veri çeker.

        Args:
            symbol: İşlem çifti.
            timeframes: Timeframe listesi. None = config'deki liste.
            limit: Her timeframe için mum sayısı.

        Returns:
            {timeframe: DataFetchResult} sözlüğü.
        """
        timeframes = timeframes or self.settings.data.timeframes
        results: Dict[str, DataFetchResult] = {}

        logger.info(
            f"Çoklu timeframe çekimi: {symbol} | "
            f"Timeframes: {timeframes}"
        )

        for tf in timeframes:
            result = self.fetch(symbol=symbol, timeframe=tf, limit=limit)
            results[tf] = result

            if result.success:
                logger.info(
                    f"  ✅ {tf}: {result.candle_count} mum "
                    f"| Son kapanış: {result.data.latest_close:.4f}"
                )
            else:
                logger.error(f"  ❌ {tf}: {result.error_message}")

            # Binance rate limit için küçük gecikme
            time.sleep(0.1)

        success_count = sum(1 for r in results.values() if r.success)
        logger.info(
            f"Çoklu timeframe tamamlandı: {success_count}/{len(timeframes)} başarılı"
        )

        return results

    # ── Piyasa Bilgisi ────────────────────────────────────────

    def get_market_info(self, symbol: str) -> Optional[MarketInfo]:
        """Sembol için piyasa meta verisi döner.

        Args:
            symbol: İşlem çifti (ör. 'BTC/USDT').

        Returns:
            MarketInfo nesnesi veya None (sembol bulunamazsa).
        """
        if not self.is_connected:
            logger.error("Piyasa bilgisi için bağlantı gerekli.")
            return None

        try:
            market = self.exchange.market(symbol)
            return MarketInfo(
                symbol=symbol,
                base=market.get("base", ""),
                quote=market.get("quote", ""),
                min_order_size=market.get("limits", {}).get("amount", {}).get("min", 0.0) or 0.0,
                price_precision=market.get("precision", {}).get("price", 8) or 8,
                amount_precision=market.get("precision", {}).get("amount", 8) or 8,
                status=MarketStatus.OPEN if market.get("active") else MarketStatus.CLOSED,
            )
        except Exception as e:
            logger.error(f"Piyasa bilgisi alınamadı ({symbol}): {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Binance ticker'dan anlık (gerçek zamanlı) fiyat çeker.

        OHLCV önbelleğini ATLATIR — her çağrıda Binance'a gider.
        monitor_positions ve portföy raporlama için kullanılır.

        Args:
            symbol: İşlem çifti (ör. 'BTC/USDT').

        Returns:
            Anlık fiyat (float) veya hata durumunda None.
        """
        if not self.is_connected:
            logger.error("Anlık fiyat için bağlantı gerekli.")
            return None

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            if price:
                logger.debug(f"[Ticker] {symbol}: ${price:,.4f}")
                return float(price)
            logger.warning(f"[Ticker] {symbol}: fiyat verisi boş döndü.")
            return None
        except Exception as e:
            logger.error(f"Anlık fiyat alınamadı ({symbol}): {e}")
            return None

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Birden fazla sembol için anlık fiyatları tek seferde çeker (batch).

        Args:
            symbols: Fiyatı istenilen semboller listesi.

        Returns:
            {symbol: fiyat} sözlüğü. Hatalı semboller dahil edilmez.
        """
        if not self.is_connected:
            logger.error("Anlık fiyatlar için bağlantı gerekli.")
            return {}

        prices: Dict[str, float] = {}
        try:
            # Tüm tickerları tek seferde çek (daha verimli)
            tickers = self.exchange.fetch_tickers(symbols)
            for sym, ticker in tickers.items():
                price = ticker.get("last") or ticker.get("close")
                if price:
                    prices[sym] = float(price)
            logger.debug(f"[Batch Ticker] {len(prices)}/{len(symbols)} sembol fiyatı alındı.")
        except Exception as e:
            logger.warning(f"Batch ticker çekimi başarısız, tekil çekime geçiliyor: {e}")
            # Hata durumunda tekil çekim yap
            for sym in symbols:
                p = self.get_current_price(sym)
                if p is not None:
                    prices[sym] = p

        return prices

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Binance Futures'tan sembolün anlık fonlama oranını çeker.

        Args:
            symbol: İşlem çifti (ör. 'BTC/USDT').

        Returns:
            Fonlama oranı (float, ör. 0.0001 = %0.01) veya None.

        Notes:
            Pozitif funding → Long'lar short'lara ödüyor (aşırı iyimserlik)
            Negatif funding → Short'lar long'lara ödüyor (aşırı kötümserlik)
        """
        if not self.is_connected:
            logger.warning("Funding rate için bağlantı gerekli.")
            return None

        try:
            # CCXT üzerinden futures funding rate
            exchange_futures_params = {
                "enableRateLimit": True,
                "timeout": 10000,
                "options": {"defaultType": "future"},
            }
            if hasattr(self.exchange, "apiKey") and self.exchange.apiKey:
                exchange_futures_params["apiKey"] = self.exchange.apiKey
                exchange_futures_params["secret"] = self.exchange.secret

            import ccxt as ccxt_module
            futures_ex = ccxt_module.binance(exchange_futures_params)
            funding_info = futures_ex.fetch_funding_rate(symbol)
            rate = funding_info.get("fundingRate")
            if rate is not None:
                logger.debug(f"[Funding Rate] {symbol}: {float(rate)*100:.4f}%")
                return float(rate)
            return None
        except Exception as e:
            logger.debug(f"Funding rate alınamadı ({symbol}): {e}")
            return None

    def get_open_interest(self, symbol: str) -> Optional[float]:
        """Binance Futures'tan sembolün açık pozisyon (OI) değerini çeker.

        Args:
            symbol: İşlem çifti (ör. 'BTC/USDT').

        Returns:
            Açık pozisyon değeri (float, USDT cinsinden) veya None.
        """
        if not self.is_connected:
            return None

        try:
            import ccxt as ccxt_module
            futures_ex = ccxt_module.binance({
                "enableRateLimit": True,
                "timeout": 10000,
                "options": {"defaultType": "future"},
            })
            oi_data = futures_ex.fetch_open_interest(symbol)
            oi = oi_data.get("openInterestAmount") or oi_data.get("openInterest")
            if oi is not None:
                logger.debug(f"[Open Interest] {symbol}: {float(oi):,.2f}")
                return float(oi)
            return None
        except Exception as e:
            logger.debug(f"Open Interest alınamadı ({symbol}): {e}")
            return None

    def validate_symbol(self, symbol: str) -> bool:
        """Sembolün Binance'ta mevcut olup olmadığını kontrol eder.

        Args:
            symbol: Kontrol edilecek sembol.

        Returns:
            True: Sembol geçerli.
            False: Sembol bulunamadı veya hata.
        """
        if not self.is_connected:
            return False

        try:
            return symbol in self.exchange.markets
        except Exception as e:
            logger.error(f"Sembol doğrulama hatası: {e}")
            return False


    # ── Yardımcı Metodlar ────────────────────────────────────

    @staticmethod
    def _raw_to_dataframe(raw_ohlcv: List[List]) -> pd.DataFrame:
        """CCXT'nin list-of-lists formatını DataFrame'e dönüştürür.

        CCXT çıktı formatı: [timestamp, open, high, low, close, volume]

        Args:
            raw_ohlcv: CCXT fetch_ohlcv() çıktısı.

        Returns:
            Sütun adlandırılmış pandas DataFrame.
        """
        df = pd.DataFrame(
            raw_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        # Timestamp'i datetime'a çevir (milisaniye → UTC datetime)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")

        return df

    @staticmethod
    def _dataframe_to_series(
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> OHLCVSeries:
        """DataFrame'i OHLCVSeries modeline dönüştürür.

        Args:
            df: Doğrulanmış OHLCV DataFrame.
            symbol: İşlem çifti.
            timeframe: Zaman dilimi.

        Returns:
            OHLCVSeries nesnesi.
        """
        candles: List[OHLCVCandle] = []

        for _, row in df.iterrows():
            try:
                candle = OHLCVCandle(
                    timestamp=int(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    # Alan adı: candle_datetime (datetime Python built-in ile çakışmayı önler)
                    candle_datetime=pd.to_datetime(
                        row["timestamp"], unit="ms", utc=True
                    ).to_pydatetime(),
                )
                candles.append(candle)
            except Exception as e:
                logger.warning(f"Mum dönüştürme hatası (atlandı): {e}")
                continue

        return OHLCVSeries(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            source=DataSource.REST,
        )

    def to_dataframe(self, series: OHLCVSeries) -> pd.DataFrame:
        """OHLCVSeries'i analiz için pandas DataFrame'e çevirir.

        TechnicalEngine bu formatı kullanır (indikatör hesaplama için).

        Args:
            series: Dönüştürülecek OHLCVSeries nesnesi.

        Returns:
            datetime index'li OHLCV DataFrame.
        """
        if not series.candles:
            return pd.DataFrame()

        records = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in series.candles
        ]
        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")
        return df

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Önbelleği temizler.

        Args:
            symbol: Belirtilirse yalnızca o sembolün önbelleğini siler.
                    None ise tüm önbelleği temizler.
        """
        if symbol is None:
            self._cache.clear()
            logger.info("Tüm önbellek temizlendi.")
        else:
            removed = [k for k in self._cache if k.startswith(symbol)]
            for k in removed:
                del self._cache[k]
            logger.info(f"'{symbol}' için önbellek temizlendi ({len(removed)} giriş).")

    def get_cache_stats(self) -> Dict:
        """Önbellek durumu hakkında istatistik döner."""
        total = len(self._cache)
        expired = sum(1 for e in self._cache.values() if e.is_expired)
        return {
            "total_entries": total,
            "active_entries": total - expired,
            "expired_entries": expired,
            "keys": list(self._cache.keys()),
        }

    def __repr__(self) -> str:
        status = "Bağlı" if self.is_connected else "Bağlı Değil"
        mode = "Paper Trade" if self.is_paper_trade else "Canlı"
        return f"DataCollector(status={status}, mode={mode})"
