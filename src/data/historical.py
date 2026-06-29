# ============================================================
# src/data/historical.py — Tarihsel Veri Çekme ve Yönetimi
#
# AMAÇ:
#   Binance'tan tarihsel OHLCV verisi çeker, CSV'ye kaydeder
#   ve backtest motoru için veri sağlar.
# ============================================================

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ccxt

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HistoricalDataFetcher:
    """Tarihsel OHLCV verisi çeken ve yöneten sınıf."""

    def __init__(
        self,
        exchange_id: str = "binance",
        rate_limit: bool = True,
        data_dir: str = "data/historical",
    ) -> None:
        """HistoricalDataFetcher başlatır.

        Args:
            exchange_id: Borsa ID'si (ccxt uyumlu).
            rate_limit: Rate limiting aktif mi.
            data_dir: Verilerin kaydedileceği dizin.
        """
        self.exchange_id = exchange_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            "enableRateLimit": rate_limit,
            "options": {"defaultType": "spot"},
        })

        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
        since: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Tek sembol için OHLCV verisi çeker.

        Args:
            symbol: İşlem çifti (ör. "BTC/USDT").
            timeframe: Zaman dilimi (ör. "1h", "4h", "1d").
            limit: Çekilecek mum sayısı.
            since: Başlangıç zaman damgası (ms).
            use_cache: Önbellek kullanılsın mı.

        Returns:
            OHLCV DataFrame'i (datetime index).
        """
        cache_key = f"{symbol}_{timeframe}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        all_ohlcv = []
        current_since = since
        remaining = limit

        while remaining > 0:
            batch_size = min(remaining, 1000)
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_since,
                    limit=batch_size,
                )
                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                remaining -= len(ohlcv)

                if len(ohlcv) < batch_size:
                    break

                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Veri çekme hatası ({symbol}, {timeframe}): {e}")
                break

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        if use_cache:
            self._cache[cache_key] = df

        return df

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str],
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """Çoklu timeframe verisi çeker.

        Args:
            symbol: İşlem çifti.
            timeframes: Zaman dilimi listesi.
            limit: Her timeframe için mum sayısı.

        Returns:
            {timeframe: DataFrame} sözlüğü.
        """
        results = {}
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, tf, limit)
            if not df.empty:
                results[tf] = df
                logger.info(f"  {symbol}@{tf}: {len(df)} mum çekildi")
            else:
                logger.warning(f"  {symbol}@{tf}: Veri alınamadı")
        return results

    def fetch_top_symbols(
        self,
        top_n: int = 100,
        quote: str = "USDT",
    ) -> List[str]:
        """En yüksek hacimli N sembolü çeker.

        Args:
            top_n: çekilecek sembol sayısı.
            quote: Quote para birimi.

        Returns:
            Sembol listesi.
        """
        try:
            tickers = self.exchange.fetch_tickers()
            usdt_pairs = []

            for sym, ticker in tickers.items():
                if not sym.endswith(f"/{quote}"):
                    continue
                if not ticker.get("active", True):
                    continue

                vol = ticker.get("quoteVolume") or 0.0
                if vol > 0:
                    usdt_pairs.append((sym, vol))

            usdt_pairs.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [sym for sym, _ in usdt_pairs[:top_n]]

            logger.info(f"En yüksek hacimli {len(top_symbols)} {quote} çifti bulundu")
            return top_symbols

        except Exception as e:
            logger.error(f"Sembol listesi çekilemedi: {e}")
            return []

    def save_csv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Path:
        """DataFrame'i CSV'ye kaydeder.

        Args:
            df: Kaydedilecek DataFrame.
            symbol: Sembol adı.
            timeframe: Zaman dilimi.

        Returns:
            Kaydedilen dosya yolu.
        """
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath)
        logger.info(f"CSV kaydedildi: {filepath}")
        return filepath

    def load_csv(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """CSV'den DataFrame yükler.

        Args:
            symbol: Sembol adı.
            timeframe: Zaman dilimi.

        Returns:
            DataFrame veya None.
        """
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        filepath = self.data_dir / filename

        if not filepath.exists():
            return None

        df = pd.read_csv(filepath, index_col="datetime", parse_dates=True)
        return df

    def fetch_and_save(
        self,
        symbols: List[str],
        timeframes: List[str],
        limit: int = 500,
    ) -> Dict[str, Dict[str, Path]]:
        """Verileri çekip CSV'ye kaydeder.

        Args:
            symbols: Sembol listesi.
            timeframes: Zaman dilimi listesi.
            limit: Her timeframe için mum sayısı.

        Returns:
            {symbol: {timeframe: filepath}} sözlüğü.
        """
        saved = {}
        total = len(symbols) * len(timeframes)
        done = 0

        for symbol in symbols:
            saved[symbol] = {}
            for tf in timeframes:
                done += 1
                logger.info(f"[{done}/{total}] {symbol}@{tf} çekiliyor...")

                df = self.fetch_ohlcv(symbol, tf, limit)
                if not df.empty:
                    filepath = self.save_csv(df, symbol, tf)
                    saved[symbol][tf] = filepath

                time.sleep(0.2)

        return saved

    def get_cached_data(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Önbellekteki veriyi döndürür.

        Args:
            symbol: Sembol adı.
            timeframe: Zaman dilimi.

        Returns:
            DataFrame veya None.
        """
        cache_key = f"{symbol}_{timeframe}"
        return self._cache.get(cache_key)

    def clear_cache(self) -> None:
        """Önbelleği temizler."""
        self._cache.clear()
