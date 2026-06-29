# ============================================================
# src/technical/engine.py — ANTIGRAVITI Trading Bot
#
# AMAÇ:
#   DataCollector'dan gelen temiz OHLCV DataFrame'ini alır,
#   tüm teknik indikatörleri hesaplar ve her bar için
#   yorumlanmış IndicatorSet nesnesi üretir.
#
# MİMARİ NOT:
#   - "ta" kütüphanesi ile vektörel hesaplama (loop yok → hız).
#   - Her indikatör ayrı private metod → test edilebilir, izole.
#   - enrich_dataframe(): tüm indikatörleri DataFrame sütunu olarak ekler
#     → görselleştirme ve backtesting için.
#   - get_latest_indicators(): sadece son bar için IndicatorSet üretir
#     → canlı sinyal üretimi için.
#
# KULLANILABİLİR PARAMETRELER (config.yaml'a eklenecek):
#   rsi_period, macd_fast/slow/signal, ema_periods,
#   atr_period, bb_period, bb_std
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v1.0 | RSI, MACD, EMA, ATR, Bollinger, Volume
# ============================================================

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume as ta_volume

from src.technical.indicators import (
    ADXResult,
    ATRResult,
    BBPosition,
    BollingerResult,
    EMAAlignment,
    EMAResult,
    FibonacciResult,
    IndicatorSet,
    MACDCrossType,
    MACDResult,
    PatternResult,
    RSIResult,
    RSIZone,
    SignalDirection,
    VolumeResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Varsayılan İndikatör Parametreleri ──────────────────────
# config.yaml'a taşınacak; şimdilik burada sabit.

DEFAULT_RSI_PERIOD     = 14
DEFAULT_MACD_FAST      = 12
DEFAULT_MACD_SLOW      = 26
DEFAULT_MACD_SIGNAL    = 9
DEFAULT_EMA_PERIODS    = [20, 50, 200]
DEFAULT_ATR_PERIOD     = 14
DEFAULT_ATR_MULTIPLIER = 2.0            # Stop-loss = fiyat ± ATR × bu değer
DEFAULT_RR_RATIO       = 2.0            # Minimum Risk/Ödül oranı
DEFAULT_BB_PERIOD      = 20
DEFAULT_BB_STD         = 2.0
DEFAULT_VOLUME_PERIOD  = 20             # Ortalama hacim için dönem
DEFAULT_BB_SQUEEZE_THR = 0.10          # Bant genişliği bu değerin altındaysa squeeze

# RSI bölge eşikleri
RSI_OVERSOLD        = 30
RSI_NEAR_OVERSOLD   = 40
RSI_NEAR_OVERBOUGHT = 60
RSI_OVERBOUGHT      = 70

# Divergence tespiti için geriye bakış
DIVERGENCE_LOOKBACK = 10


# ─── Ana TechnicalEngine Sınıfı ───────────────────────────────

class TechnicalEngine:
    """OHLCV verisinden teknik indikatörleri hesaplar ve yorumlar.

    DataCollector → TechnicalEngine → SignalGenerator akışında
    orta katmandır. Ham fiyat verisini insan/makine yorumlanabilir
    sinyallere dönüştürür.

    Attributes:
        rsi_period: RSI hesaplama periyodu.
        macd_fast: MACD hızlı periyodu.
        macd_slow: MACD yavaş periyodu.
        macd_signal: MACD sinyal periyodu.
        ema_periods: EMA periyotları listesi.
        atr_period: ATR periyodu.
        atr_multiplier: Stop-loss için ATR çarpanı.
        rr_ratio: Minimum Risk/Ödül oranı.
        bb_period: Bollinger Band periyodu.
        bb_std: Bollinger Band standart sapma sayısı.

    Example:
        >>> engine = TechnicalEngine()
        >>> df = collector.to_dataframe(series)
        >>> enriched_df = engine.enrich_dataframe(df)
        >>> indicator_set = engine.get_latest_indicators(df, "BTC/USDT", "4h")
        >>> print(indicator_set.summary())
    """

    def __init__(
        self,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        macd_fast: int = DEFAULT_MACD_FAST,
        macd_slow: int = DEFAULT_MACD_SLOW,
        macd_signal: int = DEFAULT_MACD_SIGNAL,
        ema_periods: List[int] = None,
        atr_period: int = DEFAULT_ATR_PERIOD,
        atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
        rr_ratio: float = DEFAULT_RR_RATIO,
        bb_period: int = DEFAULT_BB_PERIOD,
        bb_std: float = DEFAULT_BB_STD,
        volume_period: int = DEFAULT_VOLUME_PERIOD,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
    ) -> None:
        """TechnicalEngine başlatır.

        Args:
            rsi_period: RSI periyodu (varsayılan: 14).
            macd_fast: MACD hızlı EMA periyodu (varsayılan: 12).
            macd_slow: MACD yavaş EMA periyodu (varsayılan: 26).
            macd_signal: MACD sinyal EMA periyodu (varsayılan: 9).
            ema_periods: EMA periyotları (varsayılan: [20, 50, 200]).
            atr_period: ATR periyodu (varsayılan: 14).
            atr_multiplier: Stop-loss ATR çarpanı (varsayılan: 2.0).
            rr_ratio: Min Risk/Ödül oranı (varsayılan: 2.0).
            bb_period: Bollinger Band periyodu (varsayılan: 20).
            bb_std: Bollinger Band std sayısı (varsayılan: 2.0).
            volume_period: Ortalama hacim periyodu (varsayılan: 20).
            adx_period: ADX periyodu (varsayılan: 14).
            adx_threshold: ADX trend gücü eşiği (varsayılan: 20.0).
        """
        self.rsi_period    = rsi_period
        self.macd_fast     = macd_fast
        self.macd_slow     = macd_slow
        self.macd_signal   = macd_signal
        self.ema_periods   = ema_periods or DEFAULT_EMA_PERIODS
        self.atr_period    = atr_period
        self.atr_multiplier = atr_multiplier
        self.rr_ratio      = rr_ratio
        self.bb_period     = bb_period
        self.bb_std        = bb_std
        self.volume_period = volume_period
        self.adx_period    = adx_period
        self.adx_threshold = adx_threshold


        # Minimum gerekli bar sayısı (en uzun periyoda göre)
        self.min_bars = max(
            self.macd_slow + self.macd_signal,
            max(self.ema_periods),
            DIVERGENCE_LOOKBACK + self.rsi_period,
        ) + 5  # Güvenlik tamponu

        logger.info(
            f"TechnicalEngine başlatıldı | "
            f"RSI={rsi_period} | MACD={macd_fast}/{macd_slow}/{macd_signal} | "
            f"EMA={self.ema_periods} | ATR={atr_period} | "
            f"Min bar: {self.min_bars}"
        )

    # ── Ana Public Metodlar ───────────────────────────────────

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame'e tüm indikatör sütunlarını ekler.

        Backtesting ve görselleştirme için kullanılır.
        Her indikatör ayrı sütun olarak DataFrame'e eklenir.

        Args:
            df: OHLCV DataFrame (open, high, low, close, volume sütunları).

        Returns:
            Tüm indikatör sütunları eklenmiş DataFrame.
            Orijinal DataFrame değiştirilmez (kopya döner).
        """
        if not self._validate_dataframe(df):
            return df.copy()

        result = df.copy()
        tag = "[TechnicalEngine.enrich]"

        try:
            # RSI
            rsi_ind = ta.momentum.RSIIndicator(
                close=result["close"], window=self.rsi_period
            )
            result["rsi"] = rsi_ind.rsi()
            logger.debug(f"{tag} RSI hesaplandı")

            # MACD
            macd_ind = ta.trend.MACD(
                close=result["close"],
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal,
            )
            result["macd"]        = macd_ind.macd()
            result["macd_signal"] = macd_ind.macd_signal()
            result["macd_hist"]   = macd_ind.macd_diff()
            logger.debug(f"{tag} MACD hesaplandı")

            # EMA'lar
            for period in self.ema_periods:
                ema_ind = ta.trend.EMAIndicator(
                    close=result["close"], window=period
                )
                result[f"ema{period}"] = ema_ind.ema_indicator()
            logger.debug(f"{tag} EMA {self.ema_periods} hesaplandı")

            # ATR
            atr_ind = ta.volatility.AverageTrueRange(
                high=result["high"],
                low=result["low"],
                close=result["close"],
                window=self.atr_period,
            )
            result["atr"] = atr_ind.average_true_range()
            result["atr_pct"] = result["atr"] / result["close"] * 100
            logger.debug(f"{tag} ATR hesaplandı")

            # Bollinger Bands
            bb_ind = ta.volatility.BollingerBands(
                close=result["close"],
                window=self.bb_period,
                window_dev=self.bb_std,
            )
            result["bb_upper"]     = bb_ind.bollinger_hband()
            result["bb_middle"]    = bb_ind.bollinger_mavg()
            result["bb_lower"]     = bb_ind.bollinger_lband()
            result["bb_bandwidth"] = bb_ind.bollinger_wband()
            result["bb_pct_b"]     = bb_ind.bollinger_pband()
            logger.debug(f"{tag} Bollinger Bands hesaplandı")

            # Hacim Ortalaması
            result["volume_avg"] = (
                result["volume"].rolling(window=self.volume_period).mean()
            )
            result["volume_ratio"] = result["volume"] / result["volume_avg"]
            logger.debug(f"{tag} Volume analizi tamamlandı")

            # ADX
            adx_ind = ta.trend.ADXIndicator(
                high=result["high"],
                low=result["low"],
                close=result["close"],
                window=self.adx_period,
            )
            result["adx"] = adx_ind.adx()
            result["adx_pos"] = adx_ind.adx_pos()
            result["adx_neg"] = adx_ind.adx_neg()
            logger.debug(f"{tag} ADX hesaplandı")

            logger.info(
                f"{tag} DataFrame zenginleştirildi: "
                f"{len(result)} bar | {len(result.columns)} sütun"
            )

        except Exception as e:
            logger.error(f"{tag} İndikatör hesaplama hatası: {e}", exc_info=True)

        return result

    def get_latest_indicators(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "",
    ) -> Optional[IndicatorSet]:
        """Son bar için yorumlanmış IndicatorSet üretir.

        Canlı sinyal üretimi için kullanılan birincil metoddur.
        Önce enrich_dataframe() çağırır, ardından son satırı yorumlar.

        Args:
            df: OHLCV DataFrame.
            symbol: İşlem çifti (loglama için).
            timeframe: Zaman dilimi (loglama için).

        Returns:
            IndicatorSet nesnesi veya None (yeterli veri yoksa).
        """
        if not self._validate_dataframe(df):
            return None

        # Tüm indikatörleri hesapla
        enriched = self.enrich_dataframe(df)
        last = enriched.iloc[-1]
        current_price = float(last["close"])
        timestamp = int(last.get("timestamp", 0))

        # Her indikatörü yorumla
        rsi_result      = self._interpret_rsi(enriched)
        macd_result     = self._interpret_macd(enriched)
        ema_result      = self._interpret_ema(enriched, current_price)
        atr_result      = self._interpret_atr(enriched, current_price)
        bb_result       = self._interpret_bollinger(enriched, current_price)
        volume_result   = self._interpret_volume(enriched)
        adx_result      = self._interpret_adx(enriched)
        fib_result      = self._interpret_fibonacci(enriched)
        patterns_result = self._detect_patterns(enriched)

        # V3 / V4 Price Action Sinyallerini Hesapla
        try:
            from src.config.settings import get_settings
            cfg = get_settings()
            strategy_version = getattr(cfg.strategy, "version", "v3")
            
            if strategy_version == "v5":
                from src.strategy.v5_pa_strategy import V5PriceActionStrategy
                use_session = getattr(cfg.strategy, "use_session_filter", False)
                trend_ema_val = getattr(cfg.strategy, "trend_ema", 180)
                pa_strat = V5PriceActionStrategy(
                    sweep_window=cfg.strategy.sweep_window,
                    max_hold_sweep=cfg.strategy.max_hold_sweep,
                    target_rr=cfg.risk.min_risk_reward_ratio,
                    require_trend=cfg.strategy.require_trend,
                    trend_ema=trend_ema_val,
                    use_premium_discount=cfg.strategy.use_premium_discount,
                    atr_multiplier=cfg.strategy.displacement_atr_mult,
                    use_session_filter=use_session,
                )
            elif strategy_version == "v4":
                from src.strategy.v4_pa_strategy import V4PriceActionStrategy
                use_session = getattr(cfg.strategy, "use_session_filter", True)
                use_ote = getattr(cfg.strategy, "use_ote_filter", False)
                use_volume = getattr(cfg.strategy, "use_volume_filter", False)
                use_funding_oi = getattr(cfg.strategy, "use_funding_oi_filter", False)
                
                pa_strat = V4PriceActionStrategy(
                    sweep_window=cfg.strategy.sweep_window,
                    max_hold_sweep=cfg.strategy.max_hold_sweep,
                    target_rr=cfg.risk.min_risk_reward_ratio,
                    partial_rr=1.5,
                    require_trend=cfg.strategy.require_trend,
                    displacement_atr_mult=cfg.strategy.displacement_atr_mult,
                    use_premium_discount=cfg.strategy.use_premium_discount,
                    use_session_filter=use_session,
                    use_ote_filter=use_ote,
                    use_volume_filter=use_volume,
                    use_funding_oi_filter=use_funding_oi,
                )
            else:
                from src.strategy.v3_pa_strategy import V3PriceActionStrategy
                pa_strat = V3PriceActionStrategy(
                    sweep_window=cfg.strategy.sweep_window,
                    max_hold_sweep=cfg.strategy.max_hold_sweep,
                    target_rr=cfg.risk.min_risk_reward_ratio,
                    partial_rr=1.5,
                    require_trend=cfg.strategy.require_trend,
                    displacement_atr_mult=cfg.strategy.displacement_atr_mult,
                    use_premium_discount=cfg.strategy.use_premium_discount,
                )
                
            pa_df = pa_strat.calculate_signals(df)
            last_pa = pa_df.iloc[-1]
            pa_signal = last_pa.get("signal", "HOLD")
            pa_entry = float(last_pa.get("entry_price", 0.0))
            pa_sl = float(last_pa.get("sl_price", 0.0))
            pa_tp = float(last_pa.get("tp_price", 0.0))
            pa_fvg = bool(last_pa.get("has_fvg", False))
        except Exception as pa_err:
            logger.error(f"PA Sinyal hesaplama hatası: {pa_err}")
            pa_signal = "HOLD"
            pa_entry = pa_sl = pa_tp = 0.0
            pa_fvg = False

        # IndicatorSet oluştur
        indicator_set = IndicatorSet(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            current_price=current_price,
            rsi=rsi_result,
            macd=macd_result,
            ema=ema_result,
            atr=atr_result,
            bollinger=bb_result,
            volume=volume_result,
            adx=adx_result,
            fib=fib_result,
            patterns=patterns_result,
            pa_signal=pa_signal,
            pa_entry_price=pa_entry,
            pa_sl_price=pa_sl,
            pa_tp_price=pa_tp,
            pa_has_fvg=pa_fvg,
        )

        # Ağırlıklı puanı hesapla
        indicator_set.calculate_weighted_score()

        logger.info(
            f"[{symbol}@{timeframe}] Indikatörler hesaplandı | "
            f"Fiyat: {current_price:,.2f} | "
            f"Ağırlıklı Skor: {indicator_set.weighted_score:.3f}"
        )

        return indicator_set

    # ── RSI Yorumlama ─────────────────────────────────────────

    def _interpret_rsi(self, df: pd.DataFrame) -> Optional[RSIResult]:
        """RSI değerini hesaplar ve yorumlar.

        Args:
            df: enrich_dataframe() çıktısı (rsi sütunu mevcut).

        Returns:
            RSIResult nesnesi veya None.
        """
        if "rsi" not in df.columns or df["rsi"].isna().all():
            return None

        value = float(df["rsi"].iloc[-1])
        if np.isnan(value):
            return None

        # Bölge belirleme
        if value < RSI_OVERSOLD:
            zone = RSIZone.OVERSOLD
        elif value < RSI_NEAR_OVERSOLD:
            zone = RSIZone.NEAR_OVERSOLD
        elif value < RSI_NEAR_OVERBOUGHT:
            zone = RSIZone.NEUTRAL
        elif value < RSI_OVERBOUGHT:
            zone = RSIZone.NEAR_OVERBOUGHT
        else:
            zone = RSIZone.OVERBOUGHT

        # Temel sinyal ve güç
        if zone == RSIZone.OVERSOLD:
            signal, strength = SignalDirection.STRONG_BUY, 0.90
        elif zone == RSIZone.NEAR_OVERSOLD:
            signal, strength = SignalDirection.BUY, 0.65
        elif zone == RSIZone.NEAR_OVERBOUGHT:
            signal, strength = SignalDirection.SELL, 0.65
        elif zone == RSIZone.OVERBOUGHT:
            signal, strength = SignalDirection.STRONG_SELL, 0.90
        else:
            signal, strength = SignalDirection.NEUTRAL, 0.50

        # Divergence tespiti
        bull_div, bear_div, div_note = self._detect_rsi_divergence(df)

        # Divergence varsa sinyal güncellenir
        if bull_div:
            signal, strength = SignalDirection.STRONG_BUY, 0.95
        elif bear_div:
            signal, strength = SignalDirection.STRONG_SELL, 0.95

        return RSIResult(
            value=round(value, 2),
            zone=zone,
            signal=signal,
            signal_strength=strength,
            is_bullish_divergence=bull_div,
            is_bearish_divergence=bear_div,
            divergence_note=div_note,
        )

    def _detect_rsi_divergence(
        self, df: pd.DataFrame
    ) -> Tuple[bool, bool, str]:
        """Fiyat-RSI uyumsuzluğunu (divergence) tespit eder.

        Bullish Divergence: Fiyat yeni dip yaparken RSI yapmıyor.
        Bearish Divergence: Fiyat yeni zirve yaparken RSI yapmıyor.

        Args:
            df: RSI sütunu eklenmiş DataFrame.

        Returns:
            Tuple[is_bullish_div, is_bearish_div, açıklama]
        """
        lookback = DIVERGENCE_LOOKBACK
        if len(df) < lookback + 2:
            return False, False, ""

        recent = df.tail(lookback)
        price_series = recent["close"]
        rsi_series   = recent["rsi"].dropna()

        if len(rsi_series) < 4:
            return False, False, ""

        price_low_now  = float(price_series.iloc[-1])
        price_low_prev = float(price_series.min())
        rsi_now        = float(rsi_series.iloc[-1])
        rsi_at_prev_low = float(rsi_series.iloc[price_series.values.argmin()])

        # Bullish: Fiyat daha düşük dip, RSI daha yüksek dip
        bull_div = (
            price_low_now <= price_low_prev * 1.005   # Fiyat yeni/eşit dip
            and rsi_now > rsi_at_prev_low + 2          # RSI daha yüksek
            and rsi_now < RSI_NEAR_OVERBOUGHT          # RSI henüz aşırı alımda değil
        )

        price_high_now  = float(price_series.iloc[-1])
        price_high_prev = float(price_series.max())
        rsi_at_prev_high = float(rsi_series.iloc[price_series.values.argmax()])

        # Bearish: Fiyat daha yüksek zirve, RSI daha düşük zirve
        bear_div = (
            price_high_now >= price_high_prev * 0.995  # Fiyat yeni/eşit zirve
            and rsi_now < rsi_at_prev_high - 2          # RSI daha düşük
            and rsi_now > RSI_NEAR_OVERSOLD             # RSI henüz aşırı satımda değil
        )

        note = ""
        if bull_div:
            note = (
                f"Bullish Divergence: Fiyat dip={price_low_now:.2f}, "
                f"RSI={rsi_now:.1f} > önceki RSI={rsi_at_prev_low:.1f}"
            )
        elif bear_div:
            note = (
                f"Bearish Divergence: Fiyat zirve={price_high_now:.2f}, "
                f"RSI={rsi_now:.1f} < önceki RSI={rsi_at_prev_high:.1f}"
            )

        return bull_div, bear_div, note

    # ── MACD Yorumlama ────────────────────────────────────────

    def _interpret_macd(self, df: pd.DataFrame) -> Optional[MACDResult]:
        """MACD değerini hesaplar ve yorumlar.

        Args:
            df: enrich_dataframe() çıktısı.

        Returns:
            MACDResult nesnesi veya None.
        """
        required = {"macd", "macd_signal", "macd_hist"}
        if not required.issubset(df.columns):
            return None

        last     = df.iloc[-1]
        prev     = df.iloc[-2] if len(df) > 1 else last

        macd_val = float(last["macd"])
        sig_val  = float(last["macd_signal"])
        hist_val = float(last["macd_hist"])
        prev_hist = float(prev["macd_hist"])

        if any(np.isnan(x) for x in [macd_val, sig_val, hist_val]):
            return None

        # Kesişim tespiti
        prev_macd = float(prev["macd"])
        prev_sig  = float(prev["macd_signal"])
        if prev_macd <= prev_sig and macd_val > sig_val:
            cross_type = MACDCrossType.BULLISH_CROSS
        elif prev_macd >= prev_sig and macd_val < sig_val:
            cross_type = MACDCrossType.BEARISH_CROSS
        else:
            cross_type = MACDCrossType.NO_CROSS

        # Histogram trendi (büyüyor mu?)
        hist_trend = abs(hist_val) > abs(prev_hist)

        # Sinyal belirleme
        if cross_type == MACDCrossType.BULLISH_CROSS:
            signal, strength = SignalDirection.STRONG_BUY, 0.85
        elif cross_type == MACDCrossType.BEARISH_CROSS:
            signal, strength = SignalDirection.STRONG_SELL, 0.85
        elif macd_val > sig_val and hist_val > 0:
            signal = SignalDirection.BUY
            strength = min(0.70, 0.50 + abs(hist_val) / (abs(macd_val) + 1e-9) * 0.20)
        elif macd_val < sig_val and hist_val < 0:
            signal = SignalDirection.SELL
            strength = min(0.70, 0.50 + abs(hist_val) / (abs(macd_val) + 1e-9) * 0.20)
        else:
            signal, strength = SignalDirection.NEUTRAL, 0.50

        return MACDResult(
            macd_line=round(macd_val, 6),
            signal_line=round(sig_val, 6),
            histogram=round(hist_val, 6),
            cross_type=cross_type,
            signal=signal,
            signal_strength=round(strength, 3),
            histogram_trend=hist_trend,
        )

    # ── EMA Yorumlama ─────────────────────────────────────────

    def _interpret_ema(
        self, df: pd.DataFrame, current_price: float
    ) -> Optional[EMAResult]:
        """EMA 20/50/200 değerlerini hesaplar ve yorumlar.

        Args:
            df: enrich_dataframe() çıktısı.
            current_price: Son kapanış fiyatı.

        Returns:
            EMAResult nesnesi veya None.
        """
        cols = [f"ema{p}" for p in self.ema_periods]
        if not all(c in df.columns for c in cols):
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        ema20  = float(last.get("ema20",  np.nan))
        ema50  = float(last.get("ema50",  np.nan))
        ema200 = float(last.get("ema200", np.nan))

        if any(np.isnan(x) for x in [ema20, ema50, ema200]):
            return None

        # Hizalama
        if ema20 > ema50 > ema200:
            alignment = EMAAlignment.FULL_BULLISH
        elif ema20 > ema50 or ema50 > ema200:
            alignment = EMAAlignment.PARTIAL_BULL
        elif ema20 < ema50 < ema200:
            alignment = EMAAlignment.FULL_BEARISH
        elif ema20 < ema50 or ema50 < ema200:
            alignment = EMAAlignment.PARTIAL_BEAR
        else:
            alignment = EMAAlignment.NEUTRAL

        # Golden/Death Cross
        prev_ema50  = float(prev.get("ema50",  np.nan))
        prev_ema200 = float(prev.get("ema200", np.nan))
        golden = (
            not np.isnan(prev_ema50) and not np.isnan(prev_ema200)
            and prev_ema50 <= prev_ema200 and ema50 > ema200
        )
        death = (
            not np.isnan(prev_ema50) and not np.isnan(prev_ema200)
            and prev_ema50 >= prev_ema200 and ema50 < ema200
        )

        # Sinyal
        if alignment == EMAAlignment.FULL_BULLISH or golden:
            signal, strength = SignalDirection.STRONG_BUY, 0.85
        elif alignment == EMAAlignment.PARTIAL_BULL and current_price > ema200:
            signal, strength = SignalDirection.BUY, 0.65
        elif alignment == EMAAlignment.FULL_BEARISH or death:
            signal, strength = SignalDirection.STRONG_SELL, 0.85
        elif alignment == EMAAlignment.PARTIAL_BEAR and current_price < ema200:
            signal, strength = SignalDirection.SELL, 0.65
        else:
            signal, strength = SignalDirection.NEUTRAL, 0.50

        return EMAResult(
            ema20=round(ema20, 4),
            ema50=round(ema50, 4),
            ema200=round(ema200, 4),
            current_price=current_price,
            alignment=alignment,
            signal=signal,
            signal_strength=round(strength, 3),
            golden_cross=golden,
            death_cross=death,
        )

    # ── ATR Yorumlama ─────────────────────────────────────────

    def _interpret_atr(
        self, df: pd.DataFrame, current_price: float
    ) -> Optional[ATRResult]:
        """ATR değerini hesaplar; stop-loss ve take-profit seviyeleri üretir.

        Args:
            df: enrich_dataframe() çıktısı.
            current_price: Son kapanış fiyatı.

        Returns:
            ATRResult nesnesi veya None.
        """
        if "atr" not in df.columns or df["atr"].isna().all():
            return None

        atr_val = float(df["atr"].iloc[-1])
        if np.isnan(atr_val) or current_price == 0:
            return None

        atr_pct = (atr_val / current_price) * 100

        # Stop-loss seviyeleri
        sl_long  = current_price - (atr_val * self.atr_multiplier)
        sl_short = current_price + (atr_val * self.atr_multiplier)

        # Take-profit seviyeleri (Risk × RR oranı)
        risk_amount = atr_val * self.atr_multiplier
        tp_long  = current_price + (risk_amount * self.rr_ratio)
        tp_short = current_price - (risk_amount * self.rr_ratio)

        # Volatilite etiketi
        if atr_pct < 1.0:
            vol_label = "Dusuk"
        elif atr_pct < 3.0:
            vol_label = "Normal"
        else:
            vol_label = "Yuksek"

        return ATRResult(
            value=round(atr_val, 4),
            current_price=current_price,
            atr_pct=round(atr_pct, 3),
            stop_loss_long=round(sl_long, 4),
            stop_loss_short=round(sl_short, 4),
            take_profit_long=round(tp_long, 4),
            take_profit_short=round(tp_short, 4),
            volatility_label=vol_label,
        )

    # ── Bollinger Yorumlama ───────────────────────────────────

    def _interpret_bollinger(
        self, df: pd.DataFrame, current_price: float
    ) -> Optional[BollingerResult]:
        """Bollinger Bantlarını hesaplar ve yorumlar.

        Args:
            df: enrich_dataframe() çıktısı.
            current_price: Son kapanış fiyatı.

        Returns:
            BollingerResult nesnesi veya None.
        """
        required = {"bb_upper", "bb_middle", "bb_lower", "bb_bandwidth", "bb_pct_b"}
        if not required.issubset(df.columns):
            return None

        last = df.iloc[-1]
        upper  = float(last["bb_upper"])
        middle = float(last["bb_middle"])
        lower  = float(last["bb_lower"])
        bw     = float(last["bb_bandwidth"])
        pct_b  = float(last["bb_pct_b"])

        if any(np.isnan(x) for x in [upper, middle, lower, bw, pct_b]):
            return None

        # Konum belirleme
        if current_price > upper:
            position = BBPosition.ABOVE_UPPER
        elif current_price > (middle + (upper - middle) * 0.7):
            position = BBPosition.NEAR_UPPER
        elif current_price < lower:
            position = BBPosition.BELOW_LOWER
        elif current_price < (middle - (middle - lower) * 0.7):
            position = BBPosition.NEAR_LOWER
        else:
            position = BBPosition.MIDDLE

        # Squeeze tespiti
        is_squeeze = bw < DEFAULT_BB_SQUEEZE_THR

        # Sinyal
        if position == BBPosition.BELOW_LOWER:
            signal, strength = SignalDirection.STRONG_BUY, 0.80
        elif position == BBPosition.NEAR_LOWER:
            signal, strength = SignalDirection.BUY, 0.60
        elif position == BBPosition.ABOVE_UPPER:
            signal, strength = SignalDirection.STRONG_SELL, 0.80
        elif position == BBPosition.NEAR_UPPER:
            signal, strength = SignalDirection.SELL, 0.60
        else:
            strength = 0.50
            signal   = SignalDirection.NEUTRAL

        # Squeeze durumunda güç sıfıra yaklaşır (yön bilinmiyor)
        if is_squeeze:
            strength = 0.40
            signal   = SignalDirection.NEUTRAL

        return BollingerResult(
            upper=round(upper, 4),
            middle=round(middle, 4),
            lower=round(lower, 4),
            current_price=current_price,
            bandwidth=round(bw, 6),
            percent_b=round(pct_b, 4),
            position=position,
            is_squeeze=is_squeeze,
            signal=signal,
            signal_strength=round(strength, 3),
        )

    # ── Hacim Yorumlama ───────────────────────────────────────

    def _interpret_volume(self, df: pd.DataFrame) -> Optional[VolumeResult]:
        """Hacim analizini hesaplar ve yorumlar.

        Args:
            df: enrich_dataframe() çıktısı.

        Returns:
            VolumeResult nesnesi veya None.
        """
        if "volume_ratio" not in df.columns:
            return None

        last = df.iloc[-1]
        cur_vol = float(last["volume"])
        avg_vol = float(last.get("volume_avg", np.nan))
        ratio   = float(last["volume_ratio"])

        if np.isnan(ratio) or np.isnan(avg_vol):
            return None

        is_above = ratio >= 1.0

        if ratio >= 2.0:
            signal, strength = SignalDirection.STRONG_BUY, 0.90   # Fiyat yönüne göre güncellenir
        elif ratio >= 1.5:
            signal, strength = SignalDirection.BUY, 0.70
        elif ratio >= 0.8:
            signal, strength = SignalDirection.NEUTRAL, 0.50
        else:
            signal, strength = SignalDirection.SELL, 0.30          # Düşük hacim = güvensiz hareket

        # Fiyat yönüyle uyumu kontrol et
        price_up = float(df["close"].iloc[-1]) > float(df["close"].iloc[-2])
        if not price_up and signal in (SignalDirection.BUY, SignalDirection.STRONG_BUY):
            # Hacim yüksek ama fiyat düşüyor → Güçlü satış baskısı
            signal  = SignalDirection.SELL if ratio >= 2.0 else SignalDirection.NEUTRAL
            strength = strength * 0.8

        return VolumeResult(
            current_volume=round(cur_vol, 2),
            avg_volume=round(avg_vol, 2),
            volume_ratio=round(ratio, 3),
            is_above_average=is_above,
            signal=signal,
            signal_strength=round(strength, 3),
        )

    # ── ADX Yorumlama ─────────────────────────────────────────

    def _interpret_adx(self, df: pd.DataFrame) -> Optional[ADXResult]:
        """ADX değerini hesaplar ve yorumlar.

        Args:
            df: enrich_dataframe() çıktısı (adx, adx_pos, adx_neg sütunları mevcut).

        Returns:
            ADXResult nesnesi veya None.
        """
        required = {"adx", "adx_pos", "adx_neg"}
        if not required.issubset(df.columns) or df["adx"].isna().all():
            return None

        last = df.iloc[-1]
        adx_val = float(last["adx"])
        di_plus = float(last["adx_pos"])
        di_minus = float(last["adx_neg"])

        if any(np.isnan(x) for x in [adx_val, di_plus, di_minus]):
            return None

        # Yönlü sinyal (DI+ > DI- -> BUY, DI- > DI+ -> SELL)
        # Ancak ADX < adx_threshold ise range piyasası, nötr sinyal üretir
        if adx_val < self.adx_threshold:
            signal = SignalDirection.NEUTRAL
            strength = 0.40 # yönsüz
        else:
            if di_plus > di_minus:
                signal = SignalDirection.BUY
            else:
                signal = SignalDirection.SELL
            
            # ADX gücü: ADX ne kadar yüksekse trend o kadar güçlüdür
            if adx_val >= 40:
                strength = 0.85
            elif adx_val >= 25:
                strength = 0.70
            else:
                strength = 0.50

        return ADXResult(
            value=round(adx_val, 2),
            di_plus=round(di_plus, 2),
            di_minus=round(di_minus, 2),
            signal=signal,
            signal_strength=round(strength, 3),
        )

    def _interpret_fibonacci(self, df: pd.DataFrame) -> Optional[FibonacciResult]:
        """Son 500 muma göre Fibonacci düzeltme seviyelerini hesaplar."""
        lookback = min(500, len(df))
        if lookback < 10:
            return None

        recent = df.tail(lookback)
        swing_high = float(recent["high"].max())
        swing_low = float(recent["low"].min())

        if swing_high == swing_low:
            return None

        diff = swing_high - swing_low

        # Fibonacci düzeltme seviyelerini hesapla
        # %23.6, %38.2, %50.0, %61.8, %78.6
        fib_236 = swing_high - 0.236 * diff
        fib_382 = swing_high - 0.382 * diff
        fib_500 = swing_high - 0.500 * diff
        fib_618 = swing_high - 0.618 * diff
        fib_786 = swing_high - 0.786 * diff

        return FibonacciResult(
            swing_high=round(swing_high, 4),
            swing_low=round(swing_low, 4),
            fib_236=round(fib_236, 4),
            fib_382=round(fib_382, 4),
            fib_500=round(fib_500, 4),
            fib_618=round(fib_618, 4),
            fib_786=round(fib_786, 4),
        )

    def _detect_patterns(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """Mum ve grafik formasyonlarını tespit eder."""
        if len(df) < 20:
            return None

        # ── 1. Mum Formasyonları ──
        hammer = False
        shooting_star = False
        bullish_engulfing = False
        bearish_engulfing = False

        # Son mum ve önceki mum değerleri
        open_0 = float(df['open'].iloc[-1])
        high_0 = float(df['high'].iloc[-1])
        low_0 = float(df['low'].iloc[-1])
        close_0 = float(df['close'].iloc[-1])
        body_0 = abs(close_0 - open_0)
        range_0 = high_0 - low_0

        open_1 = float(df['open'].iloc[-2])
        close_1 = float(df['close'].iloc[-2])
        body_1 = abs(close_1 - open_1)

        is_bullish_0 = close_0 >= open_0
        is_bearish_0 = close_0 < open_0
        is_bearish_1 = close_1 < open_1
        is_bullish_1 = close_1 >= open_1

        # Ortalama mum gövde boyutu (son 10 mum)
        avg_body_10 = df['close'].sub(df['open']).abs().tail(10).mean()

        if range_0 > 0:
            upper_shadow_0 = high_0 - max(open_0, close_0)
            lower_shadow_0 = min(open_0, close_0) - low_0

            # Trend yönü teyidi için kısa dönem ortalama
            short_term_avg = df['close'].iloc[-10:-1].mean()

            # Hammer
            if lower_shadow_0 >= 2 * body_0 and upper_shadow_0 <= 0.10 * range_0 and body_0 > 0:
                if close_0 < short_term_avg:
                    hammer = True

            # Shooting Star
            if upper_shadow_0 >= 2 * body_0 and lower_shadow_0 <= 0.10 * range_0 and body_0 > 0:
                if close_0 > short_term_avg:
                    shooting_star = True

        # Bullish Engulfing
        if is_bearish_1 and is_bullish_0:
            if open_0 <= close_1 and close_0 >= open_1 and (open_0 < close_1 or close_0 > open_1):
                if body_0 >= avg_body_10 * 0.75:
                    bullish_engulfing = True

        # Bearish Engulfing
        if is_bullish_1 and is_bearish_0:
            if open_0 >= close_1 and close_0 <= open_1 and (open_0 > close_1 or close_0 < open_1):
                if body_0 >= avg_body_10 * 0.75:
                    bearish_engulfing = True

        # ── 2. Grafik Formasyonları ──
        double_bottom = False
        double_top = False

        lookback = min(150, len(df))
        recent = df.tail(lookback)
        
        lows = recent["low"].values
        highs = recent["high"].values
        
        swing_lows = []
        swing_highs = []
        
        k = 5 # window size
        for i in range(k, len(recent) - k):
            current_low = lows[i]
            window_lows = lows[i-k : i+k+1]
            if current_low == min(window_lows) and current_low < max(window_lows):
                swing_lows.append((i, current_low))
                
            current_high = highs[i]
            window_highs = highs[i-k : i+k+1]
            if current_high == max(window_highs) and current_high > min(window_highs):
                swing_highs.append((i, current_high))

        current_price = float(df["close"].iloc[-1])

        # Double Bottom
        if len(swing_lows) >= 2:
            idx1, val1 = swing_lows[-2]
            idx2, val2 = swing_lows[-1]
            if (idx2 - idx1) >= 15:
                price_diff = abs(val1 - val2) / max(val1, val2)
                if price_diff <= 0.015:
                    avg_dip = (val1 + val2) / 2.0
                    if avg_dip < current_price <= avg_dip * 1.04:
                        double_bottom = True

        # Double Top
        if len(swing_highs) >= 2:
            idx1, val1 = swing_highs[-2]
            idx2, val2 = swing_highs[-1]
            if (idx2 - idx1) >= 15:
                price_diff = abs(val1 - val2) / max(val1, val2)
                if price_diff <= 0.015:
                    avg_peak = (val1 + val2) / 2.0
                    if avg_peak * 0.96 <= current_price < avg_peak:
                        double_top = True

        # Aktif açıklamaları topla
        active = []
        if hammer: active.append("Çekiç (Hammer)")
        if shooting_star: active.append("Kayan Yıldız (Shooting Star)")
        if bullish_engulfing: active.append("Yutan Boğa (Bullish Engulfing)")
        if bearish_engulfing: active.append("Yutan Ayı (Bearish Engulfing)")
        if double_bottom: active.append("İkili Dip (Double Bottom)")
        if double_top: active.append("İkili Tepe (Double Top)")

        return PatternResult(
            hammer=hammer,
            shooting_star=shooting_star,
            bullish_engulfing=bullish_engulfing,
            bearish_engulfing=bearish_engulfing,
            double_bottom=double_bottom,
            double_top=double_top,
            active_patterns=active
        )

    # ── Yardımcı Metodlar ─────────────────────────────────────

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """DataFrame'in yeterli veri içerip içermediğini kontrol eder.

        Args:
            df: Kontrol edilecek DataFrame.

        Returns:
            True: DataFrame geçerli ve yeterli veri var.
            False: Yetersiz veya hatalı veri.
        """
        if df is None or df.empty:
            logger.warning("TechnicalEngine: Boş DataFrame.")
            return False

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            logger.error(f"TechnicalEngine: Eksik sütunlar: {missing}")
            return False

        if len(df) < self.min_bars:
            logger.warning(
                f"TechnicalEngine: Yetersiz bar sayısı. "
                f"Mevcut: {len(df)}, Gerekli: {self.min_bars}"
            )
            return False

        return True
