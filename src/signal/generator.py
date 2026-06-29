# ============================================================
# src/signal/generator.py — Trading Bot Trading Bot
#
# AMAÇ:
#   TechnicalEngine'den gelen IndicatorSet'i değerlendirerek
#   alım/satım/bekleme sinyali üretir.
#   Her karar feature_weights.py'deki ENTRY_THRESHOLD ile
#   karşılaştırılır ve insan-okunur gerekçeyle desteklenir.
#
# KARAR MANTIĞI (Sıralı Filtre Zinciri):
#   1. Ağırlıklı skor eşiği kontrolü  (>= 0.65)
#   2. Minimum hacim filtresi          (volume_ratio >= 0.8)
#   3. ATR tabanlı Risk/Ödül kontrolü  (>= 2.0)
#   4. Gerekçe üretimi
#   5. TradeSignal nesnesi inşa et
#
# PAPER TRADE MODU:
#   is_paper_trade=True iken hiçbir gerçek emir gönderilmez.
#   Sinyaller yalnızca loglanır ve SignalJournal'a kaydedilir.
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v1.0 | İlk SignalGenerator implementasyonu
# ============================================================

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.signal.models import (
    SignalEvaluation,
    SignalRejectionReason,
    SignalStrength,
    SignalType,
    TradeSignal,
)
from src.strategy.feature_weights import SCORING_CONFIG
from src.technical.indicators import (
    IndicatorSet,
    RSIZone,
    SignalDirection,
    EMAAlignment,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalGenerator:
    """IndicatorSet'i değerlendirerek TradeSignal üretir.

    Mimarideki rolü:
      DataCollector → TechnicalEngine → **SignalGenerator** → ExecutionEngine

    Attributes:
        entry_threshold: Alış sinyali için minimum ağırlıklı skor.
        exit_threshold: Çıkış sinyali için maksimum ağırlıklı skor.
        min_rr_ratio: Minimum Risk/Ödül oranı.
        min_volume_ratio: Minimum hacim oranı (anlık/ortalama).
        is_paper_trade: True = kağıt işlem modu (emir yok).
        require_trend_alignment: True = EMA trendi sinyalle uyumlu olmalı.

    Example:
        >>> engine  = TechnicalEngine()
        >>> gen     = SignalGenerator(is_paper_trade=True)
        >>> ind_set = engine.get_latest_indicators(df, "BTC/USDT", "4h")
        >>> signal  = gen.evaluate(ind_set)
        >>> print(signal.print_card())
    """

    def __init__(
        self,
        entry_threshold: Optional[float] = None,
        exit_threshold: Optional[float] = None,
        min_rr_ratio: float = 2.0,
        min_volume_ratio: float = 0.8,
        adx_threshold: float = 20.0,
        is_paper_trade: bool = True,
        require_trend_alignment: bool = True,
    ) -> None:
        """SignalGenerator başlatır.

        Args:
            entry_threshold: Alış eşiği (None = config'den alınır).
            exit_threshold: Çıkış eşiği (None = config'den alınır).
            min_rr_ratio: Min Risk/Ödül oranı (varsayılan: 2.0).
            min_volume_ratio: Min hacim oranı (varsayılan: 0.8).
            adx_threshold: ADX trend gücü eşiği (varsayılan: 20.0).
            is_paper_trade: Kağıt işlem modu (varsayılan: True).
            require_trend_alignment: EMA trend uyumu zorunlu mu?
        """
        self.entry_threshold = entry_threshold or SCORING_CONFIG["ENTRY_THRESHOLD"]
        self.exit_threshold  = exit_threshold  or SCORING_CONFIG["EXIT_THRESHOLD"]
        self.min_rr_ratio    = min_rr_ratio
        self.min_volume_ratio = min_volume_ratio
        self.adx_threshold   = adx_threshold
        self.is_paper_trade  = is_paper_trade
        self.require_trend_alignment = require_trend_alignment

        logger.info(
            f"SignalGenerator başlatıldı | "
            f"Giriş eşiği: {self.entry_threshold} | "
            f"Çıkış eşiği: {self.exit_threshold} | "
            f"Min RR: {self.min_rr_ratio} | "
            f"Mod: {'PAPER TRADE' if is_paper_trade else 'CANLI'}"
        )

    # ── Ana Değerlendirme Metodu ──────────────────────────────

    def evaluate(
        self,
        indicator_set: Optional[IndicatorSet],
        btc_trend_bearish: bool = False,
        ticker_info: Optional[dict] = None,
    ) -> TradeSignal:
        """IndicatorSet'i değerlendirerek TradeSignal üretir.

        Sinyalin tüm yaşam döngüsünü yönetir:
          1. Girdi doğrulama
          2. Skor eşiği kontrolü
          3. Filtre zinciri
          4. Sinyal yönü belirleme
          5. TradeSignal inşa et + logla

        Args:
            indicator_set: TechnicalEngine'den gelen indikatör seti.

        Returns:
            TradeSignal nesnesi (NO_SIGNAL dahil her zaman döner).
        """
        t_start = time.monotonic()

        # ── 1. Boş girdi kontrolü ─────────────────────────────
        if indicator_set is None:
            logger.warning("SignalGenerator: IndicatorSet None — NO_SIGNAL")
            return self._build_no_signal(
                symbol="UNKNOWN", timeframe="UNKNOWN",
                price=0.0, score=0.0,
                rejection=[SignalRejectionReason.MISSING_INDICATORS.value],
            )

        sym   = indicator_set.symbol
        tf    = indicator_set.timeframe
        price = indicator_set.current_price
        score = indicator_set.weighted_score

        logger.info(
            f"[{sym}@{tf}] Sinyal değerlendirmesi | "
            f"Skor: {score:.3f} | Eşik: {self.entry_threshold}"
        )

        # ── YENİ: V3 Price Action Strateji Değerlendirmesi ──
        from src.config.settings import get_settings
        try:
            cfg = get_settings()
            strategy_version = cfg.strategy.version
        except Exception:
            strategy_version = self.settings.strategy.version if hasattr(self, 'settings') else "v2"
        if hasattr(self, '_strategy_version'):
            strategy_version = self._strategy_version

        if strategy_version in ("v3", "v4", "v5"):
            pa_sig = getattr(indicator_set, "pa_signal", "HOLD")
            if pa_sig in ("BUY", "SELL"):
                from src.signal.models import SignalType as ST, SignalStrength
                from src.signal.models import TradeSignal
                
                sig_type = ST.BUY if pa_sig == "BUY" else ST.SELL
                entry = getattr(indicator_set, "pa_entry_price", price)
                sl = getattr(indicator_set, "pa_sl_price", 0.0)
                tp = getattr(indicator_set, "pa_tp_price", 0.0)
                has_fvg = getattr(indicator_set, "pa_has_fvg", False)
                
                fvg_text = " FVG Mid-Level Limit" if has_fvg else " Market Close"
                ver_label = strategy_version.upper()
                reasons = [f"{ver_label} Price Action: {pa_sig} tetiklendi ({fvg_text})"]
                
                risk = abs(entry - sl) if sl > 0 else 0
                rr = abs(tp - entry) / risk if risk > 0 else 0
                
                signal = TradeSignal(
                    symbol=sym,
                    timeframe=tf,
                    signal_type=sig_type,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    risk_reward_ratio=rr,
                    weighted_score=1.0 if pa_sig == "BUY" else 0.0,
                    signal_strength=SignalStrength.GUCLU,
                    confidence=0.90,
                    is_paper_trade=self.is_paper_trade,
                    reasons=tuple(reasons),
                    rejection_reasons=(),
                    indicator_summary={
                        "pa_signal": pa_sig,
                        "pa_entry": entry,
                        "pa_sl": sl,
                        "pa_tp": tp,
                        "has_fvg": has_fvg
                    }
                )
                logger.info(f"🏆 [{ver_label} PA Sinyali] {sym}@{tf} → Yön: {pa_sig} | Entry: {entry:.4f} | SL: {sl:.4f} | TP: {tp:.4f}")
                return signal
            else:
                return self._build_hold_signal(sym, tf, price, score)

        # ── 2. ATR kontrolü (stop-loss için zorunlu) ──────────
        if indicator_set.atr is None:
            logger.warning(f"[{sym}@{tf}] ATR eksik — NO_SIGNAL")
            return self._build_no_signal(
                sym, tf, price, score,
                rejection=[SignalRejectionReason.MISSING_INDICATORS.value],
            )

        # ── 3. Skor eşiği kontrolü ────────────────────────────
        rejections: List[str] = []

        if score >= self.entry_threshold:
            signal_type = SignalType.BUY

        elif score <= self.exit_threshold:
            signal_type = SignalType.SELL

        else:
            # Eşik arasında: HOLD
            logger.info(
                f"[{sym}@{tf}] Eşik karşılanmadı "
                f"({self.exit_threshold:.2f} < {score:.3f} < {self.entry_threshold:.2f}) → HOLD"
            )
            return self._build_hold_signal(sym, tf, price, score)

        # ── 4. Filtre zinciri ─────────────────────────────────
        passed, filter_rejects, reasons = self._run_filters(
            indicator_set, signal_type, btc_trend_bearish, ticker_info
        )
        rejections.extend(filter_rejects)

        if not passed:
            logger.info(
                f"[{sym}@{tf}] Filtre(ler) reddetti: {filter_rejects} → NO_SIGNAL"
            )
            return self._build_no_signal(sym, tf, price, score, rejections)

        # ── 5. TradeSignal inşa et ────────────────────────────
        signal = self._build_trade_signal(
            signal_type=signal_type,
            indicator_set=indicator_set,
            reasons=reasons,
            rejections=rejections,
        )

        elapsed_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            f"[{sym}@{tf}] Sinyal üretildi: {signal_type.value} | "
            f"{elapsed_ms:.1f}ms | {signal.to_log_line()}"
        )

        # Paper trade modunda sinyal kartını logla
        if self.is_paper_trade:
            logger.info(f"\n{signal.print_card()}")

        return signal

    def evaluate_batch(
        self,
        indicator_sets: Dict[str, IndicatorSet],
    ) -> Dict[str, TradeSignal]:
        """Birden fazla timeframe / sembol için toplu sinyal üretir.

        Args:
            indicator_sets: {timeframe: IndicatorSet} sözlüğü.

        Returns:
            {timeframe: TradeSignal} sözlüğü.
        """
        results: Dict[str, TradeSignal] = {}
        for key, ind_set in indicator_sets.items():
            results[key] = self.evaluate(ind_set)
        return results

    # ── Filtre Zinciri ────────────────────────────────────────

    def _run_filters(
        self,
        ind: IndicatorSet,
        signal_type: SignalType,
        btc_trend_bearish: bool = False,
        ticker_info: Optional[dict] = None,
    ) -> Tuple[bool, List[str], List[str]]:
        """Tüm filtreleri sırayla çalıştırır.

        Bir filtre başarısız olursa zincir durur (short-circuit).

        Args:
            ind: IndicatorSet.
            signal_type: BUY veya SELL.

        Returns:
            Tuple[tüm_filtreler_geçildi, red_nedenleri, gerekçeler]
        """
        rejections: List[str] = []
        reasons:    List[str] = []

        # Filtre 1: Hacim
        vol_ok, vol_reason, vol_rej = self._filter_volume(ind)
        if vol_reason:
            reasons.append(vol_reason)
        if not vol_ok:
            rejections.append(vol_rej or SignalRejectionReason.INSUFFICIENT_VOLUME.value)
            return False, rejections, reasons

        # Filtre 2: Risk/Ödül oranı
        rr_ok, rr_reason, rr_rej = self._filter_risk_reward(ind, signal_type)
        if rr_reason:
            reasons.append(rr_reason)
        if not rr_ok:
            rejections.append(rr_rej or SignalRejectionReason.POOR_RISK_REWARD.value)
            return False, rejections, reasons

        # Filtre 3: EMA trend uyumu (opsiyonel)
        if self.require_trend_alignment:
            trend_ok, trend_reason, trend_rej = self._filter_trend_alignment(
                ind, signal_type
            )
            if trend_reason:
                reasons.append(trend_reason)
            if not trend_ok:
                rejections.append(trend_rej or SignalRejectionReason.CONTRADICTING_TREND.value)
                return False, rejections, reasons

        # ── STRATEJi V2 / V2.1 FiLTRELERi ──
        from src.config.settings import get_settings
        try:
            cfg = get_settings()
            strategy_version = cfg.strategy.version
        except Exception:
            strategy_version = self.settings.strategy.version if hasattr(self, 'settings') else "v2"
        
        # settings üzerinde de bak (BotEngine tarafından inject edilmiş olabilir)
        if hasattr(self, '_strategy_version'):
            strategy_version = self._strategy_version
        
        if strategy_version in ("v2", "v2.1"):
            # Filtre 4: BTC Makro Trend Filtresi
            if strategy_version == "v2":
                # v2: sadece BUY bloke edilir
                if signal_type == SignalType.BUY and btc_trend_bearish:
                    rejections.append("F4: BTC_BEARISH_TREND_BLOCKED (BTC 4h Death Cross)")
                    return False, rejections, reasons
            elif strategy_version == "v2.1":
                # v2.1: BTC bearish → BUY bloke, BTC bullish → SHORT bloke
                if signal_type == SignalType.BUY and btc_trend_bearish:
                    rejections.append("F4: BTC_BEARISH_TREND_BLOCKED — v2.1 SHORT’a geç (BTC 4h Death Cross aktif)")
                    return False, rejections, reasons
                # Not: BTC bullish iken SHORT'u bloke etmiyoruz — teknik skor yeterliyse açılır

            # Filtre 5: Likidite ve Spread Filtresi
            if (signal_type == SignalType.BUY or (strategy_version == "v2.1" and signal_type == SignalType.SELL)) and ticker_info:
                quote_volume = ticker_info.get("quoteVolume") or ticker_info.get("info", {}).get("quoteVolume")
                bid = ticker_info.get("bid")
                ask = ticker_info.get("ask")
                
                # Koşul 1: Son 24s hacim kontrolü (< 500,000 USD ise engelle)
                if quote_volume is not None:
                    try:
                        q_vol = float(quote_volume)
                        if q_vol < 500000:
                            rejections.append(f"F5: {'SHORT_' if signal_type == SignalType.SELL else ''}INSUFFICIENT_DAILY_VOLUME (${q_vol:,.0f} < $500,000)")
                            return False, rejections, reasons
                    except Exception:
                        pass
                
                # Koşul 2: Spread kontrolü (> %1.5 ise engelle)
                if bid and ask:
                    try:
                        b_val = float(bid)
                        a_val = float(ask)
                        if b_val > 0:
                            spread_pct = (a_val - b_val) / b_val * 100
                            if spread_pct > 1.5:
                                rejections.append(f"F5: HIGH_SPREAD ({spread_pct:.2f}% > 1.5%)")
                                return False, rejections, reasons
                    except Exception:
                        pass

            # Filtre 6: ADX Range Filtresi
            if ind.adx is not None:
                if ind.adx.value < self.adx_threshold:
                    rejections.append(f"F6: ADX_RANGE_BLOCKED (ADX={ind.adx.value} < {self.adx_threshold})")
                    return False, rejections, reasons

        # Gerekçeleri zenginleştir
        reasons.extend(self._build_reasons(ind, signal_type))

        return True, rejections, reasons

    def _filter_volume(
        self, ind: IndicatorSet
    ) -> Tuple[bool, str, str]:
        """Hacim filtresi: Minimum oran sağlanmalı."""
        if ind.volume is None:
            return True, "", ""   # Hacim verisi yoksa filtreyi geç

        ratio = ind.volume.volume_ratio
        if ratio < self.min_volume_ratio:
            return (
                False,
                "",
                f"{SignalRejectionReason.INSUFFICIENT_VOLUME.value} "
                f"(Oran: {ratio:.2f} < Min: {self.min_volume_ratio})",
            )
        label = "Yüksek" if ratio >= 1.5 else "Normal"
        return (
            True,
            f"Hacim onayı: {ratio:.2f}x ortalama ({label})",
            "",
        )

    def _filter_risk_reward(
        self, ind: IndicatorSet, signal_type: SignalType
    ) -> Tuple[bool, str, str]:
        """Risk/Ödül filtresi: ATR tabanlı RR >= min_rr_ratio olmalı."""
        if ind.atr is None:
            return True, "", ""

        price   = ind.current_price
        atr_val = ind.atr.value
        risk    = atr_val * 2.0   # config'deki atr_multiplier

        if signal_type == SignalType.BUY:
            sl     = price - risk
            tp     = price + (risk * self.min_rr_ratio)
            rr_act = (tp - price) / risk if risk > 0 else 0
        else:
            sl     = price + risk
            tp     = price - (risk * self.min_rr_ratio)
            rr_act = (price - tp) / risk if risk > 0 else 0

        if rr_act < self.min_rr_ratio - 0.05:
            return (
                False,
                "",
                f"{SignalRejectionReason.POOR_RISK_REWARD.value} "
                f"(Gerçek RR: {rr_act:.2f} < Min: {self.min_rr_ratio})",
            )
        return (
            True,
            f"Risk/Ödül onayı: 1:{rr_act:.2f} (SL={sl:,.2f} / TP={tp:,.2f})",
            "",
        )

    def _filter_trend_alignment(
        self, ind: IndicatorSet, signal_type: SignalType
    ) -> Tuple[bool, str, str]:
        """EMA trend uyumu filtresi.

        BUY için: EMA hizalaması boğa veya tarafsız olmalı.
        SELL için: EMA hizalaması ayı veya tarafsız olmalı.
        """
        if ind.ema is None:
            return True, "", ""

        alignment = ind.ema.alignment

        if signal_type == SignalType.BUY:
            if alignment == EMAAlignment.FULL_BEARISH:
                return (
                    False,
                    "",
                    f"{SignalRejectionReason.CONTRADICTING_TREND.value}: "
                    f"EMA tam ayı dizilimi ({alignment.value}) — BUY sinyaline karşı",
                )
            return (
                True,
                f"EMA trend uyumu: {alignment.value} (BUY uyumlu)",
                "",
            )

        else:  # SELL
            if alignment == EMAAlignment.FULL_BULLISH:
                return (
                    False,
                    "",
                    f"{SignalRejectionReason.CONTRADICTING_TREND.value}: "
                    f"EMA tam boğa dizilimi ({alignment.value}) — SELL sinyaline karşı",
                )
            return (
                True,
                f"EMA trend uyumu: {alignment.value} (SELL uyumlu)",
                "",
            )

    # ── Gerekçe Üretimi ───────────────────────────────────────

    def _build_reasons(
        self, ind: IndicatorSet, signal_type: SignalType
    ) -> List[str]:
        """Her indikatörden gelen gerekçe cümlelerini üretir."""
        reasons: List[str] = []
        is_buy = signal_type == SignalType.BUY

        # RSI gerekçesi
        if ind.rsi:
            if ind.rsi.has_divergence:
                reasons.append(
                    f"RSI Divergence tespit edildi: {ind.rsi.divergence_note} "
                    f"(Ağırlık yükseltildi → {ind.rsi.effective_weight})"
                )
            else:
                reasons.append(
                    f"RSI: {ind.rsi.value:.1f} ({ind.rsi.zone.value}) → "
                    f"{ind.rsi.signal.value}"
                )

        # MACD gerekçesi
        if ind.macd:
            cross_note = ""
            if ind.macd.cross_type.value != "no_cross":
                cross_note = f" [{ind.macd.cross_type.value.upper()}]"
            reasons.append(
                f"MACD: {ind.macd.macd_line:.4f} / Sinyal: {ind.macd.signal_line:.4f}"
                f" / Hist: {ind.macd.histogram:.4f}{cross_note}"
            )

        # EMA gerekçesi
        if ind.ema:
            if ind.ema.golden_cross:
                reasons.append("EMA Golden Cross: EMA50 EMA200'ü yukarı kesti (güçlü boğa)")
            elif ind.ema.death_cross:
                reasons.append("EMA Death Cross: EMA50 EMA200'ü aşağı kesti (güçlü ayı)")
            else:
                reasons.append(
                    f"EMA Dizilimi: {ind.ema.alignment.value} | "
                    f"{ind.ema.price_vs_ema200}"
                )

        # Bollinger gerekçesi
        if ind.bollinger:
            if ind.bollinger.is_squeeze:
                reasons.append(
                    "Bollinger Squeeze: Bantlar daralıyor — büyük hareket bekleniyor"
                )
            else:
                reasons.append(
                    f"Bollinger %B: {ind.bollinger.percent_b:.2f} "
                    f"| Konum: {ind.bollinger.position.value}"
                )

        # ATR gerekçesi
        if ind.atr:
            reasons.append(
                f"ATR ({ind.atr.volatility_label} volatilite): {ind.atr.value:.2f} "
                f"(%{ind.atr.atr_pct:.2f}) — "
                f"SL: {ind.atr.stop_loss_long if is_buy else ind.atr.stop_loss_short:,.2f}"
            )

        return reasons

    # ── TradeSignal İnşa Metodları ────────────────────────────

    def _build_trade_signal(
        self,
        signal_type: SignalType,
        indicator_set: IndicatorSet,
        reasons: List[str],
        rejections: List[str],
    ) -> TradeSignal:
        """Filtreleri geçmiş sinyal için TradeSignal oluşturur."""
        price  = indicator_set.current_price
        score  = indicator_set.weighted_score
        is_buy = signal_type == SignalType.BUY

        # Stop-loss ve take-profit (ATR'den ve Fibonacci hizalamasından)
        if indicator_set.atr:
            sl = indicator_set.atr.stop_loss_long if is_buy else indicator_set.atr.stop_loss_short
            tp = indicator_set.atr.take_profit_long if is_buy else indicator_set.atr.take_profit_short
            
            # Fibonacci hizalama
            if indicator_set.fib:
                orig_sl, orig_tp = sl, tp
                if is_buy:
                    # TP Hizalama: Fiyatın üzerindeki dirençler
                    resistances = [
                        indicator_set.fib.fib_786, indicator_set.fib.fib_618,
                        indicator_set.fib.fib_500, indicator_set.fib.fib_382,
                        indicator_set.fib.fib_236, indicator_set.fib.swing_high
                    ]
                    above_entry = [r for r in resistances if r > price]
                    if above_entry:
                        closest_r = min(above_entry, key=lambda r: abs(tp - r))
                        if abs(tp - closest_r) / closest_r <= 0.05:
                            tp = closest_r * 0.995  # %0.5 altına hizala
                            
                    # SL Hizalama: Fiyatın altındaki destekler
                    supports = [
                        indicator_set.fib.swing_low, indicator_set.fib.fib_786,
                        indicator_set.fib.fib_618, indicator_set.fib.fib_500,
                        indicator_set.fib.fib_382, indicator_set.fib.fib_236
                    ]
                    below_entry = [s for s in supports if s < price]
                    if below_entry:
                        closest_s = min(below_entry, key=lambda s: abs(sl - s))
                        if abs(sl - closest_s) / closest_s <= 0.05:
                            sl = closest_s * 0.993  # %0.7 altına hizala
                else:
                    # SHORT TP Hizalama: Fiyatın altındaki destekler
                    supports = [
                        indicator_set.fib.swing_low, indicator_set.fib.fib_786,
                        indicator_set.fib.fib_618, indicator_set.fib.fib_500,
                        indicator_set.fib.fib_382, indicator_set.fib.fib_236
                    ]
                    below_entry = [s for s in supports if s < price]
                    if below_entry:
                        closest_s = min(below_entry, key=lambda s: abs(tp - s))
                        if abs(tp - closest_s) / closest_s <= 0.05:
                            tp = closest_s * 1.005  # %0.5 üstüne hizala
                            
                    # SHORT SL Hizalama: Fiyatın üzerindeki dirençler
                    resistances = [
                        indicator_set.fib.fib_786, indicator_set.fib.fib_618,
                        indicator_set.fib.fib_500, indicator_set.fib.fib_382,
                        indicator_set.fib.fib_236, indicator_set.fib.swing_high
                    ]
                    above_entry = [r for r in resistances if r > price]
                    if above_entry:
                        closest_r = min(above_entry, key=lambda r: abs(sl - r))
                        if abs(sl - closest_r) / closest_r <= 0.05:
                            sl = closest_r * 1.007  # %0.7 üstüne hizala
                            
                # Risk ve ödül değerlerini hesapla
                risk   = abs(price - sl)
                reward = abs(tp - price)
                rr     = round(reward / risk, 2) if risk > 0 else 0
                
                # Minimum R/R kontrolü
                if rr < self.min_rr_ratio:
                    # Risk-ödül ihlal edilirse orijinal seviyelere geri dön
                    sl, tp = orig_sl, orig_tp
                    risk   = abs(price - sl)
                    reward = abs(tp - price)
                    rr     = round(reward / risk, 2) if risk > 0 else self.min_rr_ratio
                else:
                    if sl != orig_sl:
                        reasons.append(f"Zarar Kes (SL) Fibonacci seviyesine göre hizalandı: ${sl:,.4f}")
                    if tp != orig_tp:
                        reasons.append(f"Kâr Al (TP) Fibonacci seviyesine göre hizalandı: ${tp:,.4f}")
            else:
                risk   = abs(price - sl)
                reward = abs(tp - price)
                rr     = round(reward / risk, 2) if risk > 0 else self.min_rr_ratio
        else:
            sl, tp, rr = price * 0.98, price * 1.04, 2.0

        # Güç etiketi
        if score >= 0.85:
            strength = SignalStrength.VERY_STRONG
        elif score >= 0.75:
            strength = SignalStrength.STRONG
        else:
            strength = SignalStrength.MODERATE

        # Güven skoru: ağırlıklı skor + indikatör sayısı uyumu
        active_indicators = sum(
            1 for x in [
                indicator_set.rsi, indicator_set.macd,
                indicator_set.ema, indicator_set.bollinger,
                indicator_set.volume,
            ] if x is not None
        )
        confidence = round(min(1.0, score * (active_indicators / 5)), 3)

        # Fibonacci Golden Pocket Bounce Kontrolü
        if is_buy and indicator_set.fib:
            for lvl_name, lvl_val in [("0.50", indicator_set.fib.fib_500), 
                                      ("0.618", indicator_set.fib.fib_618), 
                                      ("0.786", indicator_set.fib.fib_786)]:
                if abs(price - lvl_val) / lvl_val <= 0.015:
                    reasons.append(f"Fibonacci Altın Cephe (Golden Pocket) Desteği: Fiyat {lvl_name} seviyesine (%1.5 toleransla) çok yakın (${lvl_val:,.4f})")
                    confidence = round(min(1.0, confidence + 0.10), 3)
                    break

        # Formasyon Onay Kontrolleri ve Güven Skoru Artırımı (Confidence Boost)
        if indicator_set.patterns and indicator_set.patterns.active_patterns:
            # BUY Sinyali İçin Boğa Formasyonu Onayları
            if is_buy:
                # İkili Dip Onayı
                if indicator_set.patterns.double_bottom:
                    reasons.append("Formasyon Onayı: İkili Dip (Double Bottom) grafik yapısı onaylandı.")
                    confidence = round(min(1.0, confidence + 0.15), 3)
                
                # Yutan Boğa Onayı
                if indicator_set.patterns.bullish_engulfing:
                    reasons.append("Formasyon Onayı: Yutan Boğa (Bullish Engulfing) mum formasyonu tespit edildi.")
                    confidence = round(min(1.0, confidence + 0.10), 3)
                
                # Çekiç Onayı
                if indicator_set.patterns.hammer:
                    reasons.append("Formasyon Onayı: Çekiç (Hammer) dönüş mumu tespit edildi.")
                    confidence = round(min(1.0, confidence + 0.10), 3)
                    
            # SELL/SHORT Sinyali İçin Ayı Formasyonu Onayları
            else:
                # İkili Tepe Onayı
                if indicator_set.patterns.double_top:
                    reasons.append("Formasyon Onayı: İkili Tepe (Double Top) grafik yapısı onaylandı.")
                    confidence = round(min(1.0, confidence + 0.15), 3)
                
                # Yutan Ayı Onayı
                if indicator_set.patterns.bearish_engulfing:
                    reasons.append("Formasyon Onayı: Yutan Ayı (Bearish Engulfing) mum formasyonu tespit edildi.")
                    confidence = round(min(1.0, confidence + 0.10), 3)
                
                # Kayan Yıldız Onayı
                if indicator_set.patterns.shooting_star:
                    reasons.append("Formasyon Onayı: Kayan Yıldız (Shooting Star) dönüş mumu tespit edildi.")
                    confidence = round(min(1.0, confidence + 0.10), 3)

        # İndikatör özeti
        summary: Dict = {}
        if indicator_set.rsi:
            summary["rsi"] = round(indicator_set.rsi.value, 2)
        if indicator_set.macd:
            summary["macd_hist"] = round(indicator_set.macd.histogram, 6)
        if indicator_set.ema:
            summary["ema_alignment"] = indicator_set.ema.alignment.value
        if indicator_set.atr:
            summary["atr_pct"] = round(indicator_set.atr.atr_pct, 3)
        if indicator_set.fib:
            summary["fib_high"] = round(indicator_set.fib.swing_high, 4)
            summary["fib_low"] = round(indicator_set.fib.swing_low, 4)
            summary["fib_236"] = round(indicator_set.fib.fib_236, 4)
            summary["fib_382"] = round(indicator_set.fib.fib_382, 4)
            summary["fib_500"] = round(indicator_set.fib.fib_500, 4)
            summary["fib_618"] = round(indicator_set.fib.fib_618, 4)
            summary["fib_786"] = round(indicator_set.fib.fib_786, 4)
        if indicator_set.patterns and indicator_set.patterns.active_patterns:
            summary["active_patterns"] = ", ".join(indicator_set.patterns.active_patterns)

        return TradeSignal(
            signal_type=signal_type,
            symbol=indicator_set.symbol,
            timeframe=indicator_set.timeframe,
            entry_price=round(price, 4),
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            risk_reward_ratio=rr,
            weighted_score=score,
            signal_strength=strength,
            confidence=confidence,
            is_paper_trade=self.is_paper_trade,
            reasons=tuple(reasons),
            rejection_reasons=tuple(rejections),
            generated_at=datetime.utcnow(),
            indicator_summary=summary,
        )

    def _build_no_signal(
        self,
        symbol: str,
        timeframe: str,
        price: float,
        score: float,
        rejection: Optional[List[str]] = None,
    ) -> TradeSignal:
        """İşlem yapılmayan NO_SIGNAL üretir."""
        return TradeSignal(
            signal_type=SignalType.NO_SIGNAL,
            symbol=symbol,
            timeframe=timeframe,
            entry_price=price,
            stop_loss=0.0,
            take_profit=0.0,
            risk_reward_ratio=0.0,
            weighted_score=score,
            signal_strength=SignalStrength.WEAK,
            confidence=0.0,
            is_paper_trade=self.is_paper_trade,
            reasons=(),
            rejection_reasons=tuple(rejection or []),
        )

    def _build_hold_signal(
        self,
        symbol: str,
        timeframe: str,
        price: float,
        score: float,
    ) -> TradeSignal:
        """HOLD sinyali — mevcut pozisyonu koru / bekle."""
        return TradeSignal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            timeframe=timeframe,
            entry_price=price,
            stop_loss=0.0,
            take_profit=0.0,
            risk_reward_ratio=0.0,
            weighted_score=score,
            signal_strength=SignalStrength.WEAK,
            confidence=round(score, 3),
            is_paper_trade=self.is_paper_trade,
            reasons=(f"Skor {score:.3f} — ne giriş ({self.entry_threshold}) "
                     f"ne çıkış ({self.exit_threshold}) eşiğini geçmedi",),
            rejection_reasons=(),
        )
