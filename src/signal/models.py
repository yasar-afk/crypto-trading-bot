# ============================================================
# src/signal/models.py — Trading Bot Trading Bot
#
# AMAÇ:
#   SignalGenerator'ın ürettiği tüm sinyal tiplerini ve
#   trade kararı nesnelerini tanımlar. ExecutionEngine (Faz 2)
#   bu nesneleri tüketerek emir gönderir.
#
# MİMARİ NOT:
#   TradeSignal immutable'dır (frozen=True). Bir sinyal üretildikten
#   sonra değiştirilemez; yeni koşullar yeni sinyal üretir.
#   Bu yaklaşım audit trail ve backtesting için kritiktir.
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v1.0 | İlk sinyal modelleri
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ─── Sinyal Tipleri ───────────────────────────────────────────

class SignalType(str, Enum):
    """Üretilen ticaret sinyalinin tipi."""
    BUY       = "BUY"    # Alış pozisyonu aç
    SELL      = "SELL"   # Satış / Short pozisyon aç (Faz 2)
    HOLD      = "HOLD"   # Mevcut pozisyonu koru / bekle
    EXIT_LONG = "EXIT_LONG"    # Long pozisyonu kapat
    EXIT_SHORT= "EXIT_SHORT"   # Short pozisyonu kapat (Faz 2)
    NO_SIGNAL = "NO_SIGNAL"    # Eşik karşılanmadı, işlem yok


class SignalStrength(str, Enum):
    """Sinyalin gücü — insan okunabilir etiket."""
    VERY_STRONG = "Çok Güçlü"   # Skor > 0.85
    STRONG      = "Güçlü"       # Skor > 0.75
    MODERATE    = "Orta"        # Skor > 0.65 (eşik civarı)
    WEAK        = "Zayıf"       # Eşik altı


class SignalRejectionReason(str, Enum):
    """Sinyal reddedilme nedeni."""
    SCORE_BELOW_THRESHOLD  = "Ağırlıklı skor eşiğin altında"
    INSUFFICIENT_VOLUME    = "Hacim yetersiz"
    CONTRADICTING_TREND    = "Üst TF trendi çelişiyor"
    POOR_RISK_REWARD       = "Risk/Ödül oranı yetersiz"
    MISSING_INDICATORS     = "Gerekli indikatörler eksik"
    DAILY_DRAWDOWN_LIMIT   = "Günlük zarar limiti aşıldı"
    MAX_POSITIONS_REACHED  = "Maksimum pozisyon sayısına ulaşıldı"


# ─── Ana Sinyal Modeli ────────────────────────────────────────

@dataclass(frozen=True)
class TradeSignal:
    """Tek bir ticaret sinyalini temsil eder.

    Immutable: Üretildikten sonra değiştirilemez.
    ExecutionEngine bu nesneyi alır ve işleme karar verir.

    Attributes:
        signal_type: BUY / SELL / HOLD / NO_SIGNAL.
        symbol: İşlem çifti (ör. 'BTC/USDT').
        timeframe: Sinyal üretilen zaman dilimi.
        entry_price: Önerilen giriş fiyatı.
        stop_loss: ATR tabanlı stop-loss seviyesi.
        take_profit: Minimum hedef fiyat (RR oranına göre).
        risk_reward_ratio: Gerçek Risk/Ödül oranı.
        weighted_score: IndicatorSet'ten gelen toplam skor.
        signal_strength: Sinyalin gücü etiketi.
        confidence: Güven seviyesi (0.0–1.0).
        reasons: Bu kararın gerekçeleri listesi.
        rejection_reasons: Sinyal reddedildiyse nedenler.
        is_paper_trade: True = sanal işlem, para kaybı yok.
        generated_at: Sinyalin üretildiği zaman.
        indicator_summary: İndikatör değerlerinin kısa özeti.
    """
    signal_type: SignalType
    symbol: str
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    weighted_score: float
    signal_strength: SignalStrength
    confidence: float
    is_paper_trade: bool = True
    reasons: tuple = field(default_factory=tuple)
    rejection_reasons: tuple = field(default_factory=tuple)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    indicator_summary: Dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """İşlem yapılabilir sinyal mi? (BUY veya SELL)"""
        return self.signal_type in (SignalType.BUY, SignalType.SELL)

    @property
    def is_exit(self) -> bool:
        """Çıkış sinyali mi?"""
        return self.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT)

    @property
    def risk_amount_pct(self) -> float:
        """Stop-loss'a olan mesafe yüzdesi."""
        if self.entry_price == 0:
            return 0.0
        return abs(self.entry_price - self.stop_loss) / self.entry_price * 100

    def to_log_line(self) -> str:
        """Tek satır log formatı."""
        status = "PAPER" if self.is_paper_trade else "CANLI"
        return (
            f"[{status}] {self.signal_type.value} | {self.symbol} @ {self.timeframe} | "
            f"Fiyat: {self.entry_price:,.2f} | SL: {self.stop_loss:,.2f} | "
            f"TP: {self.take_profit:,.2f} | RR: {self.risk_reward_ratio:.1f} | "
            f"Skor: {self.weighted_score:.3f} | Güç: {self.signal_strength.value}"
        )

    def to_dict(self) -> Dict:
        """JSON/CSV kaydı için dict formatı."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "weighted_score": self.weighted_score,
            "signal_strength": self.signal_strength.value,
            "confidence": self.confidence,
            "is_paper_trade": self.is_paper_trade,
            "reasons": list(self.reasons),
            "rejection_reasons": list(self.rejection_reasons),
        }

    def print_card(self) -> str:
        """Terminal'de görüntülenecek sinyal kartı."""
        separator = "=" * 60
        lines = [
            separator,
            f"  {self.signal_type.value} SİNYALİ — {'📄 PAPER TRADE' if self.is_paper_trade else '💰 CANLI'}",
            separator,
            f"  Sembol      : {self.symbol} @ {self.timeframe}",
            f"  Fiyat       : {self.entry_price:,.4f}",
            f"  Stop-Loss   : {self.stop_loss:,.4f}  (Risk: %{self.risk_amount_pct:.2f})",
            f"  Take-Profit : {self.take_profit:,.4f}",
            f"  R/R Oranı   : 1 : {self.risk_reward_ratio:.2f}",
            f"  Ağ. Skor    : {self.weighted_score:.3f}  ({self.signal_strength.value})",
            f"  Güven       : %{self.confidence*100:.1f}",
            f"  Üretildi    : {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "-" * 60,
            "  GEREKÇELER:",
        ]
        for r in self.reasons:
            lines.append(f"    ✓ {r}")
        if self.rejection_reasons:
            lines.append("  UYARILAR:")
            for r in self.rejection_reasons:
                lines.append(f"    ⚠ {r}")
        lines.append(separator)
        return "\n".join(lines)


@dataclass
class SignalEvaluation:
    """Bir sinyal üretim döngüsünün tüm ara sonuçlarını tutar.

    Debugging ve backtesting için: neden sinyal üretildi/üretilmedi.

    Attributes:
        symbol: İşlem çifti.
        timeframe: Zaman dilimi.
        weighted_score: Ham ağırlıklı skor.
        entry_threshold: Giriş eşiği.
        exit_threshold: Çıkış eşiği.
        passed_entry: Giriş eşiğini geçti mi?
        passed_filters: Tüm filtreleri geçti mi?
        filter_results: Her filtre için geçti/geçmedi.
        final_signal: Üretilen sinyal (None = üretilmedi).
        evaluation_ms: Değerlendirme süresi (ms).
    """
    symbol: str
    timeframe: str
    weighted_score: float
    entry_threshold: float
    exit_threshold: float
    passed_entry: bool = False
    passed_filters: bool = False
    filter_results: Dict[str, bool] = field(default_factory=dict)
    final_signal: Optional[TradeSignal] = None
    evaluation_ms: float = 0.0
