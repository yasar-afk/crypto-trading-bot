# ============================================================
# src/risk/engine.py — ANTIGRAVITI Trading Bot
#
# AMAÇ:
#   Alım-satım kararlarını risk parametrelerine göre denetler.
#   - Pozisyon boyutlandırma (Risk-budgeting & Kelly)
#   - Risk/Ödül oranı doğrulaması
#   - Günlük drawdown kontrolü
#   - Portföy genel sınır denetimleri
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v1.0 | İlk RiskEngine entegrasyonu
# ============================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from src.config.settings import Settings, get_settings
from src.signal.models import TradeSignal
from src.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskAssessment:
    """RiskEngine değerlendirme raporu.

    Attributes:
        is_approved: Sinyalin risk standartlarını karşılayıp karşılamadığı.
        position_size_usdt: Ayrılacak maksimum bakiye (USDT).
        risk_pct: İşlem başına göze alınan risk (portföy yüzdesi olarak).
        stop_loss_pct: Stop-loss noktasına olan uzaklık yüzdesi.
        rejection_reasons: Reddedilme gerekçeleri.
        calculated_at: Değerlendirme zamanı.
    """
    is_approved: bool
    position_size_usdt: float
    risk_pct: float
    stop_loss_pct: float
    rejection_reasons: Tuple[str, ...] = field(default_factory=tuple)
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "is_approved": self.is_approved,
            "position_size_usdt": self.position_size_usdt,
            "risk_pct": self.risk_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "rejection_reasons": list(self.rejection_reasons),
            "calculated_at": self.calculated_at.isoformat(),
        }


class RiskEngine:
    """İşlem riskini ve portföy güvenliğini denetleyen motor.

    Kuantum risk yönetimi ve dinamik pozisyon boyutlandırma yapar.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        initial_balance: float = 10000.0,
    ) -> None:
        """RiskEngine sınıfını başlatır.

        Args:
            settings: Sistem ayarları nesnesi. None ise varsayılan yüklenir.
            initial_balance: İlk bakiye (günlük drawdown hesapları için).
        """
        self.settings = settings or get_settings()
        self._initial_balance = initial_balance
        self._current_balance = initial_balance
        self._daily_losses: List[Tuple[datetime, float]] = []

        logger.info(
            f"RiskEngine başlatıldı | "
            f"Max Risk: %{self.settings.risk.max_position_pct * 100:.1f} | "
            f"Max Drawdown: %{self.settings.risk.max_daily_drawdown_pct * 100:.1f} | "
            f"Min RR: {self.settings.risk.min_risk_reward_ratio}"
        )

    def set_balance(self, current_balance: float) -> None:
        """Güncel portföy bakiyesini günceller.

        Args:
            current_balance: Güncel USDT bakiyesi.
        """
        self._current_balance = current_balance

    def record_loss(self, loss_usdt: float) -> None:
        """Drawdown kontrolü için gerçekleşen bir zararı kaydeder.

        Args:
            loss_usdt: Kaydedilecek zarar miktarı (pozitif sayı).
        """
        if loss_usdt > 0:
            self._daily_losses.append((datetime.utcnow(), loss_usdt))
            logger.info(f"Zarar kaydedildi: -${loss_usdt:,.2f}")

    def get_daily_drawdown_pct(self) -> float:
        """Son 24 saatteki toplam zarar yüzdesini hesaplar."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=24)
        
        # Son 24 saat dışındaki eski kayıpları temizle
        self._daily_losses = [
            (t, l) for t, l in self._daily_losses if t > cutoff
        ]
        
        total_lost = sum(l for t, l in self._daily_losses)
        if self._current_balance + total_lost == 0:
            return 0.0
        
        # Drawdown = toplam_zarar / (mevcut_bakiye + toplam_zarar)
        drawdown_pct = total_lost / (self._current_balance + total_lost)
        return drawdown_pct

    def assess(
        self,
        signal: TradeSignal,
        portfolio_balance: Optional[float] = None,
        use_kelly: bool = False,
    ) -> RiskAssessment:
        """Bir işlem sinyalinin riskini değerlendirir ve pozisyon boyutu önerir.

        Args:
            signal: Değerlendirilecek TradeSignal.
            portfolio_balance: Güncel bakiye (belirtilmezse iç bakiye kullanılır).
            use_kelly: True ise pozisyon boyutunu Kelly kriteri ile hesaplar.

        Returns:
            RiskAssessment raporu.
        """
        if portfolio_balance is not None:
            self.set_balance(portfolio_balance)

        balance = self._current_balance
        rejections: List[str] = []

        # 1. Sinyal tipi kontrolü (Sadece BUY/SELL değerlendirilir)
        if not signal.is_actionable:
            return RiskAssessment(
                is_approved=False,
                position_size_usdt=0.0,
                risk_pct=0.0,
                stop_loss_pct=0.0,
                rejection_reasons=("Sinyal işlemleşebilir (BUY/SELL) değil.",),
            )

        # 2. Risk/Ödül oranı doğrulaması
        min_rr = self.settings.risk.min_risk_reward_ratio
        if signal.risk_reward_ratio < min_rr - 0.05:
            rejections.append(
                f"Düşük Risk/Ödül Oranı: {signal.risk_reward_ratio:.2f} < Min: {min_rr}"
            )

        # 3. Günlük Drawdown kontrolü
        max_dd = self.settings.risk.max_daily_drawdown_pct
        current_dd = self.get_daily_drawdown_pct()
        if current_dd >= max_dd:
            rejections.append(
                f"Maksimum Günlük Drawdown aşıldı: %{current_dd*100:.2f} >= Limit: %{max_dd*100:.2f}. "
                f"Yeni işlemlere geçici olarak kilit vuruldu."
            )

        # 4. Stop-loss uzaklık yüzdesi hesabı
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        
        if entry_price <= 0 or stop_loss <= 0:
            rejections.append("Hatalı Fiyat Seviyeleri: Giriş fiyatı ve stop-loss pozitif olmalı.")
            sl_pct = 0.0
        else:
            sl_pct = abs(entry_price - stop_loss) / entry_price

        # 5. Pozisyon Boyutu Hesaplama (Risk-Budgeting)
        # Formül: Risk Tutarı = Bakiye * risk_pct
        # Pozisyon Boyutu = Risk Tutarı / Stop-Loss Yüzdesi
        from src.signal.models import SignalStrength
        if signal.signal_strength in (SignalStrength.VERY_STRONG, SignalStrength.STRONG):
            risk_pct = getattr(self.settings.risk, "high_risk_pct", 0.20)
        else:
            risk_pct = getattr(self.settings.risk, "normal_risk_pct", 0.10)
            
        risk_usdt = balance * risk_pct

        if sl_pct > 0:
            if use_kelly:
                # Dinamik Kelly Boyutu
                # p = sinyal güven skoru (confidence), q = 1 - p
                # b = risk reward oranı
                p = signal.confidence
                q = 1.0 - p
                b = signal.risk_reward_ratio if signal.risk_reward_ratio > 0 else 1.0
                
                # Kelly f = p - q / b
                kelly_fraction = p - (q / b)
                # Yarım Kelly (Half-Kelly) daha muhafazakardır ve aşırı kaldıraç riskini önler
                half_kelly = max(0.0, kelly_fraction / 2.0)
                
                # Maksimum risk limitini aşmasın
                effective_risk_pct = min(risk_pct, half_kelly)
                position_size = (balance * effective_risk_pct) / sl_pct
                logger.info(f"Kelly hesaplandı | Güven: {p:.2f} | R/R: {b:.2f} | Kelly Kesri: {kelly_fraction:.3f} | Half-Kelly: {half_kelly:.3f}")
            else:
                position_size = risk_usdt / sl_pct
        else:
            position_size = 0.0

        # Pozisyon boyutunu kaldıraçlı bakiye ile sınırla
        leverage = getattr(self.settings.execution, "leverage", 1)
        max_leverage_size = balance * leverage
        if position_size > max_leverage_size:
            logger.debug(f"Hesaplanan boyut (${position_size:,.2f}) kaldıraçlı bakiyeyi (${max_leverage_size:,.2f}) aşıyor. Kaldıraçlı sınıra eşitlendi.")
            position_size = max_leverage_size

        # Gereken marjinin bakiyeden küçük olduğunu doğrula
        required_margin = position_size / leverage
        if required_margin > balance:
            logger.debug(f"Gerekli marjin (${required_margin:,.2f}) bakiyeyi aşıyor. Pozisyon boyutu düşürülüyor.")
            position_size = balance * leverage

        # Çok küçük pozisyonları engelle (Örn: Minimum 10 USDT emir limiti)
        if position_size < 10.0 and len(rejections) == 0:
            rejections.append(
                f"Yetersiz Pozisyon Boyutu: ${position_size:.2f} < Min Limit: $10.00. "
                f"Muhtemel stop-loss mesafesi çok geniş veya bakiye çok düşük."
            )

        # Değerlendirme sonucunu üret
        is_approved = len(rejections) == 0
        final_size = position_size if is_approved else 0.0
        final_risk_pct = (final_size * sl_pct) / balance if balance > 0 else 0.0

        if not is_approved:
            logger.warning(
                f"[{signal.symbol}@{signal.timeframe}] Risk Değerlendirmesi REDDEDİLDİ | "
                f"Nedenler: {rejections}"
            )
        else:
            logger.info(
                f"[{signal.symbol}@{signal.timeframe}] Risk Değerlendirmesi ONAYLANDI | "
                f"Önerilen Boyut: ${final_size:,.2f} (%{final_size/balance*100:.1f}) | "
                f"İşlem Riski: %{final_risk_pct*100:.2f}"
            )

        return RiskAssessment(
            is_approved=is_approved,
            position_size_usdt=round(final_size, 2),
            risk_pct=round(final_risk_pct, 4),
            stop_loss_pct=round(sl_pct, 4),
            rejection_reasons=tuple(rejections),
        )

    def check_correlation(
        self,
        new_symbol: str,
        open_positions: dict,
        price_data: dict,
        max_correlation: float = 0.80,
    ) -> bool:
        """Yeni pozisyonun mevcut pozisyonlarla korelasyonunu kontrol eder.

        Args:
            new_symbol: Açılmak istenen sembol.
            open_positions: Açık pozisyon sözlüğü {symbol: position}.
            price_data: Fiyat verileri {symbol: pd.Series}.
            max_correlation: Maksimum izin verilen korelasyon.

        Returns:
            True ise pozisyon açılabilir, False ise açılamaz.
        """
        if not open_positions or new_symbol not in price_data:
            return True

        new_prices = price_data.get(new_symbol)
        if new_prices is None or len(new_prices) < 30:
            return True

        new_returns = new_prices.pct_change().dropna()

        for sym in open_positions:
            if sym == new_symbol:
                continue
            existing_prices = price_data.get(sym)
            if existing_prices is None or len(existing_prices) < 30:
                continue

            existing_returns = existing_prices.pct_change().dropna()
            min_len = min(len(new_returns), len(existing_returns))
            if min_len < 10:
                continue

            corr = np.corrcoef(
                new_returns.iloc[-min_len:].values,
                existing_returns.iloc[-min_len:].values,
            )[0, 1]

            if abs(corr) >= max_correlation:
                logger.warning(
                    f"Korelasyon engeli: {new_symbol} ile {sym} arası korelasyon "
                    f"%{corr*100:.1f} (limit: %{max_correlation*100:.1f})"
                )
                return False

        return True

    def calculate_dynamic_trailing(
        self,
        entry_price: float,
        current_price: float,
        original_sl: float,
        side: str,
        atr: float,
    ) -> float:
        """Dinamik trailing stop loss hesaplar.

        Kâr arttıkça stop loss'u yukarı/aşağı taşır (1.5 ATR takip eden stop, 1.0 ATR kârda aktif).

        Args:
            entry_price: Giriş fiyatı.
            current_price: Güncel fiyat.
            original_sl: Orijinal stop loss.
            side: "LONG" veya "SHORT".
            atr: Mevcut ATR değeri.

        Returns:
            Güncellenmiş stop loss fiyatı.
        """
        if side == "LONG":
            profit = current_price - entry_price
            profit_atr = profit / atr if atr > 0 else 0

            if profit_atr >= 1.0:
                # 1.0 ATR kâra ulaşınca en az başabaş (entry) noktasına çek
                # Daha yüksek kârlarda ise fiyatın 1.5 ATR gerisinden takip et
                new_sl = max(entry_price, current_price - 1.5 * atr)
            else:
                new_sl = original_sl

            return max(new_sl, original_sl)

        else:  # SHORT
            profit = entry_price - current_price
            profit_atr = profit / atr if atr > 0 else 0

            if profit_atr >= 1.0:
                # 1.0 ATR kâra ulaşınca en az başabaş (entry) noktasına çek
                # Daha yüksek kârlarda ise fiyatın 1.5 ATR gerisinden takip et
                new_sl = min(entry_price, current_price + 1.5 * atr)
            else:
                new_sl = original_sl

            return min(new_sl, original_sl)
