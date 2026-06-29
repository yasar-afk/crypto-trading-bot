# ============================================================
# src/execution/models.py — ANTIGRAVITI Trading Bot
#
# AMAÇ:
#   Açık pozisyonları, emir geçmişini ve portföy durumunu temsil
#   eden veri modellerini tanımlar.
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v1.0 | İlk veri modelleri tanımı
#   2026-06-04 | v2.0 | TradeOrder'a SL/TP/close_reason eklendi
#                        MarketPosition'a trailing stop alanları eklendi
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeOrder:
    """Borsaya gönderilen veya simüle edilen bir alım-satım emri."""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # "buy" | "sell"
    price: float
    amount: float
    status: OrderStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)
    fee_usdt: float = 0.0
    error_message: Optional[str] = None
    # Yeni alanlar (v2.0)
    stop_loss: float = 0.0
    take_profit: float = 0.0
    close_reason: Optional[str] = None    # "STOP LOSS TRIGGERED" / "TAKE PROFIT TRIGGERED" / "STRATEGY EXIT"
    pnl_usdt: float = 0.0                 # Kapanış emrinde gerçekleşen PnL
    pnl_pct: float = 0.0                  # Kapanış emrinde gerçekleşen PnL %
    indicator_summary: Optional[Dict] = None  # Giriş anındaki indikatör özeti (AI analizi için)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "order_type": self.order_type.value,
            "side": self.side,
            "price": self.price,
            "amount": self.amount,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "fee_usdt": self.fee_usdt,
            "error_message": self.error_message,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "close_reason": self.close_reason,
            "pnl_usdt": self.pnl_usdt,
            "pnl_pct": self.pnl_pct,
            "indicator_summary": self.indicator_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TradeOrder:
        return cls(
            order_id=d["order_id"],
            symbol=d["symbol"],
            order_type=OrderType(d["order_type"]),
            side=d["side"],
            price=d["price"],
            amount=d["amount"],
            status=OrderStatus(d["status"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            fee_usdt=d.get("fee_usdt", 0.0),
            error_message=d.get("error_message"),
            stop_loss=d.get("stop_loss", 0.0),
            take_profit=d.get("take_profit", 0.0),
            close_reason=d.get("close_reason"),
            pnl_usdt=d.get("pnl_usdt", 0.0),
            pnl_pct=d.get("pnl_pct", 0.0),
            indicator_summary=d.get("indicator_summary"),
        )


@dataclass
class MarketPosition:
    """Aktif olarak taşınan bir alım-satım pozisyonu (Spot Long)."""
    symbol: str
    side: PositionSide
    entry_price: float
    amount: float
    cost_usdt: float  # Toplam giriş maliyeti (komisyon hariç)
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=datetime.utcnow)
    current_price: float = 0.0
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    # Trailing Stop alanları (v2.0)
    trailing_stop_active: bool = False
    trailing_stop_pct: float = 0.02      # Varsayılan %2 trailing
    highest_price: float = 0.0           # Long için takip edilen en yüksek fiyat
    indicator_summary: Optional[Dict] = None  # Giriş anındaki indikatör durumu (AI için)
    # OCO Emir Alanları (v2.1)
    oco_order_list_id: Optional[str] = None
    oco_limit_order_id: Optional[str] = None
    oco_stop_order_id: Optional[str] = None
    # Limit Buy Entegrasyonu (v2.2)
    status: str = "active"
    limit_order_id: Optional[str] = None

    def update_pnl(self, current_price: float) -> None:
        """Güncel fiyata göre kar/zarar durumunu hesaplar."""
        self.current_price = current_price
        if self.highest_price == 0.0:
            self.highest_price = current_price

        if self.side == PositionSide.LONG or self.side == "long":
            self.pnl_usdt = (current_price - self.entry_price) * self.amount
            self.pnl_pct = (current_price - self.entry_price) / self.entry_price
            # Trailing stop için en yüksek fiyatı güncelle
            if current_price > self.highest_price:
                self.highest_price = current_price
        else:
            self.pnl_usdt = (self.entry_price - current_price) * self.amount
            self.pnl_pct = (self.entry_price - current_price) / self.entry_price
            # SHORT için en düşük fiyatı (highest_price alanında) güncelle
            if current_price < self.highest_price:
                self.highest_price = current_price

    def get_trailing_stop_level(self) -> float:
        """Trailing stop seviyesini hesaplar (pozisyon kârdaysa SL'yi yukarı/aşağı çeker)."""
        if not self.trailing_stop_active or self.highest_price == 0.0:
            return self.stop_loss
        if self.side == PositionSide.LONG or self.side == "long":
            new_sl = self.highest_price * (1.0 - self.trailing_stop_pct)
            # Trailing stop hiçbir zaman mevcut SL'nin altına inemez
            return max(new_sl, self.stop_loss)
        else:
            # SHORT için trailing stop en düşük fiyattan yukarıdadır ve aşağı doğru çekilir
            new_sl = self.highest_price * (1.0 + self.trailing_stop_pct)
            # Trailing stop hiçbir zaman mevcut SL'nin üstüne çıkamaz
            return min(new_sl, self.stop_loss)

    def get_liquidation_price(self, leverage: float) -> float:
        """İzole marjin için yaklaşık likidasyon fiyatını hesaplar."""
        if leverage <= 0:
            return 0.0
        if self.side == PositionSide.LONG or self.side == "long":
            return self.entry_price * (1.0 - 1.0 / leverage)
        else:
            return self.entry_price * (1.0 + 1.0 / leverage)

    def distance_to_sl_pct(self) -> float:
        """Mevcut fiyatın stop-loss'a olan uzaklığı (%)."""
        active_sl = self.get_trailing_stop_level()
        if self.current_price <= 0 or active_sl <= 0:
            return 0.0
        return abs(self.current_price - active_sl) / self.current_price * 100

    def distance_to_tp_pct(self) -> float:
        """Mevcut fiyatın take-profit'e olan uzaklığı (%)."""
        if self.current_price <= 0 or self.take_profit <= 0:
            return 0.0
        return abs(self.take_profit - self.current_price) / self.current_price * 100

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "amount": self.amount,
            "cost_usdt": self.cost_usdt,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "opened_at": self.opened_at.isoformat(),
            "current_price": self.current_price,
            "pnl_usdt": self.pnl_usdt,
            "pnl_pct": self.pnl_pct,
            "trailing_stop_active": self.trailing_stop_active,
            "trailing_stop_pct": self.trailing_stop_pct,
            "highest_price": self.highest_price,
            "indicator_summary": self.indicator_summary,
            "oco_order_list_id": self.oco_order_list_id,
            "oco_limit_order_id": self.oco_limit_order_id,
            "oco_stop_order_id": self.oco_stop_order_id,
            "status": self.status,
            "limit_order_id": self.limit_order_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MarketPosition:
        pos = cls(
            symbol=d["symbol"],
            side=PositionSide(d["side"]),
            entry_price=d["entry_price"],
            amount=d["amount"],
            cost_usdt=d["cost_usdt"],
            stop_loss=d["stop_loss"],
            take_profit=d["take_profit"],
            opened_at=datetime.fromisoformat(d["opened_at"]),
            trailing_stop_active=d.get("trailing_stop_active", False),
            trailing_stop_pct=d.get("trailing_stop_pct", 0.02),
            highest_price=d.get("highest_price", 0.0),
            indicator_summary=d.get("indicator_summary"),
            oco_order_list_id=d.get("oco_order_list_id"),
            oco_limit_order_id=d.get("oco_limit_order_id"),
            oco_stop_order_id=d.get("oco_stop_order_id"),
            status=d.get("status", "active"),
            limit_order_id=d.get("limit_order_id"),
        )
        pos.current_price = d.get("current_price", 0.0)
        pos.pnl_usdt = d.get("pnl_usdt", 0.0)
        pos.pnl_pct = d.get("pnl_pct", 0.0)
        return pos
