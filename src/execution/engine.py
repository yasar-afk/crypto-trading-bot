# ============================================================
# src/execution/engine.py — Trading Bot Trading Bot
#
# AMAÇ:
#   Alım-satım kararlarını (Order/Position) borsaya veya kağıt üstünde
#   (Paper Trading) simülatöre gönderir ve açık pozisyonları yönetir.
#   Açık pozisyonların SL/TP takibini yapar.
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v1.0 | İlk ExecutionEngine entegrasyonu
#   2026-06-04 | v2.0 | Trailing Stop-Loss, AI post-mortem analizi,
#                        SL/TP kayıt düzeltmesi, monitor_positions save_state
# ============================================================

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt

from src.config.settings import Settings, get_settings
from src.execution.models import (
    MarketPosition,
    OrderStatus,
    OrderType,
    PositionSide,
    TradeOrder,
)
from src.risk.engine import RiskAssessment, RiskEngine
from src.signal.models import SignalStrength, SignalType, TradeSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionEngine:
    """Emir iletimini ve açık pozisyonların yönetimini koordine eden sınıf."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        risk_engine: Optional[RiskEngine] = None,
        exchange: Optional[ccxt.binance] = None,
        state_dir: str = "logs",
        state_file: str = "portfolio_state.json",
    ) -> None:
        self.settings = settings or get_settings()
        self.risk_engine = risk_engine
        self.exchange = exchange

        self.is_paper_trade = self.settings.is_paper_trade
        self.state_path = Path(state_dir) / state_file

        # İç durum değişkenleri
        self.usdt_balance: float = self.settings.paper_trading.initial_balance_usdt
        self.open_positions: Dict[str, MarketPosition] = {}
        self.orders: List[TradeOrder] = []

        # Kaydedilmiş durum varsa yükle
        self.load_state()

        # RiskEngine'e başlangıç bakiyesini haber ver
        if self.risk_engine:
            self.risk_engine.set_balance(self.usdt_balance)

        # Ensure state file is written immediately so watch.py sees the bot as running
        self.save_state()

        # Telegram Command Listener
        try:
            from src.utils.telegram_listener import start_telegram_listener
            start_telegram_listener(self)
        except Exception as e:
            logger.error(f"Failed to start Telegram listener: {e}")

        logger.info(
            f"ExecutionEngine başlatıldı | "
            f"Mod: {'PAPER TRADE' if self.is_paper_trade else 'CANLI'} | "
            f"Bakiye: ${self.usdt_balance:,.2f} USDT | "
            f"Açık Pozisyon Sayısı: {len(self.open_positions)} | "
            f"Durum Dosyası: {self.state_path}"
        )

    # ── State Yönetimi (Kalıcılık) ─────────────────────────────

    def save_state(self) -> None:
        """Portföy durumunu (bakiye, pozisyonlar, emirler) JSON dosyasına kaydeder."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            state_dict = {
                "usdt_balance": self.usdt_balance,
                "open_positions": {
                    sym: pos.to_dict() for sym, pos in self.open_positions.items()
                },
                "orders": [order.to_dict() for order in self.orders],
                "last_updated": datetime.utcnow().isoformat(),
            }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=4, ensure_ascii=False)
            logger.debug("Portföy durumu başarıyla kaydedildi.")
        except Exception as e:
            logger.error(f"Portföy durumu kaydedilemedi: {e}")

    def load_state(self) -> None:
        """Kayıtlı portföy durumunu JSON dosyasından yükler."""
        if not self.state_path.exists():
            logger.info("Kayıtlı portföy durumu bulunamadı. Yeni durum oluşturulacak.")
            return

        try:
            with open(self.state_path, encoding="utf-8") as f:
                state_dict = json.load(f)

            self.usdt_balance = state_dict.get(
                "usdt_balance",
                self.settings.paper_trading.initial_balance_usdt
            )

            # Pozisyonları yükle
            positions_data = state_dict.get("open_positions", {})
            self.open_positions = {
                sym: MarketPosition.from_dict(pos) for sym, pos in positions_data.items()
            }

            # Emirleri yükle
            orders_data = state_dict.get("orders", [])
            self.orders = [TradeOrder.from_dict(o) for o in orders_data]

            logger.info(
                f"Kayıtlı portföy durumu başarıyla yüklendi | "
                f"Bakiye: ${self.usdt_balance:,.2f} USDT | "
                f"Açık Pozisyonlar: {list(self.open_positions.keys())}"
            )
        except Exception as e:
            logger.error(f"Portföy durumu yüklenirken hata oluştu: {e}. Varsayılan değerler kullanılıyor.")

    @property
    def total_portfolio_value(self) -> float:
        """Toplam portföy değerini hesaplar (nakit + marjin maliyetleri + anlık kâr/zarar)."""
        open_cost = sum(pos.cost_usdt for pos in self.open_positions.values())
        open_pnl = sum(pos.pnl_usdt for pos in self.open_positions.values())
        return self.usdt_balance + open_cost + open_pnl

    # ── Pozisyon ve Sinyal Yönetimi ────────────────────────────

    def execute_signal(
        self,
        signal: TradeSignal,
        assessment: RiskAssessment,
    ) -> Optional[TradeOrder]:
        """Bir işlem sinyalini değerlendirme onayına göre işleme sokar."""
        symbol = signal.symbol
        is_buy = signal.signal_type == SignalType.BUY
        is_sell = signal.signal_type == SignalType.SELL

        if not signal.is_actionable:
            return None

        # ── Circuit Breakers Check (New Entries Block - V2 Only) ──
        is_new_entry = False
        if is_buy and (symbol not in self.open_positions):
            is_new_entry = True
        elif is_sell and (symbol not in self.open_positions):
            is_new_entry = True

        if is_new_entry and self.settings.strategy.version == "v2":
            from datetime import timedelta
            now = datetime.utcnow()
            
            # Portföy toplam değerini hesapla
            open_cost = sum(pos.cost_usdt for pos in self.open_positions.values())
            open_pnl = sum(pos.pnl_usdt for pos in self.open_positions.values())
            total_portfolio = self.usdt_balance + open_cost + open_pnl
            
            if total_portfolio > 0:
                # L1: Günlük Zarar Limiti (-3% / 24 saat)
                cutoff_day = now - timedelta(hours=24)
                daily_trades = [o for o in self.orders if o.status == OrderStatus.FILLED and o.close_reason is not None and o.timestamp > cutoff_day]
                total_daily_pnl = sum(o.pnl_usdt for o in daily_trades)
                daily_loss_pct = -total_daily_pnl / total_portfolio
                if total_daily_pnl < 0 and daily_loss_pct >= 0.03:
                    logger.warning(f"⛔ [CIRCUIT BREAKER L1] Günlük zarar limiti aşıldı! Kayıp: %{daily_loss_pct*100:.2f} >= %3.0. Yeni giriş engellendi.")
                    return None

                # L3: Haftalık Zarar Limiti (-7% / 7 gün)
                cutoff_week = now - timedelta(days=7)
                weekly_trades = [o for o in self.orders if o.status == OrderStatus.FILLED and o.close_reason is not None and o.timestamp > cutoff_week]
                total_weekly_pnl = sum(o.pnl_usdt for o in weekly_trades)
                weekly_loss_pct = -total_weekly_pnl / total_portfolio
                if total_weekly_pnl < 0 and weekly_loss_pct >= 0.07:
                    logger.warning(f"⛔ [CIRCUIT BREAKER L3] Haftalık zarar limiti aşıldı! Kayıp: %{weekly_loss_pct*100:.2f} >= %7.0. Yeni giriş engellendi.")
                    return None

            # L2: Ardışık 3 Zarar Kontrolü (24 saat duraklama)
            closed_trades = [o for o in self.orders if o.status == OrderStatus.FILLED and o.close_reason is not None]
            closed_trades.sort(key=lambda x: x.timestamp)
            last_3 = closed_trades[-3:]
            if len(last_3) == 3 and all(o.pnl_usdt < 0 for o in last_3):
                last_close_time = last_3[-1].timestamp
                if (now - last_close_time).total_seconds() < 24 * 3600:
                    remaining_hours = 24 - (now - last_close_time).total_seconds() / 3600
                    logger.warning(f"⛔ [CIRCUIT BREAKER L2] Ardışık 3 zarar! Bot 24 saat duraklatıldı. Kalan süre: {remaining_hours:.1f} saat. Yeni giriş engellendi.")
                    return None

        # 1. Alış (BUY) Senaryosu (LONG Giriş veya SHORT Kapatma)
        if is_buy:
            if symbol in self.open_positions:
                pos = self.open_positions[symbol]
                if pos.side == PositionSide.SHORT or pos.side == "short":
                    logger.info(f"🔄 [{symbol}] Alış sinyali geldi. Mevcut SHORT pozisyon kapatılıyor...")
                    return self._close_position(pos, signal.entry_price, "STRATEGY EXIT")
                else:
                    logger.warning(f"[{symbol}] için zaten açık bir LONG pozisyon var. İkinci pozisyon açılmıyor.")
                    return None

            if not assessment.is_approved:
                logger.warning(f"[{symbol}] Alış sinyali RiskEngine tarafından reddedildi. Pozisyon açılmıyor.")
                return None

            # ── MAX_CONCURRENT_POSITIONS Limit Kontrolü ──
            from src.strategy.feature_weights import SCORING_CONFIG
            max_pos = SCORING_CONFIG.get("MAX_CONCURRENT_POSITIONS", 5)
            if len(self.open_positions) >= max_pos:
                logger.warning(
                    f"⛔ [{symbol}] Açık pozisyon limiti doldu! "
                    f"Mevcut: {len(self.open_positions)} / Limit: {max_pos}. "
                    f"Yeni alış engellendi."
                )
                return None

            return self._open_position(signal, assessment)

        # 2. Satış (SELL) Senaryosu (SHORT Giriş veya LONG Kapatma)
        elif is_sell:
            if symbol in self.open_positions:
                pos = self.open_positions[symbol]
                if pos.side == PositionSide.LONG or pos.side == "long":
                    logger.info(f"🔄 [{symbol}] Satış sinyali geldi. Mevcut LONG pozisyon kapatılıyor...")
                    return self._close_position(pos, signal.entry_price, "STRATEGY EXIT")
                else:
                    logger.warning(f"[{symbol}] için zaten açık bir SHORT pozisyon var. İkinci pozisyon açılmıyor.")
                    return None

            # Long-only stratejilerde SHORT açılmasını engelle (V1 ve V2 long-only'dir)
            strategy_ver = getattr(self.settings.strategy, "version", "v2")
            if strategy_ver not in ("v2.1", "v3"):
                logger.info(f"ℹ️ [{symbol}] Satış sinyali geldi ancak strateji '{strategy_ver}' (long-only). Yeni SHORT pozisyon açılmıyor.")
                return None

            if not assessment.is_approved:
                logger.warning(f"[{symbol}] Satış sinyali RiskEngine tarafından reddedildi. Pozisyon açılmıyor.")
                return None

            # ── MAX_CONCURRENT_POSITIONS Limit Kontrolü ──
            from src.strategy.feature_weights import SCORING_CONFIG
            max_pos = SCORING_CONFIG.get("MAX_CONCURRENT_POSITIONS", 5)
            if len(self.open_positions) >= max_pos:
                logger.warning(
                    f"⛔ [{symbol}] Açık pozisyon limiti doldu! "
                    f"Mevcut: {len(self.open_positions)} / Limit: {max_pos}. "
                    f"Yeni satış engellendi."
                )
                return None

            return self._open_position(signal, assessment)

        return None

    def _open_position(
        self,
        signal: TradeSignal,
        assessment: RiskAssessment,
    ) -> Optional[TradeOrder]:
        """Açık pozisyon oluşturma işlemini yönetir (Paper veya Canlı)."""
        symbol = signal.symbol
        price = signal.entry_price
        size_usdt = assessment.position_size_usdt

        commission_rate = self.settings.paper_trading.commission_rate
        fee_usdt = size_usdt * commission_rate
        total_cost = size_usdt + fee_usdt

        order_id = str(uuid.uuid4())[:8]

        order_type_setting = getattr(self.settings.execution, "order_type", "market")
        is_limit = order_type_setting == "limit"

        is_buy = signal.signal_type == SignalType.BUY
        side_str = "buy" if is_buy else "sell"
        pos_side = PositionSide.LONG if is_buy else PositionSide.SHORT

        # ── Paper Trade Modu ─────────────────────────────────
        if self.is_paper_trade:
            if self.usdt_balance < total_cost:
                logger.error(
                    f"❌ [{symbol}] Yetersiz bakiye! "
                    f"Mevcut: ${self.usdt_balance:.2f} | Gerekli: ${total_cost:.2f}"
                )
                order = TradeOrder(
                    order_id=order_id, symbol=symbol, order_type=OrderType.LIMIT if is_limit else OrderType.MARKET,
                    side=side_str, price=price, amount=0.0, status=OrderStatus.FAILED,
                    fee_usdt=0.0, error_message="Insufficient balance",
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                )
                self.orders.append(order)
                self.save_state()
                return order

            amount = size_usdt / price
            self.usdt_balance -= total_cost

            if self.risk_engine:
                self.risk_engine.set_balance(self.usdt_balance)

            if is_limit:
                pos = MarketPosition(
                    symbol=symbol,
                    side=pos_side,
                    entry_price=price,
                    amount=amount,
                    cost_usdt=size_usdt,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    opened_at=datetime.utcnow(),
                    indicator_summary=signal.indicator_summary if hasattr(signal, 'indicator_summary') else None,
                    status="pending",
                    limit_order_id=order_id
                )
                pos.update_pnl(price)
                self.open_positions[symbol] = pos

                order = TradeOrder(
                    order_id=order_id, symbol=symbol, order_type=OrderType.LIMIT,
                    side=side_str, price=price, amount=amount, status=OrderStatus.PENDING,
                    fee_usdt=fee_usdt,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )
                self.orders.append(order)
                self.save_state()
                self._send_portfolio_summary(f"Limit Emir Yerleştirildi ({side_str.upper()}): {symbol} @ {price:,.4f}")

                logger.info(
                    f"⏳ [PAPER-LIMIT] Limit {side_str.upper()} Emri yerlestirildi: {symbol} | "
                    f"Miktar: {amount:.6f} | Limit Fiyat: ${price:,.4f} | "
                    f"Maliyet: ${size_usdt:,.2f} | Komisyon: ${fee_usdt:.2f} | "
                    f"SL: ${signal.stop_loss:,.4f} | TP: ${signal.take_profit:,.4f} | "
                    f"Kalan Kullanilabilir Bakiye: ${self.usdt_balance:,.2f} USDT"
                )
                return order
            else:
                pos = MarketPosition(
                    symbol=symbol,
                    side=pos_side,
                    entry_price=price,
                    amount=amount,
                    cost_usdt=size_usdt,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    opened_at=datetime.utcnow(),
                    indicator_summary=signal.indicator_summary if hasattr(signal, 'indicator_summary') else None,
                )
                pos.update_pnl(price)
                self.open_positions[symbol] = pos

                order = TradeOrder(
                    order_id=order_id, symbol=symbol, order_type=OrderType.MARKET,
                    side=side_str, price=price, amount=amount, status=OrderStatus.FILLED,
                    fee_usdt=fee_usdt,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )
                self.orders.append(order)
                self.save_state()
                self._send_portfolio_summary(f"Pozisyon Açıldı ({pos_side.value.upper()}): {symbol} @ {price:,.4f}")

                logger.info(
                    f"🟢 [PAPER] Pozisyon AÇILDI ({pos_side.value.upper()}): {symbol} | "
                    f"Miktar: {amount:.6f} | Fiyat: ${price:,.4f} | "
                    f"Maliyet: ${size_usdt:,.2f} | Komisyon: ${fee_usdt:.2f} | "
                    f"SL: ${signal.stop_loss:,.4f} | TP: ${signal.take_profit:,.4f} | "
                    f"Kalan Bakiye: ${self.usdt_balance:,.2f} USDT"
                )
                return order

        # ── Canlı Mod (Live CCXT) ────────────────────────────
        else:
            if not self.exchange:
                logger.error("❌ HATA: Canlı mod aktif ancak CCXT exchange nesnesi bulunamadı!")
                return None

            # ── Kaldıraç ve İzole Marjin Ayarlama ──
            is_futures = getattr(self.settings.exchange, "default_type", "spot") == "future"
            if is_futures:
                try:
                    leverage_val = getattr(self.settings.execution, "leverage", 5)
                    margin_mode_val = getattr(self.settings.execution, "margin_mode", "ISOLATED").upper()

                    logger.info(f"⚙️ [CANLI] Futures kaldıraç ayarlanıyor: {symbol} -> {leverage_val}x")
                    self.exchange.set_leverage(leverage_val, symbol)

                    logger.info(f"⚙️ [CANLI] Futures marjin tipi ayarlanıyor: {symbol} -> {margin_mode_val}")
                    try:
                        self.exchange.set_margin_mode(margin_mode_val, symbol)
                    except Exception as margin_err:
                        logger.info(f"ℹ️ Marjin tipi zaten {margin_mode_val} olabilir: {margin_err}")
                except Exception as exchange_config_err:
                    logger.error(f"⚠️ [CANLI] Kaldıraç veya Marjin ayarlama hatası: {exchange_config_err}")

            if is_limit:
                try:
                    amount = size_usdt / price
                    amount = self.exchange.amount_to_precision(symbol, amount)

                    logger.info(f"💰 [CANLI] Borsa limit emri: {symbol} | Yön: {side_str.upper()} | Miktar: {amount} | Fiyat: {price}")
                    ccxt_order = self.exchange.create_order(symbol, "limit", side_str, float(amount), float(price))

                    ccxt_id = ccxt_order.get("id") or order_id
                    ccxt_status = ccxt_order.get("status")  # 'open', 'closed', 'canceled', etc.
                    ccxt_price = ccxt_order.get("price") or price
                    ccxt_amount = ccxt_order.get("amount") or float(amount)

                    pos = MarketPosition(
                        symbol=symbol,
                        side=pos_side,
                        entry_price=ccxt_price,
                        amount=ccxt_amount,
                        cost_usdt=ccxt_amount * ccxt_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        opened_at=datetime.utcnow(),
                        indicator_summary=signal.indicator_summary if hasattr(signal, 'indicator_summary') else None,
                        status="pending" if ccxt_status != "closed" else "active",
                        limit_order_id=ccxt_id
                    )
                    pos.update_pnl(ccxt_price)
                    self.open_positions[symbol] = pos

                    if ccxt_status == "closed":
                        # Limit emir anında dolduysa korumaları kur
                        if signal.stop_loss > 0 and signal.take_profit > 0:
                            if is_futures:
                                self._create_futures_protection_orders(symbol, pos, signal.stop_loss, signal.take_profit)
                            else:
                                try:
                                    stop_loss_price = signal.stop_loss
                                    take_profit_price = signal.take_profit
                                    stop_limit_price = stop_loss_price * 0.99
                                    oco_res = self.exchange.privatePostOrderOco({
                                        'symbol': symbol.replace('/', ''),
                                        'side': 'SELL',
                                        'quantity': self.exchange.amount_to_precision(symbol, ccxt_amount),
                                        'price': self.exchange.price_to_precision(symbol, take_profit_price),
                                        'stopPrice': self.exchange.price_to_precision(symbol, stop_loss_price),
                                        'stopLimitPrice': self.exchange.price_to_precision(symbol, stop_limit_price),
                                        'stopLimitTimeInForce': 'GTC'
                                    })
                                    pos.oco_order_list_id = str(oco_res.get('orderListId'))
                                    for o in oco_res.get('orders', []):
                                        if o.get('type') == 'LIMIT_MAKER':
                                            pos.oco_limit_order_id = str(o.get('orderId'))
                                        elif o.get('type') == 'STOP_LOSS_LIMIT':
                                            pos.oco_stop_order_id = str(o.get('orderId'))
                                except Exception as e:
                                    logger.error(f"❌ [CANLI] OCO Emri Gönderilemedi (Anında dolan limit sonrası): {e}", exc_info=True)

                    order = TradeOrder(
                        order_id=ccxt_id,
                        symbol=symbol, order_type=OrderType.LIMIT,
                        side=side_str, price=ccxt_price, amount=ccxt_amount,
                        status=OrderStatus.FILLED if ccxt_status == "closed" else OrderStatus.PENDING,
                        fee_usdt=0.0,
                        stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    )
                    self.orders.append(order)
                    self.save_state()
                    status_text = "Pozisyon Açıldı" if ccxt_status == "closed" else "Limit Emir Yerleştirildi"
                    self._send_portfolio_summary(f"{status_text} ({pos_side.value.upper()}): {symbol} @ {ccxt_price:,.4f}")

                    if ccxt_status == "closed":
                        logger.info(
                            f"🔥 [CANLI] Limit emri anında doldu ve pozisyon AÇILDI ({pos_side.value.upper()}): {symbol} | "
                            f"Miktar: {ccxt_amount} | Fiyat: ${ccxt_price:,.4f}"
                        )
                    else:
                        logger.info(
                            f"⏳ [CANLI] Limit emri yerleştirildi (Beklemede): {symbol} | "
                            f"Miktar: {ccxt_amount} | Fiyat: ${ccxt_price:,.4f} | ID: {ccxt_id}"
                        )
                    return order

                except Exception as e:
                    logger.error(f"❌ [CANLI] Limit emri hatası: {e}", exc_info=True)
                    order = TradeOrder(
                        order_id=order_id, symbol=symbol, order_type=OrderType.LIMIT,
                        side=side_str, price=price, amount=0.0, status=OrderStatus.FAILED,
                        error_message=str(e), stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    )
                    self.orders.append(order)
                    self.save_state()
                    return order
            else:
                try:
                    amount = size_usdt / price
                    amount = self.exchange.amount_to_precision(symbol, amount)

                    logger.info(f"💰 [CANLI] Borsa market emri: {symbol} | Yön: {side_str.upper()} | Miktar: {amount}")
                    ccxt_order = self.exchange.create_order(symbol, "market", side_str, float(amount))

                    filled_price = ccxt_order.get("average") or ccxt_order.get("price") or price
                    filled_amount = ccxt_order.get("filled") or ccxt_order.get("amount") or float(amount)
                    ccxt_fee = ccxt_order.get("fee", {})
                    ccxt_fee_cost = ccxt_fee.get("cost", 0.0) if ccxt_fee else fee_usdt

                    pos = MarketPosition(
                        symbol=symbol,
                        side=pos_side,
                        entry_price=filled_price,
                        amount=filled_amount,
                        cost_usdt=filled_amount * filled_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        opened_at=datetime.utcnow(),
                        indicator_summary=signal.indicator_summary if hasattr(signal, 'indicator_summary') else None,
                    )
                    pos.update_pnl(filled_price)

                    # ── Borsa Tarafında Korumaları Oluşturma ───
                    if signal.stop_loss > 0 and signal.take_profit > 0:
                        if is_futures:
                            self._create_futures_protection_orders(symbol, pos, signal.stop_loss, signal.take_profit)
                        else:
                            try:
                                stop_loss_price = signal.stop_loss
                                take_profit_price = signal.take_profit
                                stop_limit_price = stop_loss_price * 0.99
                                
                                logger.info(f"📊 [CANLI] OCO Emri Gönderiliyor: {symbol} | TP: {take_profit_price} | SL: {stop_loss_price}")
                                
                                oco_res = self.exchange.privatePostOrderOco({
                                    'symbol': symbol.replace('/', ''),
                                    'side': 'SELL',
                                    'quantity': self.exchange.amount_to_precision(symbol, filled_amount),
                                    'price': self.exchange.price_to_precision(symbol, take_profit_price),
                                    'stopPrice': self.exchange.price_to_precision(symbol, stop_loss_price),
                                    'stopLimitPrice': self.exchange.price_to_precision(symbol, stop_limit_price),
                                    'stopLimitTimeInForce': 'GTC'
                                })
                                
                                pos.oco_order_list_id = str(oco_res.get('orderListId'))
                                orders_info = oco_res.get('orders', [])
                                for o in orders_info:
                                    if o.get('type') == 'LIMIT_MAKER':
                                        pos.oco_limit_order_id = str(o.get('orderId'))
                                    elif o.get('type') == 'STOP_LOSS_LIMIT':
                                        pos.oco_stop_order_id = str(o.get('orderId'))
                                        
                                logger.info(f"🔥 [CANLI] OCO Emri Oluşturuldu: listId={pos.oco_order_list_id} | LimitId={pos.oco_limit_order_id} | StopId={pos.oco_stop_order_id}")
                            except Exception as e:
                                logger.error(f"❌ [CANLI] OCO Emri Gönderilemedi: {e}", exc_info=True)

                    self.open_positions[symbol] = pos

                    order = TradeOrder(
                        order_id=ccxt_order.get("id", order_id),
                        symbol=symbol, order_type=OrderType.MARKET,
                        side=side_str, price=filled_price, amount=filled_amount,
                        status=OrderStatus.FILLED, fee_usdt=ccxt_fee_cost,
                        stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    )
                    self.orders.append(order)
                    self.save_state()
                    self._send_portfolio_summary(f"Pozisyon Açıldı ({pos_side.value.upper()}): {symbol} @ {filled_price:,.4f}")

                    logger.info(
                        f"🔥 [CANLI] Pozisyon AÇILDI ({pos_side.value.upper()}): {symbol} | "
                        f"Miktar: {filled_amount} | Fiyat: ${filled_price:,.4f} | "
                        f"SL: {signal.stop_loss} | TP: {signal.take_profit}"
                    )
                    return order

                except Exception as e:
                    logger.error(f"❌ [CANLI] Açılış emri hatası: {e}", exc_info=True)
                    order = TradeOrder(
                        order_id=order_id, symbol=symbol, order_type=OrderType.MARKET,
                        side=side_str, price=price, amount=0.0, status=OrderStatus.FAILED,
                        error_message=str(e), stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    )
                    self.orders.append(order)
                    self.save_state()
                    return order

    def _close_position(
        self,
        position: MarketPosition,
        exit_price: float,
        reason: str,
    ) -> Optional[TradeOrder]:
        """Açık bir pozisyonu kapatma işlemini yönetir (Paper veya Canlı)."""
        symbol = position.symbol
        amount = position.amount

        commission_rate = self.settings.paper_trading.commission_rate
        revenue = amount * exit_price
        fee_usdt = revenue * commission_rate

        # Yönlü PnL Hesaplaması
        if position.side == PositionSide.LONG or position.side == "long":
            raw_pnl = (exit_price - position.entry_price) * amount
            close_side = "sell"
        else:
            raw_pnl = (position.entry_price - exit_price) * amount
            close_side = "buy"

        pnl_usdt = raw_pnl - fee_usdt
        pnl_pct = pnl_usdt / position.cost_usdt if position.cost_usdt > 0 else 0.0

        order_id = str(uuid.uuid4())[:8]

        # ── Paper Trade Modu ─────────────────────────────────
        if self.is_paper_trade:
            # Paper modda bakiyeye kilitli pozisyon maliyetini ve PnL'i iade et
            self.usdt_balance += (position.cost_usdt + pnl_usdt)

            if self.risk_engine:
                self.risk_engine.set_balance(self.usdt_balance)
                if pnl_usdt < 0:
                    self.risk_engine.record_loss(abs(pnl_usdt))

            order = TradeOrder(
                order_id=order_id, symbol=symbol, order_type=OrderType.MARKET,
                side=close_side, price=exit_price, amount=amount, status=OrderStatus.FILLED,
                fee_usdt=fee_usdt,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                close_reason=reason,
                pnl_usdt=round(pnl_usdt, 2),
                pnl_pct=round(pnl_pct, 4),
                indicator_summary=position.indicator_summary,
            )
            self.orders.append(order)
            del self.open_positions[symbol]
            self.save_state()
            pnl_sign = "+" if pnl_usdt >= 0 else ""
            self._send_portfolio_summary(f"Pozisyon Kapatıldı ({reason}): {symbol} | PnL: {pnl_sign}${pnl_usdt:.2f} ({pnl_pct*100:+.2f}%)")

            pnl_emoji = "🟢 KAR" if pnl_usdt >= 0 else "🔴 ZARAR"
            logger.info(
                f"🔴 [PAPER] Pozisyon KAPATILDI ({reason}): {symbol} | Yön: {position.side.value.upper()} | "
                f"Cikis: ${exit_price:,.4f} | "
                f"PnL: {pnl_emoji} ${pnl_usdt:,.2f} (%{pnl_pct*100:.2f}) | "
                f"Yeni Bakiye: ${self.usdt_balance:,.2f} USDT"
            )

            # ── AI Post-Mortem Analizi ─────────────────────────────
            self._run_ai_postmortem(position, exit_price, pnl_usdt, pnl_pct, reason)

            return order

        # ── Canlı Mod (Live CCXT) ────────────────────────────
        else:
            if not self.exchange:
                logger.error("❌ HATA: Canlı mod aktif ancak CCXT exchange nesnesi bulunamadı!")
                return None

            try:
                is_futures = getattr(self.settings.exchange, "default_type", "spot") == "future"

                # ── Futures Koruma Emirlerini İptal Et ──
                if is_futures:
                    for protection_id in [position.oco_stop_order_id, position.oco_limit_order_id]:
                        if protection_id:
                            try:
                                logger.info(f"🧹 [CANLI] Aktif Futures koruma emri iptal ediliyor: {protection_id}")
                                self.exchange.cancel_order(protection_id, symbol)
                            except Exception as e:
                                logger.warning(f"⚠️ Koruma emri iptal edilemedi: {e}")
                # ── Borsa Tarafında Aktif OCO Emri Varsa İptal Et (Spot) ───
                elif position.oco_order_list_id:
                    try:
                        logger.info(f"🧹 [CANLI] Pozisyon manuel/strateji ile kapatılıyor. Aktif OCO emri iptal ediliyor: listId={position.oco_order_list_id}")
                        self.exchange.privateDeleteOrderList({
                            'symbol': symbol.replace('/', ''),
                            'orderListId': position.oco_order_list_id
                        })
                        logger.info("✅ OCO emri başarıyla iptal edildi.")
                    except Exception as e:
                        logger.warning(f"⚠️ OCO emri iptal edilemedi (zaten dolmuş veya iptal edilmiş olabilir): {e}")

                amount_prec = self.exchange.amount_to_precision(symbol, amount)
                logger.info(f"💰 [CANLI] Market {close_side.upper()} order (Close): {symbol} | Miktar: {amount_prec}")

                ccxt_order = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=close_side,
                    amount=float(amount_prec),
                    price=None,
                    params={'reduceOnly': True} if is_futures else {}
                )

                filled_price = ccxt_order.get("average") or ccxt_order.get("price") or exit_price
                filled_amount = ccxt_order.get("filled") or ccxt_order.get("amount") or float(amount_prec)
                ccxt_fee = ccxt_order.get("fee", {})
                ccxt_fee_cost = ccxt_fee.get("cost", 0.0) if ccxt_fee else fee_usdt

                if position.side == PositionSide.LONG or position.side == "long":
                    live_pnl = (filled_price - position.entry_price) * filled_amount - ccxt_fee_cost
                else:
                    live_pnl = (position.entry_price - filled_price) * filled_amount - ccxt_fee_cost

                live_pnl_pct = live_pnl / position.cost_usdt if position.cost_usdt > 0 else 0.0

                if self.risk_engine and live_pnl < 0:
                    self.risk_engine.record_loss(abs(live_pnl))

                order = TradeOrder(
                    order_id=ccxt_order.get("id", order_id),
                    symbol=symbol, order_type=OrderType.MARKET,
                    side=close_side, price=filled_price, amount=filled_amount,
                    status=OrderStatus.FILLED, fee_usdt=ccxt_fee_cost,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    close_reason=reason,
                    pnl_usdt=round(live_pnl, 2),
                    pnl_pct=round(live_pnl_pct, 4),
                    indicator_summary=position.indicator_summary,
                )
                self.orders.append(order)
                del self.open_positions[symbol]
                self.save_state()
                pnl_sign = "+" if live_pnl >= 0 else ""
                self._send_portfolio_summary(f"Pozisyon Kapatıldı ({reason}): {symbol} | PnL: {pnl_sign}${live_pnl:.2f} ({live_pnl_pct*100:+.2f}%)")

                pnl_emoji = "🟢 KAR" if live_pnl >= 0 else "🔴 ZARAR"
                logger.info(
                    f"🔥 [CANLI] Pozisyon KAPATILDI ({reason}): {symbol} | Yön: {position.side.value.upper()} | "
                    f"Cikis: ${filled_price:,.4f} | "
                    f"PnL: {pnl_emoji} ${live_pnl:,.2f} (%{live_pnl_pct*100:.2f})"
                )

                self._run_ai_postmortem(position, filled_price, live_pnl, live_pnl_pct, reason)
                return order

            except Exception as e:
                logger.error(f"❌ [CANLI] Kapatma emri hatası: {e}", exc_info=True)
                order = TradeOrder(
                    order_id=order_id, symbol=symbol, order_type=OrderType.MARKET,
                    side=close_side, price=exit_price, amount=amount, status=OrderStatus.FAILED,
                    error_message=str(e), close_reason=reason,
                )
                self.orders.append(order)
                self.save_state()
                return order

    def _run_ai_postmortem(
        self,
        position: MarketPosition,
        exit_price: float,
        pnl_usdt: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        """Pozisyon kapandığında Gemini AI ile post-mortem analizi üretir."""
        try:
            from src.utils.ai_analyzer import generate_post_mortem_analysis
            analysis = generate_post_mortem_analysis(
                symbol=position.symbol,
                side=position.side.value,
                entry_price=position.entry_price,
                exit_price=exit_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                pnl_usdt=pnl_usdt,
                pnl_pct=pnl_pct,
                reason=reason,
                indicator_summary=position.indicator_summary,
            )
            pnl_emoji = "🟢" if pnl_usdt >= 0 else "🔴"
            logger.info(
                f"\n{'='*60}\n"
                f"🤖 AI POST-MORTEM ANALİZİ — {position.symbol}\n"
                f"{'='*60}\n"
                f"  PnL: {pnl_emoji} ${pnl_usdt:+,.2f} (%{pnl_pct*100:+.2f})\n"
                f"  Neden: {reason}\n"
                f"  AI Analizi: {analysis}\n"
                f"{'='*60}"
            )
        except Exception as e:
            logger.debug(f"AI post-mortem analizi üretilemedi: {e}")

    def _close_position_locally(
        self,
        position: MarketPosition,
        exit_price: float,
        reason: str,
    ) -> Optional[TradeOrder]:
        """Borsaya emir göndermeden açık pozisyonu lokalde kapatır (OCO/Futures tetiklemeleri için)."""
        symbol = position.symbol
        amount = position.amount

        commission_rate = self.settings.paper_trading.commission_rate

        # Yönlü PnL
        if position.side == PositionSide.LONG or position.side == "long":
            raw_pnl = (exit_price - position.entry_price) * amount
            close_side = "sell"
        else:
            raw_pnl = (position.entry_price - exit_price) * amount
            close_side = "buy"

        fee_usdt = (amount * exit_price) * commission_rate
        pnl_usdt = raw_pnl - fee_usdt
        pnl_pct = pnl_usdt / position.cost_usdt if position.cost_usdt > 0 else 0.0

        order_id = str(uuid.uuid4())[:8]

        # Lokal bakiyeyi güncelle
        self.usdt_balance += (position.cost_usdt + pnl_usdt)

        if self.risk_engine:
            self.risk_engine.set_balance(self.usdt_balance)
            if pnl_usdt < 0:
                self.risk_engine.record_loss(abs(pnl_usdt))

        order = TradeOrder(
            order_id=order_id, symbol=symbol, order_type=OrderType.MARKET,
            side=close_side, price=exit_price, amount=amount, status=OrderStatus.FILLED,
            fee_usdt=fee_usdt,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            close_reason=reason,
            pnl_usdt=round(pnl_usdt, 2),
            pnl_pct=round(pnl_pct, 4),
            indicator_summary=position.indicator_summary,
        )
        self.orders.append(order)
        del self.open_positions[symbol]
        self.save_state()
        pnl_sign = "+" if pnl_usdt >= 0 else ""
        self._send_portfolio_summary(f"Pozisyon Kapatıldı ({reason}): {symbol} | PnL: {pnl_sign}${pnl_usdt:.2f} ({pnl_pct*100:+.2f}%)")

        pnl_emoji = "🟢 KAR" if pnl_usdt >= 0 else "🔴 ZARAR"
        logger.info(
            f"📥 [LOKAL-KAPAT] Pozisyon borsada kapandığı için lokalde de kapatıldı ({reason}): {symbol} | "
            f"Cikis: ${exit_price:,.4f} | "
            f"PnL: {pnl_emoji} ${pnl_usdt:,.2f} (%{pnl_pct*100:.2f})"
        )

        self._run_ai_postmortem(position, exit_price, pnl_usdt, pnl_pct, reason)
        return order

    def _create_futures_protection_orders(
        self,
        symbol: str,
        position: MarketPosition,
        sl_price: float,
        tp_price: float,
    ) -> None:
        """Vadeli İşlemler piyasasında Stop-Loss ve Take-Profit tetiklemeli koruma emirlerini oluşturur."""
        if not self.exchange:
            return

        side = "sell" if position.side == PositionSide.LONG or position.side == "long" else "buy"
        amount_prec = self.exchange.amount_to_precision(symbol, position.amount)

        # 1. Stop-Loss (STOP_MARKET)
        if sl_price > 0:
            try:
                sl_price_prec = self.exchange.price_to_precision(symbol, sl_price)
                logger.info(f"🛡️ [CANLI] Futures Stop-Loss emri gönderiliyor: {symbol} @ {sl_price}")
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=side,
                    amount=float(amount_prec),
                    price=None,
                    params={
                        'stopPrice': float(sl_price_prec),
                        'reduceOnly': True,
                        'closePosition': True
                    }
                )
                position.oco_stop_order_id = str(sl_order.get('id'))
                logger.info(f"🛡️ [CANLI] Futures Stop-Loss başarıyla kuruldu. ID: {position.oco_stop_order_id}")
            except Exception as e:
                logger.error(f"❌ [CANLI] Futures Stop-Loss emri gönderilemedi: {e}")

        # 2. Take-Profit (TAKE_PROFIT_MARKET)
        if tp_price > 0:
            try:
                tp_price_prec = self.exchange.price_to_precision(symbol, tp_price)
                logger.info(f"🎯 [CANLI] Futures Take-Profit emri gönderiliyor: {symbol} @ {tp_price}")
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=side,
                    amount=float(amount_prec),
                    price=None,
                    params={
                        'stopPrice': float(tp_price_prec),
                        'reduceOnly': True,
                        'closePosition': True
                    }
                )
                position.oco_limit_order_id = str(tp_order.get('id'))
                logger.info(f"🎯 [CANLI] Futures Take-Profit başarıyla kuruldu. ID: {position.oco_limit_order_id}")
            except Exception as e:
                logger.error(f"❌ [CANLI] Futures Take-Profit emri gönderilemedi: {e}")

    # ── Periyodik SL/TP + Trailing Stop Kontrolü ──────────────

    def monitor_positions(self, current_prices: Dict[str, float]) -> List[TradeOrder]:
        """Açık pozisyonları güncel fiyatlarla tarayarak SL/TP kontrolleri yapar."""
        closed_orders: List[TradeOrder] = []
        state_changed = False
        active_symbols = list(self.open_positions.keys())

        limit_timeout_min = getattr(self.settings.execution, "limit_timeout_minutes", 15)
        timeout_seconds = limit_timeout_min * 60
        is_futures = getattr(self.settings.exchange, "default_type", "spot") == "future"

        for sym in active_symbols:
            pos = self.open_positions.get(sym)
            if pos is None:
                continue

            # ── 1. BEKLEYEN LİMİT EMİRLERİNİN KONTROLÜ ──
            if pos.status == "pending":
                # Zaman aşımı kontrolü
                elapsed_sec = (datetime.utcnow() - pos.opened_at).total_seconds()
                if elapsed_sec > timeout_seconds:
                    if self.is_paper_trade:
                        # Bakiyeyi iade et
                        commission_rate = self.settings.paper_trading.commission_rate
                        refund_usdt = pos.cost_usdt * (1.0 + commission_rate)
                        self.usdt_balance += refund_usdt
                        if self.risk_engine:
                            self.risk_engine.set_balance(self.usdt_balance)
                        
                        # Sipariş geçmişini güncelle
                        for o in self.orders:
                            if o.symbol == sym and o.status == OrderStatus.PENDING:
                                o.status = OrderStatus.CANCELLED
                                break
                        
                        logger.info(f"⏰ [PAPER-LIMIT] Limit emri zaman asimina ugradi ve iptal edildi: {sym}")
                        del self.open_positions[sym]
                        self.save_state()
                        continue
                    else:
                        # Canlı modda borsadan iptal et
                        try:
                            logger.info(f"⏰ [CANLI-LIMIT] Limit emri zaman asimina ugradi. Borsadan iptal ediliyor: {sym} | ID: {pos.limit_order_id}")
                            self.exchange.cancel_order(pos.limit_order_id, sym)
                            
                            for o in self.orders:
                                if o.order_id == pos.limit_order_id:
                                    o.status = OrderStatus.CANCELLED
                                    break
                            
                            del self.open_positions[sym]
                            self.save_state()
                            continue
                        except Exception as e:
                            logger.error(f"⚠️ [CANLI-LIMIT] Limit emri iptal edilirken hata ({sym}): {e}")
                
                # Gerçekleşme (Fill) kontrolü
                if self.is_paper_trade:
                    if sym in current_prices:
                        price = current_prices[sym]
                        is_filled = False
                        if pos.side == PositionSide.LONG or pos.side == "long":
                            is_filled = price <= pos.entry_price
                        else:
                            is_filled = price >= pos.entry_price

                        if is_filled:
                            # Emir gerçekleşti!
                            pos.status = "active"
                            pos.opened_at = datetime.utcnow()
                            pos.update_pnl(price)
                            
                            for o in self.orders:
                                if o.symbol == sym and o.status == OrderStatus.PENDING:
                                    o.status = OrderStatus.FILLED
                                    break
                            
                            logger.info(
                                f"🟢 [PAPER-LIMIT] Limit emri gerceklesti ({pos.side.value.upper()}): {sym} | "
                                f"Fiyat: ${price:,.4f} | Maliyet: ${pos.cost_usdt:,.2f}"
                            )
                            state_changed = True
                            self._send_portfolio_summary(f"Limit Emir Gerçekleşti ({pos.side.value.upper()}): {sym} @ {price:,.4f}")
                            # ── YENİ: Fill olan pozisyon için hemen SL/TP kontrol et ──
                            # continue kaldırıldı; aşağıdaki SL/TP bloğuna düşüyor
                        else:
                            # Henüz gerçekleşmedi, bu sembolü atla
                            continue
                    else:
                        continue
                else:
                    # Canlı modda borsa durumunu sorgula
                    if pos.limit_order_id:
                        try:
                            o_info = self.exchange.fetch_order(pos.limit_order_id, sym)
                            ccxt_status = o_info.get("status")
                            
                            if ccxt_status == "closed":
                                pos.status = "active"
                                pos.opened_at = datetime.utcnow()
                                
                                for o in self.orders:
                                    if o.order_id == pos.limit_order_id:
                                        o.status = OrderStatus.FILLED
                                        break
                                
                                logger.info(f"🟢 [CANLI-LIMIT] Limit emri borsada DOLDU: {sym}. Korumalar kuruluyor...")
                                self._send_portfolio_summary(f"Limit Emir Gerçekleşti ({pos.side.value.upper()}): {sym} @ {pos.entry_price:,.4f}")
                                
                                # Korumaları kur
                                if pos.stop_loss > 0 and pos.take_profit > 0:
                                    if is_futures:
                                        self._create_futures_protection_orders(sym, pos, pos.stop_loss, pos.take_profit)
                                    else:
                                        try:
                                            stop_loss_price = pos.stop_loss
                                            take_profit_price = pos.take_profit
                                            stop_limit_price = stop_loss_price * 0.99
                                            oco_res = self.exchange.privatePostOrderOco({
                                                'symbol': sym.replace('/', ''),
                                                'side': 'SELL',
                                                'quantity': self.exchange.amount_to_precision(sym, pos.amount),
                                                'price': self.exchange.price_to_precision(sym, take_profit_price),
                                                'stopPrice': self.exchange.price_to_precision(sym, stop_loss_price),
                                                'stopLimitPrice': self.exchange.price_to_precision(sym, stop_limit_price),
                                                'stopLimitTimeInForce': 'GTC'
                                            })
                                            pos.oco_order_list_id = str(oco_res.get('orderListId'))
                                            for o in oco_res.get('orders', []):
                                                if o.get('type') == 'LIMIT_MAKER':
                                                    pos.oco_limit_order_id = str(o.get('orderId'))
                                                elif o.get('type') == 'STOP_LOSS_LIMIT':
                                                    pos.oco_stop_order_id = str(o.get('orderId'))
                                            logger.info(f"🔥 [CANLI-LIMIT] OCO Emri Olusturuldu: listId={pos.oco_order_list_id} | TP ID={pos.oco_limit_order_id} | SL ID={pos.oco_stop_order_id}")
                                        except Exception as e:
                                            logger.error(f"❌ [CANLI-LIMIT] OCO Emri Gönderilemedi: {e}", exc_info=True)
                                
                                self.save_state()
                            elif ccxt_status == "canceled":
                                logger.warning(f"⚠️ [CANLI-LIMIT] Limit emri borsada iptal edilmis: {sym}")
                                for o in self.orders:
                                    if o.order_id == pos.limit_order_id:
                                        o.status = OrderStatus.CANCELLED
                                        break
                                del self.open_positions[sym]
                                self.save_state()
                                continue
                        except Exception as e:
                            logger.error(f"⚠️ [CANLI-LIMIT] Limit emri durum sorgulamasi basarisiz ({sym}): {e}")
                    continue

            # ── 2. CANLI MOD: KORUMA EMİRLERİ (OCO veya Futures Ayrı SL/TP) KONTROLÜ ──
            if not self.is_paper_trade and self.exchange and (pos.oco_order_list_id or pos.oco_limit_order_id or pos.oco_stop_order_id):
                try:
                    limit_closed = False
                    stop_closed = False
                    
                    if pos.oco_limit_order_id:
                        o_lim = self.exchange.fetch_order(pos.oco_limit_order_id, sym)
                        if o_lim.get('status') == 'closed':
                            limit_closed = True
                            
                    if pos.oco_stop_order_id and not limit_closed:
                        o_stop = self.exchange.fetch_order(pos.oco_stop_order_id, sym)
                        if o_stop.get('status') == 'closed':
                            stop_closed = True
                            
                    if limit_closed:
                        logger.info(f"🎉 [CANLI] Koruma (Take Profit) borsada DOLDU: {sym}")
                        order = self._close_position_locally(pos, pos.take_profit, "TAKE PROFIT TRIGGERED")
                        if order:
                            closed_orders.append(order)
                        # Futures için diğer açık emri iptal et
                        if is_futures and pos.oco_stop_order_id:
                            try:
                                self.exchange.cancel_order(pos.oco_stop_order_id, sym)
                            except Exception as cancel_err:
                                logger.debug(f"Stop-Loss iptal edilemedi: {cancel_err}")
                        continue
                    elif stop_closed:
                        logger.info(f"🚨 [CANLI] Koruma (Stop Loss) borsada DOLDU: {sym}")
                        trigger_price = o_stop.get('price') or pos.stop_loss
                        order = self._close_position_locally(pos, trigger_price, "STOP LOSS TRIGGERED")
                        if order:
                            closed_orders.append(order)
                        # Futures için diğer açık emri iptal et
                        if is_futures and pos.oco_limit_order_id:
                            try:
                                self.exchange.cancel_order(pos.oco_limit_order_id, sym)
                            except Exception as cancel_err:
                                logger.debug(f"Take-Profit iptal edilemedi: {cancel_err}")
                        continue
                except Exception as e:
                    logger.error(f"⚠️ [CANLI] Koruma durum kontrolü başarısız ({sym}): {e}")

            # ── PAPER TRADE veya MANUAL DURUMLAR: Lokal Fiyat Kontrolü ───
            # Not: pending→active geçişi yukarıda yaptıysa price zaten set edildi.
            # Aktif pozisyon için güncel fiyatı kullan.
            if sym not in current_prices:
                continue

            price = current_prices[sym]
            pos.update_pnl(price)
            state_changed = True

            # Likidasyon Kontrolü (Paper Trade için)
            if self.is_paper_trade:
                leverage_val = getattr(self.settings.execution, "leverage", 1)
                if leverage_val > 1:
                    liq_price = pos.get_liquidation_price(leverage_val)
                    is_liquidated = False
                    if pos.side == PositionSide.LONG or pos.side == "long":
                        is_liquidated = price <= liq_price
                    else:
                        is_liquidated = price >= liq_price

                    if is_liquidated:
                        logger.warning(
                            f"💥 [{sym}] LİKİDASYON gerçekleşti! "
                            f"Fiyat: ${price:,.4f} | Likidasyon Fiyatı: ${liq_price:,.4f} | Yön: {pos.side.value.upper()}"
                        )
                        order = self._close_position(pos, price, "LIQUIDATION")
                        if order:
                            closed_orders.append(order)
                        state_changed = False
                        continue

            # ── STRATEJİ V2 ÇIKIŞ FİLTRELERİ ──
            if self.settings.strategy.version == "v2":
                # E5: Zaman Bazlı Çıkış (48 saat sınırı)
                elapsed_hours = (datetime.utcnow() - pos.opened_at).total_seconds() / 3600
                if elapsed_hours >= 48:
                    logger.info(f"⏰ [TIME EXIT] {sym} 48 saati doldurdu ({elapsed_hours:.1f} saat). Piyasa fiyatından kapatılıyor...")
                    order = self._close_position(pos, price, "TIME-BASED EXIT TRIGGERED")
                    if order:
                        closed_orders.append(order)
                    state_changed = False  # _close_position zaten save_state çağırır
                    continue

                # E4: Trailing Stop (Breakeven - başa baş noktası)
                # Orijinal ATR'yi giriş fiyatı ile stop_loss farkından geri hesapla (Stop = Giriş - 2*ATR)
                original_sl = pos.stop_loss
                if pos.side == PositionSide.LONG or pos.side == "long":
                    atr_val = (pos.entry_price - original_sl) / 2.0
                    if atr_val > 0 and original_sl < pos.entry_price and price >= pos.entry_price + atr_val:
                        pos.stop_loss = pos.entry_price
                        state_changed = True
                        logger.info(f"🛡️ [BREAKEVEN] {sym} +1x ATR kâra ulaştı. Stop-loss giriş fiyatına ({pos.entry_price}) çekildi.")
                else:
                    atr_val = (original_sl - pos.entry_price) / 2.0
                    if atr_val > 0 and original_sl > pos.entry_price and price <= pos.entry_price - atr_val:
                        pos.stop_loss = pos.entry_price
                        state_changed = True
                        logger.info(f"🛡️ [BREAKEVEN] {sym} +1x ATR kâra ulaştı. Stop-loss giriş fiyatına ({pos.entry_price}) çekildi.")

            # ── V3 Price Action Kısmi Kâr Alma ve BE Çekme ──
            if self.settings.strategy.version == "v3":
                if pos.stop_loss != pos.entry_price: # Stage 1 check (stop not yet moved to entry)
                    if pos.side == PositionSide.LONG or pos.side == "long":
                        risk = pos.entry_price - pos.stop_loss
                        if risk > 0:
                            partial_tp_target = pos.entry_price + (1.5 * risk)
                            if price >= partial_tp_target:
                                original_amount = pos.amount
                                close_amount = original_amount * 0.5
                                pos.amount = original_amount - close_amount
                                pos.cost_usdt = pos.amount * pos.entry_price
                                
                                raw_pnl = (partial_tp_target - pos.entry_price) * close_amount
                                commission_rate = self.settings.paper_trading.commission_rate
                                fee_usdt = (close_amount * partial_tp_target) * commission_rate
                                net_pnl_usdt = raw_pnl - fee_usdt
                                
                                self.usdt_balance += (close_amount * pos.entry_price + net_pnl_usdt)
                                if self.risk_engine:
                                    self.risk_engine.set_balance(self.usdt_balance)
                                    
                                pos.stop_loss = pos.entry_price
                                state_changed = True
                                
                                import uuid
                                order_id = str(uuid.uuid4())[:8]
                                order = TradeOrder(
                                    order_id=order_id, symbol=sym, order_type=OrderType.MARKET,
                                    side="sell", price=partial_tp_target, amount=close_amount, status=OrderStatus.FILLED,
                                    fee_usdt=fee_usdt, stop_loss=pos.entry_price, take_profit=pos.take_profit,
                                    close_reason="PARTIAL TAKE PROFIT (50%)", pnl_usdt=round(net_pnl_usdt, 2),
                                    pnl_pct=round(net_pnl_usdt / (close_amount * pos.entry_price), 4),
                                    indicator_summary=pos.indicator_summary,
                                )
                                self.orders.append(order)
                                closed_orders.append(order)
                                
                                logger.info(
                                    f"🛡️ [V3 PARTIAL TP] {sym} 1.5 R kâra ulaştı. "
                                    f"Pozisyonun %50'si kapatıldı (Miktar: {close_amount:.6f} @ ${partial_tp_target:.4f}). "
                                    f"Stop-loss Giriş fiyatına ({pos.entry_price}) çekildi."
                                )
                                self._send_portfolio_summary(f"Kısmi TP Alındı (%50) & BE Çekildi: {sym} @ {partial_tp_target:,.4f}")
                                
                    else: # SHORT
                        risk = pos.stop_loss - pos.entry_price
                        if risk > 0:
                            partial_tp_target = pos.entry_price - (1.5 * risk)
                            if price <= partial_tp_target:
                                original_amount = pos.amount
                                close_amount = original_amount * 0.5
                                pos.amount = original_amount - close_amount
                                pos.cost_usdt = pos.amount * pos.entry_price
                                
                                raw_pnl = (pos.entry_price - partial_tp_target) * close_amount
                                commission_rate = self.settings.paper_trading.commission_rate
                                fee_usdt = (close_amount * partial_tp_target) * commission_rate
                                net_pnl_usdt = raw_pnl - fee_usdt
                                
                                self.usdt_balance += (close_amount * pos.entry_price + net_pnl_usdt)
                                if self.risk_engine:
                                    self.risk_engine.set_balance(self.usdt_balance)
                                    
                                pos.stop_loss = pos.entry_price
                                state_changed = True
                                
                                import uuid
                                order_id = str(uuid.uuid4())[:8]
                                order = TradeOrder(
                                    order_id=order_id, symbol=sym, order_type=OrderType.MARKET,
                                    side="buy", price=partial_tp_target, amount=close_amount, status=OrderStatus.FILLED,
                                    fee_usdt=fee_usdt, stop_loss=pos.entry_price, take_profit=pos.take_profit,
                                    close_reason="PARTIAL TAKE PROFIT (50%)", pnl_usdt=round(net_pnl_usdt, 2),
                                    pnl_pct=round(net_pnl_usdt / (close_amount * pos.entry_price), 4),
                                    indicator_summary=pos.indicator_summary,
                                )
                                self.orders.append(order)
                                closed_orders.append(order)
                                
                                logger.info(
                                    f"🛡️ [V3 PARTIAL TP] {sym} 1.5 R kâra ulaştı. "
                                    f"Pozisyonun %50'si kapatıldı (Miktar: {close_amount:.6f} @ ${partial_tp_target:.4f}). "
                                    f"Stop-loss Giriş fiyatına ({pos.entry_price}) çekildi."
                                )
                                self._send_portfolio_summary(f"Kısmi TP Alındı (%50) & BE Çekildi: {sym} @ {partial_tp_target:,.4f}")

            # Trailing stop seviyesini hesapla
            active_sl = pos.get_trailing_stop_level()

            # 1. Zarar Durdur (Stop-Loss / Trailing Stop) Kontrolü
            is_sl_triggered = False
            if active_sl > 0:
                if pos.side == PositionSide.LONG or pos.side == "long":
                    is_sl_triggered = price <= active_sl
                else:
                    is_sl_triggered = price >= active_sl

            if is_sl_triggered:
                sl_type = "TRAILING STOP" if pos.trailing_stop_active and ((pos.side in (PositionSide.LONG, "long") and active_sl > pos.stop_loss) or (pos.side in (PositionSide.SHORT, "short") and active_sl < pos.stop_loss)) else "STOP LOSS"
                logger.warning(
                    f"🚨 [{sym}] {sl_type} tetiklendi! "
                    f"Seviye: ${active_sl:,.4f} | Fiyat: ${price:,.4f} | "
                    f"PnL: ${pos.pnl_usdt:+,.2f}"
                )
                order = self._close_position(pos, price, f"{sl_type} TRIGGERED")
                if order:
                    closed_orders.append(order)
                state_changed = False  # _close_position zaten save_state çağırır
                continue

            # 2. Kar Al (Take-Profit) Kontrolü
            is_tp_triggered = False
            if pos.take_profit > 0:
                if pos.side == PositionSide.LONG or pos.side == "long":
                    is_tp_triggered = price >= pos.take_profit
                else:
                    is_tp_triggered = price <= pos.take_profit

            if is_tp_triggered:
                logger.info(
                    f"🎉 [{sym}] TAKE PROFIT tetiklendi! "
                    f"Seviye: ${pos.take_profit:,.4f} | Fiyat: ${price:,.4f} | "
                    f"PnL: ${pos.pnl_usdt:+,.2f}"
                )
                order = self._close_position(pos, price, "TAKE PROFIT TRIGGERED")
                if order:
                    closed_orders.append(order)
                state_changed = False  # _close_position zaten save_state çağırır
                continue

        # Fiyat güncellendi ama SL/TP tetiklenmedi → durum dosyasını güncelle
        if state_changed:
            self.save_state()

        return closed_orders

    def _send_portfolio_summary(self, trigger_message: str) -> None:
        try:
            from src.utils.telegram_notifier import send_telegram_notification
            
            positions = self.open_positions
            balance = self.usdt_balance
            version = self.settings.strategy.version.upper()
            
            msg_lines = []
            msg_lines.append(f"🔔 <b>[{version}] {trigger_message}</b>\n")
            msg_lines.append(f"<b>Boştaki Nakit:</b> ${balance:,.2f} USDT")
            msg_lines.append(f"<b>Başlangıç Sermayesi:</b> $10,000.00 USDT")
            msg_lines.append(f"<b>Toplam Net Kâr:</b> +${balance - 10000.0:,.2f} USDT")
            msg_lines.append(f"<b>Açık Pozisyon Sayısı:</b> {len(positions)} adet\n")

            total_pnl = 0.0
            total_cost = 0.0

            if positions:
                msg_lines.append("<b>Açık Pozisyon Detayları:</b>\n")
                for sym, pos in positions.items():
                    side = pos.side.value.upper() if hasattr(pos.side, 'value') else str(pos.side).upper()
                    entry = pos.entry_price
                    current = pos.current_price
                    pnl = pos.pnl_usdt
                    pnl_pct = pos.pnl_pct * 100
                    cost = pos.cost_usdt
                    sl = pos.stop_loss
                    tp = pos.take_profit
                    
                    total_pnl += pnl
                    total_cost += cost
                    
                    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                    side_emoji = "🟢" if side == "LONG" else "🔴"
                    
                    risked_usdt = abs(entry - sl) * pos.amount
                    
                    msg_lines.append(
                        f"• <b>{sym}</b> ({side_emoji} {side})\n"
                        f"  Giriş: ${entry:,.4f} | Güncel: ${current:,.4f}\n"
                        f"  SL: ${sl:,.4f} | TP: ${tp:,.4f}\n"
                        f"  Maliyet: ${cost:,.2f} USDT | <b>Riske Atılan:</b> ${risked_usdt:,.2f} USDT\n"
                        f"  PnL: {pnl_emoji} ${pnl:+.2f} ({pnl_pct:+.2f}%)\n"
                    )
                
                total_value = balance + total_cost + total_pnl
                msg_lines.append("--------------------------------------")
                msg_lines.append(f"<b>Toplam Pozisyon Maliyeti:</b> ${total_cost:,.2f} USDT")
                pnl_total_emoji = "🟢" if total_pnl >= 0 else "🔴"
                msg_lines.append(f"<b>Toplam Anlık PnL:</b> {pnl_total_emoji} ${total_pnl:+.2f}")
                msg_lines.append(f"<b>Toplam Portföy Değeri:</b> ${total_value:,.2f} USDT")
            else:
                msg_lines.append("<i>Şu anda açık pozisyon bulunmuyor.</i>")

            message = "\n".join(msg_lines)
            send_telegram_notification(message)
        except Exception as e:
            logger.error(f"Error sending portfolio summary to Telegram: {e}")
