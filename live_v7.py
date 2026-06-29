# -*- coding: utf-8 -*-
"""
live_v7.py — V7 Price Action LIVE Trading Bot
Backtest Optimized: 100 coin, %114 ROI, %26.6 win rate.

Çalıştırma:
  python live_v7.py                    # Paper trade modu
  python live_v7.py --live             # Canlı mod
  python live_v7.py --single-run       # Tek seferlik tarama
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml

from src.data.historical import HistoricalDataFetcher
from src.strategy.v7_pa_strategy import V7PriceActionStrategy
from src.strategy.adaptive_learner import AdaptiveLearner
from src.strategy.regime_detector import RegimeDetector, MarketRegime
from src.strategy.ai_chart_verifier import AIChartVerifier
from src.backtest.engine import BacktestEngine, TradeRecord
from src.risk.engine import RiskEngine
from src.utils.telegram_notifier import send_telegram_notification
from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.utils.telegram_listener import start_telegram_listener

logger = get_logger("live_v7")


def format_price(price: float) -> str:
    if price is None or price == 0:
        return "-"
    if price >= 100:
        return f"${price:,.2f}"
    elif price >= 1.0:
        return f"${price:,.4f}"
    elif price >= 0.0001:
        return f"${price:,.6f}"
    else:
        return f"${price:,.8f}"


class LiveV7Bot:
    """V7 Price Action canlı trading botu — backtest optimized."""

    MAJOR_PAIRS = {"BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"}
    EXCLUDED_SYMBOLS = {"PAXG/USDT", "RLUSD/USDT", "FDUSD/USDT", "USDC/USDT", "HMSTR/USDT"}
    SECTORS = {
        "l1": {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT", "NEAR/USDT", "DOT/USDT", "ADA/USDT"},
        "l2": {"ARB/USDT", "OP/USDT", "MATIC/USDT", "IMX/USDT"},
        "defi": {"UNI/USDT", "AAVE/USDT", "LINK/USDT", "LDO/USDT", "MKR/USDT"},
        "meme": {"DOGE/USDT", "PEPE/USDT", "SHIB/USDT", "LUNC/USDT"},
        "ai": {"FET/USDT", "RENDER/USDT", "TAO/USDT"},
    }

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_positions: int = 10,
        position_pct: float = 0.02,
        is_live: bool = False,
        top_n: int = 200,
    ) -> None:
        self.portfolio_state_path = "logs/portfolio_state_v7.json"
        self.signals_jsonl_path = "logs/signals_v7.jsonl"
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_pct = position_pct
        self.is_live = is_live
        self.top_n = top_n

        # Config'den strateji parametrelerini yükle
        cfg = self._load_config()
        strat_cfg = cfg.get("strategy", {})
        risk_cfg = cfg.get("risk", {})
        exec_cfg = cfg.get("execution", {})

        self.fetcher = HistoricalDataFetcher()
        self.strategy = V7PriceActionStrategy(
            sweep_window=strat_cfg.get("sweep_window", 100),
            max_hold_sweep=strat_cfg.get("max_hold_sweep", 7),
            target_rr=strat_cfg.get("target_rr", 3.0),
            trend_ema=strat_cfg.get("trend_ema", 180),
            atr_multiplier=strat_cfg.get("atr_multiplier", 0.6),
            use_volume_filter=strat_cfg.get("use_volume_filter", True),
            volume_threshold=strat_cfg.get("volume_threshold", 0.5),
            min_volatility_pct=strat_cfg.get("min_volatility_pct", 0.3),
            use_premium_discount=strat_cfg.get("use_premium_discount", True),
            max_tp_pct=strat_cfg.get("max_tp_pct", 0.50),
            min_sl_pct=strat_cfg.get("min_sl_pct", 0.02),
            max_sl_pct=strat_cfg.get("max_sl_pct", 0.08),
        )
        self.regime_detector = RegimeDetector()
        self.balance = initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[TradeRecord] = []
        self.cycle_events: List[str] = []

        # Config'den risk parametrelerini yükle
        self.max_daily_drawdown_pct = risk_cfg.get("max_daily_drawdown_pct", 0.05)
        self.min_risk_reward_ratio = risk_cfg.get("min_risk_reward_ratio", 5.5)
        self.leverage = exec_cfg.get("leverage", 5)
        self.commission_rate = cfg.get("paper_trading", {}).get("commission_rate", 0.00063)

        # RiskEngine
        self.risk_engine = RiskEngine(initial_balance=initial_capital)

        # Cooldown
        self._cooldown_map: Dict[str, dict] = {}
        self.COOLDOWN_SL_HOURS = 24
        self._consecutive_losses: Dict[str, int] = {}

        # Win rate takibi (pozisyon boyutlandırma için)
        self._symbol_stats: Dict[str, Dict] = {}  # {sym: {"wins": int, "losses": int, "total_pnl": float}}

        self._load_state()
        
        # Adaptif öğrenme
        self.learner = AdaptiveLearner()
        self._last_daily_optimize = None
        
        # AI Grafik Doğrulayıcı (Xiaomi MiMo v2.5)
        self.ai_verifier = AIChartVerifier()
        
        # MiMo pozisyon inceleme — her 1 saatte bir
        self._last_ai_review_time = None
        self._ai_review_interval_min = 60   # dakika cinsinden
        
        # Settings (telegram_listener için gerekli)
        self.settings = get_settings()
        self.settings.strategy.version = "v7"

        logger.info(f"LiveV7Bot başlatıldı | Sermaye: ${initial_capital:,.2f} | {'LIVE' if is_live else 'PAPER'} | Top {top_n}")
        logger.info(f"Config yüklendi | sweep_window={self.strategy.sweep_window} | max_hold={self.strategy.max_hold_sweep} | target_rr={self.strategy.target_rr} | ema={self.strategy.trend_ema} | leverage={self.leverage}")

        mod = '🔴 CANLI' if is_live else '🟢 PAPER'
        version = self.learner.get_version()
        msg = (
            f"🤖 V{version} Price Action Bot Başlatıldı\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Sermaye: ${initial_capital:,.2f}\n"
            f"📊 Mod: {mod}\n"
            f"🎯 Max Pozisyon: {max_positions}\n"
            f"📏 Risk/Trade: %{position_pct*100:.0f}\n"
            f"📋 Semboller: Top {top_n} (hacme göre)\n"
            f"🛡️ RR Hedefi: {self.strategy.target_rr} | ATR Stop: {self.strategy.atr_multiplier}x\n"
            f"📈 Backtest ROI: %+114.6\n"
            f"🧠 Günlük Otomatik Öğrenme: Aktif"
        )
        send_telegram_notification(msg)

        # Listener run_all_bots.py tarafindan tek seferlik baslatilir

    def _load_config(self) -> dict:
        """config_v7.yaml dosyasından parametreleri yükler."""
        config_path = Path("config_v7.yaml")
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Config yüklenemedi: {e}")
        return {}

    def _send_portfolio_summary(self, trigger_message: str = "") -> None:
        self._print_status()

    # ═══════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def _load_state(self) -> None:
        try:
            path = Path(self.portfolio_state_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.balance = state.get("balance", self.initial_capital)
                self.positions = state.get("positions", {})
                self._symbol_stats = state.get("symbol_stats", {})
                logger.info(f"State yüklendi: ${self.balance:,.2f} bakiye, {len(self.positions)} pozisyon")

            learner_stats = self.learner.state.get("symbol_stats", {})
            for sym, ls in learner_stats.items():
                if sym not in self._symbol_stats:
                    self._symbol_stats[sym] = {
                        "wins": ls.get("wins", 0),
                        "losses": ls.get("losses", 0),
                        "total_pnl": ls.get("total_pnl", 0.0),
                    }
                else:
                    ps = self._symbol_stats[sym]
                    ls_wins = ls.get("wins", 0)
                    ls_losses = ls.get("losses", 0)
                    if ls_wins + ls_losses > ps.get("wins", 0) + ps.get("losses", 0):
                        self._symbol_stats[sym] = {
                            "wins": ls_wins,
                            "losses": ls_losses,
                            "total_pnl": ls.get("total_pnl", 0.0),
                        }
        except Exception as e:
            logger.warning(f"State yüklenemedi: {e}")

    def _save_state(self) -> None:
        try:
            Path("logs").mkdir(exist_ok=True)
            state = {
                "balance": self.balance,
                "positions": self.positions,
                "symbol_stats": self._symbol_stats,
                "updated_at": pd.Timestamp.now().isoformat(),
            }
            with open(self.portfolio_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"State kayıt hatası: {e}")

    # ═══════════════════════════════════════════════════════════
    # COOLDOWN
    # ═══════════════════════════════════════════════════════════

    def _is_in_cooldown(self, symbol: str) -> Tuple[bool, str]:
        if symbol not in self._cooldown_map:
            return False, ""
        cd = self._cooldown_map[symbol]
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        if now < cd["until"]:
            remaining = (cd["until"] - now).total_seconds() / 3600
            return True, f"Cooldown: {cd['reason']} ({remaining:.1f}h kaldı)"
        if now >= cd["until"]:
            del self._cooldown_map[symbol]
        return False, ""

    def _set_cooldown(self, symbol: str, reason: str, hours: float) -> None:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        self._cooldown_map[symbol] = {"until": now + timedelta(hours=hours), "reason": reason}

    def _track_consecutive_loss(self, symbol: str, pnl: float) -> None:
        if pnl <= 0:
            self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
        else:
            self._consecutive_losses[symbol] = 0

    def _get_consecutive_loss_multiplier(self, symbol: str) -> float:
        losses = self._consecutive_losses.get(symbol, 0)
        if losses >= 3:
            return 0.25
        elif losses >= 2:
            return 0.50
        return 1.0

    def _get_dynamic_rr(self, symbol: str) -> float:
        try:
            df_4h = self.fetcher.fetch_ohlcv(symbol, "4h", limit=50)
            if df_4h.empty or len(df_4h) < 20:
                return self.strategy.target_rr
            close = df_4h["close"].astype(float)
            high = df_4h["high"].astype(float)
            low = df_4h["low"].astype(float)
            tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
            plus_di = 100 * plus_dm.rolling(14).mean() / atr
            minus_di = 100 * minus_dm.rolling(14).mean() / atr
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.rolling(14).mean()
            current_adx = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0
            if current_adx > 25:
                return 4.0
            elif current_adx > 15:
                return 3.0
            else:
                return 0
        except Exception:
            return self.strategy.target_rr

    # ═══════════════════════════════════════════════════════════
    # POZİSYON BOYUTLANDIRMA (Win Rate Bazlı)
    # ═══════════════════════════════════════════════════════════

    def _get_position_size_multiplier(self, symbol: str) -> float:
        """Win rate'e göre pozisyon boyutu çarpanı."""
        stats = self._symbol_stats.get(symbol, {"wins": 0, "losses": 0})
        total = stats["wins"] + stats["losses"]
        if total < 5:
            return 1.0  # Yeterli veri yok, normal boyut
        wr = stats["wins"] / total
        if wr > 0.30:
            return 1.5  # Yüksek win rate → daha büyük
        elif wr > 0.20:
            return 1.0  # Normal
        else:
            return 0.5  # Düşük win rate → daha küçük

    # ═══════════════════════════════════════════════════════════
    # FİLTRELER
    # ═══════════════════════════════════════════════════════════

    def _check_symbol_quality(self, symbol: str) -> Tuple[bool, str]:
        if symbol in self.EXCLUDED_SYMBOLS:
            return False, f"Hariç tutulan: {symbol}"
        if self.learner.is_symbol_blacklisted(symbol):
            return False, f"Öğrenme kara listesinde: {symbol}"
        try:
            ticker = self.fetcher.exchange.fetch_ticker(symbol)
            last_price = float(ticker.get("last", 0))
            if last_price > 0 and last_price < 0.01:
                return False, f"Cok dusuk fiyat: ${last_price:.6f} (<$0.01)"
        except Exception:
            pass
        # ── Yeni listelenen coin filtresi (hafıza kontrolü) ──────────────
        # Botun kendisi kontrol eder — yapay zekaya sormaya gerek yok
        try:
            df_daily = self.fetcher.fetch_ohlcv(symbol, "1d", limit=60)
            if df_daily.empty or len(df_daily) < 30:
                return False, f"Yetersiz geçmiş: yalnızca {len(df_daily)} günlük veri — yeni listeleme olabilir"
        except Exception:
            pass   # veri çekilemezse geç, sonraki filtrelere bırak
        # Volatilite kontrolü
        try:
            df = self.fetcher.fetch_ohlcv(symbol, "1h", limit=50)
            if df.empty or len(df) < 20:
                return False, "Yeterli veri yok"
            volatility = df["close"].pct_change(20).abs().iloc[-1] * 100
            if pd.isna(volatility) or volatility < 0.3:
                return False, f"Volatilite çok düşük: %{volatility:.2f}"
        except Exception:
            pass
        return True, ""

    def _check_sector_limit(self, symbol: str) -> Tuple[bool, str]:
        for sector, syms in self.SECTORS.items():
            if symbol in syms:
                count = sum(1 for s in self.positions if s in syms)
                if count >= 2:
                    return False, f"Sektör limiti: {sector}'da {count} pozisyon"
        return True, ""

    def _check_regime(self, symbol: str, signal: str) -> Tuple[bool, str, float]:
        try:
            df_4h = self.fetcher.fetch_ohlcv(symbol, "4h", limit=200)
            if df_4h.empty or len(df_4h) < 50:
                return True, "", 1.0
            df_4h = self.regime_detector.calculate_indicators(df_4h)
            regime = self.regime_detector.detect(df_4h, len(df_4h) - 1)
            if regime == MarketRegime.TREND_UP and signal == "SELL":
                return False, "TREND_UP — SHORT engellendi", 0.0
            if regime == MarketRegime.TREND_DOWN and signal == "BUY":
                return False, "TREND_DOWN — LONG engellendi", 0.0
            if regime == MarketRegime.RANGE:
                return False, "RANGE — trend stratejisi calismaz", 0.0
            if regime == MarketRegime.VOLATILE:
                return True, "VOLATILE — pozisyon kucultuldu", 0.5
            return True, f"Rejim: {regime.value}", 1.0
        except Exception:
            return True, "", 1.0

    # ═══════════════════════════════════════════════════════════
    # TARAMA
    # ═══════════════════════════════════════════════════════════

    def run_single_scan(self) -> List[dict]:
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        if now_utc.hour in [23, 0]:
            logger.info("Dusuk likidite saati (UTC 23-01) — tarama atlandi")
            return []

        self.fetcher.clear_cache()
        logger.info("=" * 60)
        logger.info("V7 TARAMA BAŞLIYOR")
        logger.info("=" * 60)

        try:
            symbols = self.fetcher.fetch_top_symbols(top_n=self.top_n, quote="USDT")
            if not symbols:
                logger.error("Sembol listesi alınamadı")
                return []
            logger.info(f"Top {len(symbols)} USDT çifti taranacak")
        except Exception as e:
            logger.error(f"Sembol listesi hatası: {e}")
            return []

        signals = []
        seen_symbols = set()

        for symbol in symbols:
            try:
                if symbol in self.positions:
                    continue

                quality_ok, quality_reason = self._check_symbol_quality(symbol)
                if not quality_ok:
                    continue

                tf_signals = {}
                for tf in ["15m", "1h", "4h"]:
                    df = self.fetcher.fetch_ohlcv(symbol, tf, limit=500)
                    if df.empty or len(df) < 200:
                        continue

                    df_signals = self.strategy.calculate_signals(df)

                    for i in range(max(0, len(df_signals) - 15), len(df_signals)):
                        sig = df_signals.iloc[i]["signal"]
                        if sig in ("BUY", "SELL"):
                            hist_price = float(df_signals.iloc[i]["close"])
                            sl_hist    = float(df_signals.iloc[i]["sl_price"])
                            tp_hist    = float(df_signals.iloc[i]["tp_price"])
                            atr        = float(df_signals.iloc[i].get("atr", hist_price * 0.02))
                            tf_signals[tf] = {
                                "signal": sig, "price": hist_price,
                                "sl": sl_hist, "tp": tp_hist, "atr": atr,
                                "time": df.index[i],
                            }
                            break

                if len(tf_signals) < 2:
                    continue

                signal_dirs = [v["signal"] for v in tf_signals.values()]
                if not (signal_dirs.count("BUY") >= 2 or signal_dirs.count("SELL") >= 2):
                    continue

                primary_tf = "1h" if "1h" in tf_signals else list(tf_signals.keys())[0]
                sig_data = tf_signals[primary_tf]
                sig = sig_data["signal"]
                hist_price = sig_data["price"]

                try:
                    ticker = self.fetcher.exchange.fetch_ticker(symbol)
                    live_price = float(ticker["last"])
                except Exception:
                    live_price = hist_price

                risk = abs(hist_price - sig_data["sl"])
                if risk <= 0:
                    continue

                if sig == "BUY":
                    sl = live_price - risk
                    tp = live_price + (self.strategy.target_rr * risk)
                else:
                    sl = live_price + risk
                    tp = live_price - (self.strategy.target_rr * risk)

                if tp <= 0:
                    continue

                signals.append({
                    "symbol": symbol,
                    "signal": sig,
                    "price": live_price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "atr": sig_data["atr"],
                    "time": str(sig_data["time"]),
                })
                seen_symbols.add(symbol)
                logger.info(f"SİNYAL (2+ TF): {sig} {symbol} @ {format_price(live_price)} | SL: {format_price(sl)} | TP: {format_price(tp)}")


            except Exception as e:
                logger.error(f"{symbol} tarama hatası: {e}")

        # Sinyal loglama
        try:
            Path("logs").mkdir(exist_ok=True)
            with open(self.signals_jsonl_path, "a", encoding="utf-8") as f:
                now_str = pd.Timestamp.now().isoformat()
                for s in signals:
                    f.write(json.dumps({
                        "generated_at": now_str,
                        "symbol": s["symbol"],
                        "signal_type": s["signal"],
                        "price": s["price"],
                        "stop_loss": s["stop_loss"],
                        "take_profit": s["take_profit"],
                    }) + "\n")
        except Exception as e:
            logger.error(f"Sinyal loglama hatası: {e}")

        logger.info(f"Tarama tamamlandı: {len(signals)} sinyal")
        return signals

    # ═══════════════════════════════════════════════════════════
    # İŞLEM AÇMA
    # ═══════════════════════════════════════════════════════════

    def open_position(self, signal: dict) -> bool:
        symbol = signal["symbol"]
        direction = signal["signal"]
        price = signal["price"]
        sl = signal["stop_loss"]
        tp = signal["take_profit"]

        # Kontroller
        if symbol in self.positions:
            print(f"[V7] {symbol} zaten acik")
            return False

        if len(self.positions) >= self.max_positions:
            print(f"[V7] Max pozisyon dolu ({self.max_positions})")
            return False

        in_cooldown, cooldown_reason = self._is_in_cooldown(symbol)
        if in_cooldown:
            print(f"[V7] {symbol} cooldown: {cooldown_reason}")
            return False

        # Günlük drawdown kontrolü
        current_dd = self.risk_engine.get_daily_drawdown_pct()
        if current_dd >= self.max_daily_drawdown_pct:
            logger.warning(f"Günlük drawdown limiti aşıldı: %{current_dd*100:.2f} >= %{self.max_daily_drawdown_pct*100:.2f}. Yeni pozisyon açılmıyor.")
            print(f"[V7] Drawdown kilidi: %{current_dd*100:.2f}")
            return False

        sector_ok, sector_reason = self._check_sector_limit(symbol)
        if not sector_ok:
            print(f"[V7] {symbol} sektor: {sector_reason}")
            return False

        regime_ok, regime_reason, regime_mult = self._check_regime(symbol, direction)
        if not regime_ok:
            print(f"[V7] {symbol} rejim: {regime_reason}")
            return False

        # Risk/Ödül oranı kontrolü (dinamik RR)
        effective_rr = self._get_dynamic_rr(symbol)
        if effective_rr == 0:
            print(f"[V7] {symbol} ADX zayıf — işlem atlandı")
            return False
        if direction == "BUY":
            reward = tp - price
            risk = price - sl
        else:
            reward = price - tp
            risk = sl - price
        if risk > 0 and reward / risk < effective_rr:
            print(f"[V7] {symbol} RR düşük: {reward/risk:.2f} < {effective_rr:.1f}")
            return False

        # Pozisyon boyutu
        size_mult = self._get_position_size_multiplier(symbol)
        consec_mult = self._get_consecutive_loss_multiplier(symbol)
        ai_mult = signal.get("size_mult", 1.0)
        risk_amount = self.balance * self.position_pct * size_mult * regime_mult * consec_mult * ai_mult

        if direction == "BUY":
            risk_per_unit = price - sl
        else:
            risk_per_unit = sl - price

        if risk_per_unit <= 0:
            print(f"[V7] {symbol} risk <= 0: price={price}, sl={sl}")
            return False

        position_size = risk_amount / risk_per_unit
        notional = position_size * price

        if notional > self.balance * 0.1:  # Max %10 sermaye
            notional = self.balance * 0.1
            position_size = notional / price

        # Pozisyon kaydı
        self.positions[symbol] = {
            "direction": direction,
            "entry_price": price,
            "stop_loss": sl,
            "take_profit": tp,
            "size": position_size,
            "notional": notional,
            "opened_at": pd.Timestamp.now().isoformat(),
        }

        self.balance -= notional * self.commission_rate  # Komisyon
        self._save_state()

        return {
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "sl": sl,
            "tp": tp,
            "notional": notional,
        }

    # ═══════════════════════════════════════════════════════════
    # POZİSYON İZLEME
    # ═══════════════════════════════════════════════════════════

    def monitor_positions(self) -> None:
        if not self.positions:
            return

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            try:
                ticker = self.fetcher.exchange.fetch_ticker(symbol)
                current_price = ticker["last"]

                # Trailing stop güncelle
                try:
                    df_atr = self.fetcher.fetch_ohlcv(symbol, "1h", limit=20)
                    if not df_atr.empty and len(df_atr) >= 14:
                        high_low = df_atr["high"] - df_atr["low"]
                        high_close = (df_atr["high"] - df_atr["close"].shift(1)).abs()
                        low_close = (df_atr["low"] - df_atr["close"].shift(1)).abs()
                        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                        atr_val = float(tr.rolling(window=14).mean().iloc[-1])
                        if atr_val > 0:
                            side = "LONG" if pos["direction"] == "BUY" else "SHORT"
                            new_sl = self.risk_engine.calculate_dynamic_trailing(
                                entry_price=pos["entry_price"],
                                current_price=current_price,
                                original_sl=pos["stop_loss"],
                                side=side,
                                atr=atr_val,
                            )
                            if new_sl != pos["stop_loss"]:
                                pos["stop_loss"] = new_sl
                                logger.info(f"{symbol} trailing stop güncellendi: {format_price(new_sl)}")
                except Exception:
                    pass

                if pos["direction"] == "BUY":
                    if current_price <= pos["stop_loss"]:
                        self._close_position(symbol, current_price, "STOP_LOSS")
                    elif current_price >= pos["take_profit"]:
                        self._close_position(symbol, current_price, "TAKE_PROFIT")
                else:
                    if current_price >= pos["stop_loss"]:
                        self._close_position(symbol, current_price, "STOP_LOSS")
                    elif current_price <= pos["take_profit"]:
                        self._close_position(symbol, current_price, "TAKE_PROFIT")

            except Exception as e:
                logger.error(f"{symbol} fiyat kontrolü hatası: {e}")

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        pos = self.positions.pop(symbol)
        entry = pos["entry_price"]
        size = pos["size"]

        if pos["direction"] == "BUY":
            pnl_usd = (exit_price - entry) * size * self.leverage
        else:
            pnl_usd = (entry - exit_price) * size * self.leverage

        commission = size * entry * self.commission_rate * 2
        net_pnl = pnl_usd - commission
        pnl_pct = pnl_usd / (entry * size * self.leverage) if entry * size * self.leverage > 0 else 0

        self.balance += net_pnl

        # Risk engine'e zararı kaydet
        if net_pnl < 0:
            self.risk_engine.record_loss(abs(net_pnl))
        self.risk_engine.set_balance(self.balance)

        # Win rate takibi
        if symbol not in self._symbol_stats:
            self._symbol_stats[symbol] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
        self._symbol_stats[symbol]["total_pnl"] += net_pnl
        if net_pnl > 0:
            self._symbol_stats[symbol]["wins"] += 1
        else:
            self._symbol_stats[symbol]["losses"] += 1
            self._set_cooldown(symbol, "STOP_LOSS", self.COOLDOWN_SL_HOURS)

        self._track_consecutive_loss(symbol, net_pnl)

        # Trades listesine kaydet (adaptif öğrenme için)
        self.trades.append(TradeRecord(
            symbol=symbol,
            side=pos["direction"],
            entry_price=entry,
            exit_price=exit_price,
            entry_time=pd.to_datetime(pos["opened_at"]),
            exit_time=pd.Timestamp.now(),
            amount=size,
            pnl_usdt=net_pnl,
            pnl_pct=pnl_pct * 100,
            commission=commission,
            hold_duration_hours=(pd.Timestamp.now() - pd.to_datetime(pos["opened_at"])).total_seconds() / 3600,
            exit_reason=reason,
        ))

        # Adaptif öğrenmeye de kaydet
        self.learner.update_symbol_stats(symbol, net_pnl)

        # Telegram — Coin özel K/Z
        emoji = "✅" if net_pnl > 0 else "❌"
        
        # Bu coin'in toplam istatistiği
        coin_stats = self._symbol_stats.get(symbol, {"wins": 0, "losses": 0, "total_pnl": 0.0})
        coin_total_pnl = coin_stats["total_pnl"]
        coin_trades = coin_stats["wins"] + coin_stats["losses"]
        coin_wr = coin_stats["wins"] / coin_trades * 100 if coin_trades else 0
        
        msg = (
            f"{emoji} [V7] POZİSYON KAPATILDI — {symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Yön: {pos['direction']}\n"
            f"💰 Giriş: {format_price(entry)}\n"
            f"💵 Çıkış: {format_price(exit_price)}\n"
            f"📈 İşlem K/Z: {format_price(net_pnl)} (%{pnl_pct*100:+.1f})\n"
            f"📋 Neden: {reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} Özeti:\n"
            f"   İşlem: {coin_trades} | WR: %{coin_wr:.0f}\n"
            f"   Toplam K/Z: {format_price(coin_total_pnl)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Bakiye: {format_price(self.balance)}"
        )
        send_telegram_notification(msg)

        self._save_state()

    def _partial_close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        entry = pos["entry_price"]
        size = pos["size"]
        
        # Pozisyonun %50'sini kapat
        close_size = size * 0.5
        
        if pos["direction"] == "BUY":
            pnl_usd = (exit_price - entry) * close_size * self.leverage
        else:
            pnl_usd = (entry - exit_price) * close_size * self.leverage
            
        commission = close_size * entry * self.commission_rate * 2
        net_pnl = pnl_usd - commission
        pnl_pct = pnl_usd / (entry * close_size * self.leverage) if entry * close_size * self.leverage > 0 else 0
        
        self.balance += net_pnl
        
        # Risk engine kaydı
        if net_pnl < 0:
            self.risk_engine.record_loss(abs(net_pnl))
        self.risk_engine.set_balance(self.balance)
        
        # Sembol istatistiği güncelle
        if symbol not in self._symbol_stats:
            self._symbol_stats[symbol] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
        self._symbol_stats[symbol]["total_pnl"] += net_pnl
        
        # Kalan pozisyonu güncelle
        pos["size"] -= close_size
        pos["partial_closes"] = pos.get("partial_closes", 0) + 1
        
        # Stop loss'u giriş fiyatına çek (Breakeven)
        pos["stop_loss"] = entry
        
        # TradeRecord ekle
        self.trades.append(TradeRecord(
            symbol=symbol,
            side=pos["direction"],
            entry_price=entry,
            exit_price=exit_price,
            entry_time=pd.to_datetime(pos["opened_at"]),
            exit_time=pd.Timestamp.now(),
            amount=close_size,
            pnl_usdt=net_pnl,
            pnl_pct=pnl_pct * 100,
            commission=commission,
            hold_duration_hours=(pd.Timestamp.now() - pd.to_datetime(pos["opened_at"])).total_seconds() / 3600,
            exit_reason=f"PARTIAL_CLOSE: {reason}",
        ))
        
        # Öğrenmeye de kaydet
        self.learner.update_symbol_stats(symbol, net_pnl)
        
        # Telegram bildirimi
        emoji = "💰" if net_pnl > 0 else "📉"
        msg = (
            f"{emoji} [V7] KISMİ POZİSYON KAPATILDI (%50) — {symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Yön: {pos['direction']}\n"
            f"💰 Giriş: {format_price(entry)}\n"
            f"💵 Kısmi Çıkış: {format_price(exit_price)}\n"
            f"📈 Kısmi K/Z: {format_price(net_pnl)} (%{pnl_pct*100:+.1f})\n"
            f"🛡️ SL Maliyete Çekildi (Breakeven: {format_price(entry)})\n"
            f"📋 Neden: {reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Bakiye: {format_price(self.balance)}"
        )
        send_telegram_notification(msg)
        self._save_state()

    # ═══════════════════════════════════════════════════════════
    # ANA DÖNGÜ
    # ═══════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════
    # MiMo POZISİYON İNCELEME
    # ═══════════════════════════════════════════════════════════

    def _review_positions_with_ai(self) -> None:
        """
        Her döngüde çalışır; açık her pozisyon için MiMo'ya grafik gönderir.
        MiMo CLOSE derse pozisyon kapatılır, Telegram'a bildirim gider.
        """
        if not self.ai_verifier.is_enabled or not self.positions:
            return

        logger.info(f"[MiMo] {len(self.positions)} açık pozisyon inceleniyor...")
        symbols_to_close = []

        for symbol, pos in list(self.positions.items()):
            try:
                # Anlık fiyatı al
                ticker = self.fetcher.exchange.fetch_ticker(symbol)
                current_price = float(ticker["last"])

                # Grafik için veri çek
                df = self.fetcher.fetch_ohlcv(symbol, "1h", limit=100)
                if df.empty:
                    continue

                review = self.ai_verifier.review_position(
                    symbol=symbol,
                    direction=pos["direction"],
                    entry_price=float(pos["entry_price"]),
                    current_price=current_price,
                    sl=float(pos["stop_loss"]),
                    tp=float(pos["take_profit"]),
                    df=df,
                    opened_at=pos.get("opened_at", ""),
                    timeframe="1h",
                )

                if not review.get("skipped_ai"):
                    action = review.get("action", "HOLD")
                    reason = review.get("reason", "MiMo kararı")

                    if action == "CLOSE":
                        pnl_pct = review.get("pnl_pct", 0.0)
                        symbols_to_close.append((symbol, reason, pnl_pct))
                        logger.info(f"[MiMo] {symbol} KAPATILACAK: {reason}")
                    elif action == "PARTIAL_CLOSE":
                        # Zaten kısmi kapatma yapılmadıysa kapat
                        if pos.get("partial_closes", 0) == 0:
                            logger.info(f"[MiMo] {symbol} KISMİ KAPATILACAK: {reason}")
                            self._partial_close_position(symbol, current_price, reason)
                        else:
                            logger.info(f"[MiMo] {symbol} için PARTIAL_CLOSE istendi fakat zaten kısmi kapatma yapılmış, HOLD olarak devam ediliyor.")

            except Exception as e:
                logger.warning(f"[MiMo] {symbol} inceleme atlandı: {e}")

        # Kapatma işlemleri
        for symbol, reason, pnl_pct in symbols_to_close:
            if symbol not in self.positions:
                continue
            try:
                pos = self.positions[symbol]
                ticker = self.fetcher.exchange.fetch_ticker(symbol)
                exit_price = float(ticker["last"])

                self._close_position(symbol, exit_price, f"MiMo: {reason}")

                pnl_sign = "+" if pnl_pct >= 0 else ""
                send_telegram_notification(
                    f"\U0001f916 [MiMo] POZİSİYON KAPATILDI\n"
                    f"Coin  : {symbol} ({pos['direction']})\n"
                    f"K/Z   : {pnl_sign}{pnl_pct:.2f}%\n"
                    f"Sebep : {reason}"
                )
            except Exception as e:
                logger.error(f"[MiMo] {symbol} kapatılamıyor: {e}")

    # ═══════════════════════════════════════════════════════════
    def run_cycle(self) -> None:
        logger.info(f"Döngü başlıyor | Bakiye: {format_price(self.balance)} | Pozisyon: {len(self.positions)}")
        newly_opened = []  # Bu döngüde açılan pozisyonları topla

        # Pozisyon izle
        self.monitor_positions()

        # MiMo ile açık pozisyonları incele (her 1 saatte bir)
        now = pd.Timestamp.now()
        elapsed_min = (
            (now - self._last_ai_review_time).total_seconds() / 60
            if self._last_ai_review_time is not None else 9999
        )
        if elapsed_min >= self._ai_review_interval_min:
            self._last_ai_review_time = now
            self._review_positions_with_ai()

        # Tarama yap
        signals = self.run_single_scan()

        # Sinyalleri işle
        for signal in signals:
            if len(self.positions) >= self.max_positions:
                break

            # ── AI Grafik Doğrulama ─────────────────────────────────────
            try:
                sym = signal["symbol"]
                df_for_ai = self.fetcher.fetch_ohlcv(sym, "1h", limit=120)
                if not df_for_ai.empty:
                    ai_result = self.ai_verifier.verify(
                        symbol=sym,
                        direction=signal["signal"],
                        price=signal["price"],
                        sl=signal["stop_loss"],
                        tp=signal["take_profit"],
                        df=df_for_ai,
                        timeframe="1h",
                        timestamp=signal.get("time"),
                    )
                    if not ai_result["approved"]:
                        reason = ai_result.get("reason", "MiMo kararı: SKIP")
                        logger.info(f"[MiMo AI] {sym} atlandı — {reason}")
                        send_telegram_notification(
                            f"🚫 [MiMo AI] {sym} ({signal['signal']}) işlemi engellendi!\n"
                            f"Engellenen Giriş Fiyatı: {format_price(signal['price'])}\n"
                            f"Sebep: {reason}"
                        )
                        continue
                    else:
                        logger.info(f"[MiMo AI] {sym} onaylandı — {ai_result.get('reason', '')}")
                        confidence = ai_result.get("confidence", "MEDIUM")
                        if confidence == "LOW":
                            signal["size_mult"] = 0.5
                        elif confidence == "HIGH":
                            signal["size_mult"] = 1.25
            except Exception as e:
                logger.warning(f"[AI] {signal['symbol']} doğrulama atlandı: {e}")
                continue
            # ── AI Doğrulama Sonu ───────────────────────────────────────

            result = self.open_position(signal)
            if result:
                newly_opened.append(result)

        # Yeni açılan pozisyonları tek mesajda gönder
        if newly_opened:
            lines = [f"🤖 [V7] {len(newly_opened)} YENİ POZİSYON AÇILDI"]
            lines.append(f"💰 Bakiye: {format_price(self.balance)} | Pozisyon: {len(self.positions)}/{self.max_positions}")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for p in newly_opened:
                yon_emoji = "🟢" if p["direction"] == "BUY" else "🔴"
                lines.append(f"{yon_emoji} {p['symbol']} | {p['direction']}")
                lines.append(f"   Giriş: {format_price(p['price'])}  SL: {format_price(p['sl'])}  TP: {format_price(p['tp'])}")
                lines.append(f"   Değer: {format_price(p['notional'])}")
            # Açık pozisyonların toplam anlık K/Z'si
            total_unrealized = 0.0
            for sym, pos in self.positions.items():
                try:
                    ticker = self.fetcher.exchange.fetch_ticker(sym)
                    current = float(ticker["last"])
                    if pos["direction"] == "BUY":
                        total_unrealized += (current - pos["entry_price"]) * pos["size"] * self.leverage
                    else:
                        total_unrealized += (pos["entry_price"] - current) * pos["size"] * self.leverage
                except Exception:
                    pass
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"📊 Açık Pozisyon K/Z: {format_price(total_unrealized)}")
            send_telegram_notification("\n".join(lines))

        # Günlük optimizasyon kontrolü
        now = datetime.now()
        if self._last_daily_optimize is None or (now - self._last_daily_optimize).total_seconds() > 86400:
            self._run_daily_optimization()
            self._last_daily_optimize = now

        self._print_status()

    def _run_daily_optimization(self) -> None:
        """Günlük optimizasyon — adaptif öğrenme."""
        try:
            # İşlemleri learner'a aktar
            for trade in self.trades:
                self.learner.update_symbol_stats(trade.symbol, trade.pnl_usdt)

            # Optimizasyonu çalıştır
            result = self.learner.daily_optimize(self._get_portfolio_state())

            if result.get("optimized"):
                logger.info(f"Günlük optimizasyon tamamlandı: {result.get('changes', [])}")
                new_params = self.learner.get_current_params()
                self.strategy.target_rr = new_params.get("target_rr", self.strategy.target_rr)
                self.strategy.sweep_window = new_params.get("sweep_window", self.strategy.sweep_window)
                self.strategy.max_hold_sweep = new_params.get("max_hold_sweep", self.strategy.max_hold_sweep)
                self.strategy.atr_multiplier = new_params.get("atr_multiplier", self.strategy.atr_multiplier)
                self.strategy.volume_threshold = new_params.get("volume_threshold", self.strategy.volume_threshold)
                self.strategy.min_volatility_pct = new_params.get("min_volatility", self.strategy.min_volatility_pct)
                self.position_pct = new_params.get("risk_pct", self.position_pct)
            else:
                logger.info(f"Günlük optimizasyon atlandı: {result.get('reason')}")

        except Exception as e:
            logger.error(f"Günlük optimizasyon hatası: {e}")

    def _get_portfolio_state(self) -> dict:
        """Portföy durumunu döndür."""
        return {
            "balance": self.balance,
            "positions": self.positions,
            "symbol_stats": self._symbol_stats,
        }

    def _send_hourly_summary(self) -> None:
        """Saatlik Telegram özeti — anlık K/Z dahil."""
        now = datetime.now()
        total_pnl = sum(
            self._symbol_stats.get(s, {}).get("total_pnl", 0)
            for s in self._symbol_stats
        )
        total_wins = sum(self._symbol_stats.get(s, {}).get("wins", 0) for s in self._symbol_stats)
        total_losses = sum(self._symbol_stats.get(s, {}).get("losses", 0) for s in self._symbol_stats)
        total_trades = total_wins + total_losses
        wr = total_wins / total_trades * 100 if total_trades else 0

        # Anlık (realized edilmemiş) K/Z hesapla
        unrealized_pnl = 0.0
        for sym, pos in self.positions.items():
            try:
                ticker = self.fetcher.exchange.fetch_ticker(sym)
                current = float(ticker["last"])
                if pos["direction"] == "BUY":
                    unrealized_pnl += (current - pos["entry_price"]) * pos["size"] * self.leverage
                else:
                    unrealized_pnl += (pos["entry_price"] - current) * pos["size"] * self.leverage
            except Exception:
                pass

        version = self.learner.get_version()
        msg = (
            f"📊 SAATLIK ÖZET — V{version} Bot\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Bakiye: {format_price(self.balance)}\n"
            f"📈 Gerçekleşen K/Z: {format_price(total_pnl)}\n"
            f"📊 Açık Pozisyon K/Z: {format_price(unrealized_pnl)}\n"
            f"💼 Toplam: {format_price(total_pnl + unrealized_pnl)}\n"
            f"📊 İşlemler: {total_trades} (%{wr:.1f} WR)\n"
            f"🔓 Açık: {len(self.positions)}/{self.max_positions}\n"
            f"⏰ {now.strftime('%H:%M')}"
        )
        send_telegram_notification(msg)

    def _print_status(self) -> None:
        unrealized = 0.0
        for sym, pos in self.positions.items():
            try:
                ticker = self.fetcher.exchange.fetch_ticker(sym)
                current = float(ticker["last"])
                if pos["direction"] == "BUY":
                    pnl = (current - pos["entry_price"]) * pos["size"] * self.leverage
                else:
                    pnl = (pos["entry_price"] - current) * pos["size"] * self.leverage
                unrealized += pnl
                emoji = "🟢" if pnl >= 0 else "🔴"
                logger.info(f"  {emoji} {sym}: {pos['direction']} @ {format_price(pos['entry_price'])} | K/Z: {format_price(pnl)}")
            except Exception:
                logger.info(f"  {sym}: {pos['direction']} @ {format_price(pos['entry_price'])}")
        logger.info(f"Bakiye: {format_price(self.balance)} | Açık K/Z: {format_price(unrealized)} | Pozisyon: {len(self.positions)}")

    def run(self, interval_minutes: int = 15) -> None:
        """Ana döngü."""
        logger.info(f"V7 Bot döngüsü başlatıldı — {interval_minutes} dakika aralıkla")

        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Bot durduruldu (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"Döngü hatası: {e}")
                send_telegram_notification(f"⚠️ V7 Bot HATA: {e}")

            time.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="ANTIGRAVITI V7 Price Action Bot")
    parser.add_argument("--live", action="store_true", help="Canlı mod")
    parser.add_argument("--paper", action="store_true", help="Paper trade modu (varsayılan)")
    parser.add_argument("--single-run", action="store_true", help="Tek seferlik tarama")
    parser.add_argument("--interval", type=int, default=15, help="Tarama aralığı (dakika)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Başlangıç sermayesi")
    parser.add_argument("--top-n", type=int, default=200, help="Taranacak sembol sayısı")
    args = parser.parse_args()

    is_live = args.live and not args.paper

    bot = LiveV7Bot(
        initial_capital=args.capital,
        max_positions=10,
        position_pct=0.01,
        is_live=is_live,
        top_n=args.top_n,
    )

    if args.single_run:
        signals = bot.run_single_scan()
        bot._print_status()
    else:
        bot.run(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
