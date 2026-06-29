# -*- coding: utf-8 -*-
"""
src/strategy/adaptive_learner.py — Günlük Otomatik Öğrenme & Optimizasyon
Her gün sonunda:
  1. İşlemleri analiz et
  2. En iyi/kötü coin'leri belirle
  3. Parametreleri optimize et
  4. Versiyon güncelle (v7.001, v7.002, ...)
  5. Telegram'a bildirim gönder
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.telegram_notifier import send_telegram_notification

logger = get_logger("adaptive_learner")

VERSION_FILE = "logs/v7_version.json"
OPTIMIZATION_HISTORY = "logs/optimization_history.jsonl"
LEARNING_STATE = "logs/learning_state.json"


class AdaptiveLearner:
    """Günlük otomatik öğrenme ve parametre optimizasyonu."""

    def __init__(self):
        self.version = self._load_version()
        self.state = self._load_state()
        self.history: List[dict] = self._load_history()

    # ═══════════════════════════════════════════════════════════
    # VERSION MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def _load_version(self) -> str:
        try:
            if Path(VERSION_FILE).exists():
                with open(VERSION_FILE, "r") as f:
                    data = json.load(f)
                return data.get("version", "v7.000")
        except Exception:
            pass
        return "v7.000"

    def _save_version(self) -> None:
        Path("logs").mkdir(exist_ok=True)
        with open(VERSION_FILE, "w") as f:
            json.dump({
                "version": self.version,
                "updated_at": datetime.now().isoformat(),
                "changes": self.state.get("last_changes", [])
            }, f, indent=2, ensure_ascii=False)

    def _increment_version(self) -> str:
        """Versiyonu artır: v7.000 → v7.001 → v7.002"""
        try:
            parts = self.version.split(".")
            minor = int(parts[1]) + 1
            self.version = f"{parts[0]}.{minor:03d}"
        except (IndexError, ValueError):
            self.version = "v7.001"
        self._save_version()
        return self.version

    # ═══════════════════════════════════════════════════════════
    # STATE PERSISTENCE (Bot kapansa bile korunur)
    # ═══════════════════════════════════════════════════════════

    def _load_state(self) -> dict:
        try:
            if Path(LEARNING_STATE).exists():
                with open(LEARNING_STATE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "symbol_stats": {},  # {sym: {wins, losses, total_pnl, avg_hold_time}}
            "param_history": [],  # [{version, params, date}]
            "last_optimization": None,
            "last_changes": [],
            "best_symbols": [],
            "worst_symbols": [],
        }

    def _save_state(self) -> None:
        Path("logs").mkdir(exist_ok=True)
        self.state["last_save"] = datetime.now().isoformat()
        with open(LEARNING_STATE, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _load_history(self) -> List[dict]:
        history = []
        try:
            if Path(OPTIMIZATION_HISTORY).exists():
                with open(OPTIMIZATION_HISTORY, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            history.append(json.loads(line))
        except Exception:
            pass
        return history

    def _save_history_entry(self, entry: dict) -> None:
        Path("logs").mkdir(exist_ok=True)
        with open(OPTIMIZATION_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ═══════════════════════════════════════════════════════════
    # GÜNLÜK ANALİZ
    # ═══════════════════════════════════════════════════════════

    def analyze_trades(self, trades: List[dict]) -> dict:
        """Günlük işlemleri analiz et."""
        if not trades:
            return {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0, "win_rate": 0.0}

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        return {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100,
            "total_pnl": total_pnl,
            "avg_win": np.mean([t["pnl"] for t in wins]) if wins else 0,
            "avg_loss": np.mean([t["pnl"] for t in losses]) if losses else 0,
            "best_trade": max(trades, key=lambda t: t.get("pnl", 0)) if trades else None,
            "worst_trade": min(trades, key=lambda t: t.get("pnl", 0)) if trades else None,
        }

    def update_symbol_stats(self, symbol: str, pnl: float) -> None:
        """Coin istatistiklerini güncelle."""
        if symbol not in self.state["symbol_stats"]:
            self.state["symbol_stats"][symbol] = {
                "wins": 0, "losses": 0, "total_pnl": 0.0,
                "trades": 0, "last_trade": None
            }

        stats = self.state["symbol_stats"][symbol]
        stats["trades"] += 1
        stats["total_pnl"] += pnl
        stats["last_trade"] = datetime.now().isoformat()

        if pnl > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        if stats["losses"] >= 2 and stats["wins"] == 0:
            if symbol not in self.state.get("worst_symbols", []):
                self.state.setdefault("worst_symbols", []).append(symbol)
                logger.warning(f"{symbol} aninda blacklist'e alindi ({stats['losses']} kayip)")

        self._save_state()

    # ═══════════════════════════════════════════════════════════
    # GÜNLÜK OPTİMİZASYON
    # ═══════════════════════════════════════════════════════════

    def daily_optimize(self, portfolio_state: dict) -> dict:
        """Günlük optimizasyon — parametreleri ayarla, Telegram'a bildir."""
        changes = []
        now = datetime.now()

        last_opt = self.state.get("last_optimization")
        if last_opt:
            last_dt = datetime.fromisoformat(last_opt)
            if (now - last_dt).total_seconds() < 3600:
                return {"optimized": False, "reason": "Son optimizasyondan 1 saat gecmedi"}

        stats = self.state.get("symbol_stats", {})

        ranked = []
        immediate_blacklist = []
        for sym, s in stats.items():
            total = s["wins"] + s["losses"]
            if total >= 2:
                wr = s["wins"] / total
                ranked.append({"sym": sym, "wr": wr, "pnl": s["total_pnl"], "trades": total})
                if s["losses"] >= 3 or (s["losses"] >= 2 and s["wins"] == 0):
                    immediate_blacklist.append(sym)
            elif total >= 1 and s["losses"] >= 2:
                immediate_blacklist.append(sym)

        ranked.sort(key=lambda x: x["wr"], reverse=True)

        self.state["best_symbols"] = [r["sym"] for r in ranked[:10] if r["wr"] > 0.25]
        auto_worst = [r["sym"] for r in ranked[-10:] if r["wr"] < 0.15]
        all_worst = list(set(auto_worst + immediate_blacklist))
        self.state["worst_symbols"] = all_worst

        # ─── 2. Parametre ayarlamaları ────────────────────
        current_params = self.state.get("current_params", {
            "risk_pct": 0.02,
            "target_rr": 5.5,
            "sweep_window": 100,
            "max_hold_sweep": 7,
            "atr_multiplier": 0.6,
            "volume_threshold": 0.5,
            "min_volatility": 0.3,
        })

        # Genel win rate'e göre risk ayarı
        total_wins = sum(s["wins"] for s in stats.values())
        total_losses = sum(s["losses"] for s in stats.values())
        total_trades = total_wins + total_losses

        if total_trades >= 10:
            overall_wr = total_wins / total_trades

            if overall_wr > 0.30 and current_params["risk_pct"] < 0.02:
                new_risk = min(current_params["risk_pct"] + 0.002, 0.02)
                if new_risk != current_params["risk_pct"]:
                    changes.append(f"Risk artırıldı: %{current_params['risk_pct']*100:.1f} → %{new_risk*100:.1f} (WR>%30)")
                    current_params["risk_pct"] = new_risk

            elif overall_wr < 0.18 and current_params["risk_pct"] > 0.005:
                new_risk = max(current_params["risk_pct"] - 0.002, 0.005)
                if new_risk != current_params["risk_pct"]:
                    changes.append(f"Risk azaltıldı: %{current_params['risk_pct']*100:.1f} → %{new_risk*100:.1f} (WR<%18)")
                    current_params["risk_pct"] = new_risk

        # Kötü coin'leri黒liste al
        if self.state["worst_symbols"]:
            changes.append(f"黒liste: {len(self.state['worst_symbols'])} coin askıya alındı")

        # ─── 3. Parametre geçmişini kaydet ────────────────
        self.state["param_history"].append({
            "version": self.version,
            "params": current_params.copy(),
            "date": now.isoformat(),
            "changes": changes,
            "overall_wr": total_wins / total_trades * 100 if total_trades else 0,
        })

        # ─── 4. Versiyon güncelle ─────────────────────────
        if changes:
            new_version = self._increment_version()
            changes.insert(0, f"Versiyon: {self.version} → {new_version}")

        self.state["current_params"] = current_params
        self.state["last_optimization"] = now.isoformat()
        self.state["last_changes"] = changes
        self._save_state()

        # ─── 5. Telegram bildirimi ────────────────────────
        if changes:
            best_str = ", ".join(self.state["best_symbols"][:5]) if self.state["best_symbols"] else "Yok"
            worst_str = ", ".join(self.state["worst_symbols"][:5]) if self.state["worst_symbols"] else "Yok"

            msg = (
                f"🧠 GÜNLÜK ÖĞRENME — {self.version}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 İşlemler: {total_trades} (%{total_wins/total_trades*100:.1f} WR)\n"
                f"💰 Toplam K/Z: ${sum(s['total_pnl'] for s in stats.values()):+,.2f}\n"
                f"\n🔄 Yapılan Değişiklikler:\n"
            )
            for ch in changes:
                msg += f"  • {ch}\n"
            msg += (
                f"\n🏆 En İyi Coin: {best_str}\n"
                f"⚠️ En Kötü Coin: {worst_str}\n"
                f"\n⚙️ Güncel Parametreler:\n"
                f"  Risk: %{current_params['risk_pct']*100:.1f}\n"
                f"  RR: {current_params['target_rr']}\n"
                f"  Sweep: {current_params['sweep_window']}\n"
                f"  Volatilite: >%{current_params['min_volatility']}"
            )
            send_telegram_notification(msg)

        # Geçmişi kaydet
        self._save_history_entry({
            "date": now.isoformat(),
            "version": self.version,
            "changes": changes,
            "stats": {"total": total_trades, "wr": total_wins/total_trades*100 if total_trades else 0},
        })

        return {"optimized": True, "changes": changes, "version": self.version}

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def get_current_params(self) -> dict:
        """Mevcut optimizasyon parametrelerini döndür."""
        return self.state.get("current_params", {
            "risk_pct": 0.02,
            "target_rr": 5.5,
            "sweep_window": 100,
            "max_hold_sweep": 7,
            "atr_multiplier": 0.6,
            "volume_threshold": 0.5,
            "min_volatility": 0.3,
        })

    def is_symbol_blacklisted(self, symbol: str) -> bool:
        """Coin黒listede mi?"""
        return symbol in self.state.get("worst_symbols", [])

    def get_version(self) -> str:
        return self.version

    def get_learning_summary(self) -> dict:
        """Öğrenme durumu özeti."""
        stats = self.state.get("symbol_stats", {})
        total_trades = sum(s["trades"] for s in stats.values())
        total_wins = sum(s["wins"] for s in stats.values())

        return {
            "version": self.version,
            "total_symbols_tracked": len(stats),
            "total_trades": total_trades,
            "overall_win_rate": total_wins / total_trades * 100 if total_trades else 0,
            "best_symbols": self.state.get("best_symbols", []),
            "worst_symbols": self.state.get("worst_symbols", []),
            "last_optimization": self.state.get("last_optimization"),
            "optimization_count": len(self.history),
        }
