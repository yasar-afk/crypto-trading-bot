# ============================================================
# src/signal/journal.py — ANTIGRAVITI Trading Bot
#
# AMAÇ:
#   Üretilen tüm sinyalleri dosyaya ve belleğe kaydeden sinyal
#   defteri. Paper trade modunda performans takibi yapar;
#   Faz 2'de backtesting motoru bu veriyi tüketir.
#
# KAYIT FORMATLARI:
#   - signals.jsonl : Her sinyal bir satır JSON (append-only)
#   - Bellek        : Son N sinyali RAM'de tutar (hızlı erişim)
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v1.0 | İlk SignalJournal implementasyonu
# ============================================================

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

from src.signal.models import SignalType, TradeSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Bellekte tutulacak maksimum sinyal sayısı
MAX_IN_MEMORY = 500


class SignalJournal:
    """Üretilen tüm sinyalleri kaydeder ve istatistik üretir.

    Attributes:
        journal_path: JSONL dosya yolu.
        _signals: Bellekteki son MAX_IN_MEMORY sinyal.
        _stats: Güncel istatistikler.

    Example:
        >>> journal = SignalJournal()
        >>> journal.record(signal)
        >>> print(journal.summary())
    """

    def __init__(
        self,
        journal_dir: str = "logs",
        journal_file: str = "signals.jsonl",
    ) -> None:
        """SignalJournal başlatır.

        Args:
            journal_dir: Log klasörü.
            journal_file: Sinyal defteri dosyası adı.
        """
        Path(journal_dir).mkdir(parents=True, exist_ok=True)
        self.journal_path = Path(journal_dir) / journal_file
        self._signals: Deque[TradeSignal] = deque(maxlen=MAX_IN_MEMORY)
        self._stats: Dict = self._empty_stats()

        logger.info(f"SignalJournal başlatıldı: {self.journal_path}")

    # ── Kayıt Metodları ──────────────────────────────────────

    def record(self, signal: TradeSignal) -> None:
        """Sinyali deftere kaydeder (hem dosyaya hem belleğe).

        Args:
            signal: Kaydedilecek TradeSignal.
        """
        self._signals.append(signal)
        self._update_stats(signal)
        self._append_to_file(signal)

        logger.debug(
            f"Sinyal kaydedildi: {signal.signal_type.value} | "
            f"{signal.symbol}@{signal.timeframe} | "
            f"Toplam: {self._stats['total_signals']}"
        )

    def _append_to_file(self, signal: TradeSignal) -> None:
        """Sinyali JSONL dosyasına ekler (append-only)."""
        try:
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(signal.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Sinyal dosyaya yazılamadı: {e}")

    # ── İstatistik Metodları ──────────────────────────────────

    def _update_stats(self, signal: TradeSignal) -> None:
        """İstatistikleri günceller."""
        self._stats["total_signals"] += 1

        stype = signal.signal_type.value
        self._stats["by_type"][stype] = self._stats["by_type"].get(stype, 0) + 1

        if signal.is_actionable:
            self._stats["actionable_signals"] += 1
            scores = self._stats["score_history"]
            scores.append(signal.weighted_score)
            self._stats["avg_score"] = round(sum(scores) / len(scores), 4)

    def _empty_stats(self) -> Dict:
        """Boş istatistik sözlüğü."""
        return {
            "total_signals": 0,
            "actionable_signals": 0,
            "avg_score": 0.0,
            "by_type": {},
            "score_history": [],
        }

    # ── Sorgulama Metodları ───────────────────────────────────

    def get_recent(self, n: int = 10) -> List[TradeSignal]:
        """Son N sinyali döner.

        Args:
            n: Kaç sinyal alınacak.

        Returns:
            En yeni sinyal başta olmak üzere liste.
        """
        signals = list(self._signals)
        return list(reversed(signals[-n:]))

    def get_by_type(self, signal_type: SignalType) -> List[TradeSignal]:
        """Belirli tipte sinyalleri döner."""
        return [s for s in self._signals if s.signal_type == signal_type]

    def get_last_actionable(self) -> Optional[TradeSignal]:
        """En son işlemleşebilir (BUY/SELL) sinyali döner."""
        for signal in reversed(list(self._signals)):
            if signal.is_actionable:
                return signal
        return None

    def summary(self) -> str:
        """Sinyal defterin özet istatistiklerini döner."""
        lines = [
            "=" * 50,
            "  ANTIGRAVITI — Sinyal Defteri Özeti",
            "=" * 50,
            f"  Toplam Sinyal     : {self._stats['total_signals']}",
            f"  İşlemleşebilir   : {self._stats['actionable_signals']}",
            f"  Ort. Skor        : {self._stats['avg_score']:.3f}",
            "  Tipe Göre Dağılım:",
        ]
        for stype, count in self._stats["by_type"].items():
            lines.append(f"    {stype:<15}: {count}")
        lines.append(f"  Dosya: {self.journal_path}")
        lines.append("=" * 50)
        return "\n".join(lines)

    @property
    def total_signals(self) -> int:
        return self._stats["total_signals"]

    @property
    def actionable_count(self) -> int:
        return self._stats["actionable_signals"]
