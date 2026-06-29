# ============================================================
# src/backtest/report.py — Backtest Rapor Üretici
#
# AMAÇ:
#   Backtest sonuçlarını kapsamlı HTML ve konsol raporları
#   olarak üretir.
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.backtest.engine import BacktestMetrics, BacktestResult, TradeRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestReporter:
    """Backtest raporları üreten sınıf."""

    def __init__(self, output_dir: str = "raporlar/backtest") -> None:
        """BacktestReporter başlatır.

        Args:
            output_dir: Raporların kaydedileceği dizin.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_console_report(self, result: BacktestResult) -> None:
        """Konsol için detaylı rapor yazdırır.

        Args:
            result: Backtest sonucu.
        """
        m = result.metrics

        print("\n" + "=" * 80)
        print(f"📊 BACKTEST RAPORU — {result.symbol} | {result.strategy_name} | {result.timeframe}")
        print("=" * 80)

        # Temel bilgiler
        print(f"\n💰 FİNANSAL SONUÇLAR")
        print(f"  Başlangıç Sermayesi : ${m.initial_capital:,.2f}")
        print(f"  Bitiş Sermayesi     : ${m.final_capital:,.2f}")
        print(f"  Toplam Kâr/Zarar    : ${m.total_pnl:+,.2f} ({m.total_pnl_pct:+.2f}%)")
        print(f"  Yıllık Getiri       : {m.annual_return_pct:+.2f}%")

        # İşlem metrikleri
        print(f"\n📈 İŞLEM METRİKLERİ")
        print(f"  Toplam İşlem        : {m.total_trades}")
        print(f"  Kârlı İşlem         : {m.winning_trades}")
        print(f"  Zararlı İşlem       : {m.losing_trades}")
        print(f"  Kazanma Oranı       : %{m.win_rate:.2f}")
        print(f"  Ortalama Kâr        : %{m.avg_win_pct:.2f}")
        print(f"  Ortalama Zarar      : %{m.avg_loss_pct:.2f}")
        print(f"  En Büyük Kâr        : ${m.largest_win_usdt:+,.2f}")
        print(f"  En Büyük Zarar      : ${m.largest_loss_usdt:+,.2f}")
        print(f"  Ortalama Süre       : {m.avg_hold_hours:.1f} saat")
        print(f"  En Uzun Pozisyon    : {m.max_hold_hours:.1f} saat")
        print(f"  En Kısa Pozisyon    : {m.min_hold_hours:.1f} saat")

        # Risk metrikleri
        print(f"\n🛡️ RİSK METRİKLERİ")
        print(f"  Sharpe Ratio        : {m.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio       : {m.sortino_ratio:.3f}")
        print(f"  Maksimum Drawdown   : %{m.max_drawdown_pct:.2f}")
        print(f"  DD Süresi           : {m.max_drawdown_duration_days:.1f} gün")
        print(f"  Calmar Ratio        : {m.calmar_ratio:.3f}")
        print(f"  Profit Factor       : {m.profit_factor:.3f}")
        print(f"  Expectancy          : ${m.expectancy:+,.2f}")
        print(f"  Recovery Factor     : {m.recovery_factor:.3f}")

        # İşlem dağılımı
        print(f"\n📊 İŞLEM DAĞILIMI")
        for side, count in m.trade_distribution.items():
            print(f"  {side}: {count} işlem")

        # Aylık getiriler
        if m.monthly_returns:
            print(f"\n📅 AYLIK GETİRİLER")
            for month, ret in m.monthly_returns.items():
                emoji = "🟢" if ret > 0 else "🔴"
                print(f"  {emoji} {month}: {ret:+.2f}%")

        # En iyi ve kötü 5 işlem
        if result.trades:
            print(f"\n🏆 EN İYİ 5 İŞLEM")
            sorted_trades = sorted(result.trades, key=lambda t: t.pnl_usdt, reverse=True)
            for i, t in enumerate(sorted_trades[:5], 1):
                print(f"  {i}. {t.symbol} | {t.side} | ${t.pnl_usdt:+,.2f} ({t.pnl_pct:+.2f}%) | {t.hold_duration_hours:.1f}h")

            print(f"\n💀 EN KÖTÜ 5 İŞLEM")
            for i, t in enumerate(sorted_trades[-5:], 1):
                print(f"  {i}. {t.symbol} | {t.side} | ${t.pnl_usdt:+,.2f} ({t.pnl_pct:+.2f}%) | {t.hold_duration_hours:.1f}h")

        print("\n" + "=" * 80)

    def print_comparison_table(
        self,
        results: List[BacktestResult],
    ) -> None:
        """Çoklu strateji karşılaştırma tablosu yazdırır.

        Args:
            results: BacktestResult listesi.
        """
        if not results:
            print("Sonuç bulunamadı.")
            return

        print("\n" + "=" * 120)
        print("📊 STRATEJİ KARŞILAŞTIRMA TABLOSU")
        print("=" * 120)

        header = f"{'Strateji':<20} {'Sembol':<12} {'Başlangıç':>12} {'Bitiş':>12} {'Kâr/Zarar':>12} {'%Getiri':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>8} {'İşlem':>6}"
        print(header)
        print("-" * 120)

        for r in sorted(results, key=lambda x: x.metrics.total_pnl, reverse=True):
            m = r.metrics
            line = (
                f"{r.strategy_name:<20} "
                f"{r.symbol:<12} "
                f"${m.initial_capital:>10,.2f} "
                f"${m.final_capital:>10,.2f} "
                f"${m.total_pnl:>+10,.2f} "
                f"{m.total_pnl_pct:>+7.2f}% "
                f"{m.sharpe_ratio:>7.3f} "
                f"{m.max_drawdown_pct:>7.2f}% "
                f"{m.win_rate:>7.2f}% "
                f"{m.profit_factor:>7.3f} "
                f"{m.total_trades:>5}"
            )
            print(line)

        print("=" * 120)

    def print_portfolio_summary(self, summary: Dict) -> None:
        """Portföy özeti yazdırır.

        Args:
            summary: MultiSymbolBacktest.get_portfolio_summary() sözlüğü.
        """
        if not summary:
            print("Portföy özeti bulunamadı.")
            return

        print("\n" + "=" * 80)
        print("💼 PORTFÖY ÖZET RAPORU")
        print("=" * 80)
        print(f"  Toplam Sembol        : {summary['total_symbols']}")
        print(f"  Toplam İşlem         : {summary['total_trades']}")
        print(f"  Kârlı / Zararlı      : {summary['winning_trades']} / {summary['losing_trades']}")
        print(f"  Genel Kazanma Oranı  : %{summary['win_rate']:.2f}")
        print(f"  Toplam Kâr/Zarar     : ${summary['total_pnl']:+,.2f} ({summary['total_pnl_pct']:+.2f}%)")
        print(f"  En İyi Sembol        : {summary['best_symbol']} (${summary['best_symbol_pnl']:+,.2f})")
        print(f"  En Kötü Sembol       : {summary['worst_symbol']} (${summary['worst_symbol_pnl']:+,.2f})")
        print(f"  Ortalama Sharpe      : {summary['avg_sharpe']:.3f}")
        print(f"  Ortalama Sortino     : {summary['avg_sortino']:.3f}")
        print(f"  Ortalama Max DD      : %{summary['avg_max_dd']:.2f}")
        print(f"  Ortalama Win Rate    : %{summary['avg_win_rate']:.2f}")
        print(f"  Ortalama Profit Factor: {summary['avg_profit_factor']:.3f}")
        print("=" * 80)

    def save_html_report(
        self,
        result: BacktestResult,
        filename: Optional[str] = None,
    ) -> Path:
        """HTML rapor kaydeder.

        Args:
            result: Backtest sonucu.
            filename: Dosya adı (None ise otomatik üretilir).

        Returns:
            Kaydedilen dosya yolu.
        """
        if filename is None:
            safe_symbol = result.symbol.replace("/", "_").replace("\\", "_")
            filename = f"backtest_{safe_symbol}_{result.strategy_name}.html"

        filepath = self.output_dir / filename
        m = result.metrics

        # Aylık getiri tablosu
        monthly_rows = ""
        for month, ret in m.monthly_returns.items():
            color = "#22c55e" if ret > 0 else "#ef4444"
            monthly_rows += f'<tr><td>{month}</td><td style="color:{color};font-weight:bold">{ret:+.2f}%</td></tr>\n'

        # İşlem listesi
        trade_rows = ""
        for t in result.trades[-50:]:  # Son 50 işlem
            color = "#22c55e" if t.pnl_usdt > 0 else "#ef4444"
            trade_rows += (
                f'<tr>'
                f'<td>{t.entry_time.strftime("%Y-%m-%d %H:%M")}</td>'
                f'<td>{t.exit_time.strftime("%Y-%m-%d %H:%M")}</td>'
                f'<td>{t.symbol}</td>'
                f'<td>{t.side}</td>'
                f'<td>${t.entry_price:,.4f}</td>'
                f'<td>${t.exit_price:,.4f}</td>'
                f'<td style="color:{color};font-weight:bold">${t.pnl_usdt:+,.2f}</td>'
                f'<td style="color:{color}">{t.pnl_pct:+.2f}%</td>'
                f'<td>{t.hold_duration_hours:.1f}h</td>'
                f'<td>{t.exit_reason}</td>'
                f'</tr>\n'
            )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ANTIGRAVITI Backtest Raporu — {result.symbol}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #38bdf8; border-bottom: 2px solid #38bdf8; padding-bottom: 10px; }}
        h2 {{ color: #94a3b8; margin-top: 30px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-label {{ color: #94a3b8; font-size: 0.85em; }}
        .metric-value {{ font-size: 1.8em; font-weight: bold; color: #38bdf8; }}
        .metric-value.positive {{ color: #22c55e; }}
        .metric-value.negative {{ color: #ef4444; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: #334155; color: #e2e8f0; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #334155; }}
        tr:hover {{ background: #1e293b; }}
        .green {{ color: #22c55e; }}
        .red {{ color: #ef4444; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 ANTIGRAVITI Backtest Raporu</h1>
    <p style="color:#94a3b8">Sembol: <strong>{result.symbol}</strong> | Strateji: <strong>{result.strategy_name}</strong> | Timeframe: <strong>{result.timeframe}</strong></p>

    <div class="card">
        <h2>💰 Finansal Sonuçlar</h2>
        <div class="metric">
            <div class="metric-label">Başlangıç</div>
            <div class="metric-value">${m.initial_capital:,.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Bitiş</div>
            <div class="metric-value {'positive' if m.final_capital >= m.initial_capital else 'negative'}">${m.final_capital:,.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Toplam Kâr/Zarar</div>
            <div class="metric-value {'positive' if m.total_pnl >= 0 else 'negative'}">${m.total_pnl:+,.2f} ({m.total_pnl_pct:+.2f}%)</div>
        </div>
        <div class="metric">
            <div class="metric-label">Yıllık Getiri</div>
            <div class="metric-value {'positive' if m.annual_return_pct >= 0 else 'negative'}">{m.annual_return_pct:+.2f}%</div>
        </div>
    </div>

    <div class="card">
        <h2>📈 İşlem Metrikleri</h2>
        <div class="metric">
            <div class="metric-label">Toplam İşlem</div>
            <div class="metric-value">{m.total_trades}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Kârlı / Zararlı</div>
            <div class="metric-value"><span class="green">{m.winning_trades}</span> / <span class="red">{m.losing_trades}</span></div>
        </div>
        <div class="metric">
            <div class="metric-label">Kazanma Oranı</div>
            <div class="metric-value">{m.win_rate:.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Ortalama Kâr</div>
            <div class="metric-value green">{m.avg_win_pct:+.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Ortalama Zarar</div>
            <div class="metric-value red">{m.avg_loss_pct:+.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">En Büyük Kâr</div>
            <div class="metric-value green">${m.largest_win_usdt:+,.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">En Büyük Zarar</div>
            <div class="metric-value red">${m.largest_loss_usdt:+,.2f}</div>
        </div>
    </div>

    <div class="card">
        <h2>🛡️ Risk Metrikleri</h2>
        <div class="metric">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{m.sharpe_ratio:.3f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Sortino Ratio</div>
            <div class="metric-value">{m.sortino_ratio:.3f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Maks. Drawdown</div>
            <div class="metric-value negative">%{m.max_drawdown_pct:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Calmar Ratio</div>
            <div class="metric-value">{m.calmar_ratio:.3f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">{m.profit_factor:.3f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Expectancy</div>
            <div class="metric-value {'positive' if m.expectancy >= 0 else 'negative'}">${m.expectancy:+,.2f}</div>
        </div>
    </div>

    <div class="card">
        <h2>📅 Aylık Getiriler</h2>
        <table>
            <tr><th>Ay</th><th>Getiri</th></tr>
            {monthly_rows}
        </table>
    </div>

    <div class="card">
        <h2>📋 Son İşlemler (Son 50)</h2>
        <table>
            <tr><th>Giriş</th><th>Çıkış</th><th>Sembol</th><th>Yön</th><th>Giriş Fiyatı</th><th>Çıkış Fiyatı</th><th>Kâr/Zarar</th><th>%</th><th>Süre</th><th>Neden</th></tr>
            {trade_rows}
        </table>
    </div>
</div>
</body>
</html>"""

        filepath.write_text(html, encoding="utf-8")
        logger.info(f"HTML rapor kaydedildi: {filepath}")
        return filepath

    def save_all_results(
        self,
        results: List[BacktestResult],
    ) -> List[Path]:
        """Tüm sonuçları HTML olarak kaydeder.

        Args:
            results: BacktestResult listesi.

        Returns:
            Kaydedilen dosya yolları.
        """
        saved = []
        for r in results:
            filepath = self.save_html_report(r)
            saved.append(filepath)
        return saved
