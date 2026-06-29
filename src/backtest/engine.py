# ============================================================
# src/backtest/engine.py — ANTIGRAVITI Backtest Motoru
#
# AMAÇ:
#   Geçmiş verilerle stratejileri test eden, kapsamlı metrikler
#   üreten ve HTML raporlar oluşturan backtest motoru.
# ============================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    """Tek bir işlemin kaydı."""
    symbol: str
    side: str  # "LONG" veya "SHORT"
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    amount: float
    pnl_usdt: float
    pnl_pct: float
    commission: float
    hold_duration_hours: float
    exit_reason: str


@dataclass
class BacktestMetrics:
    """Backtest metrikleri."""
    # Temel metrikler
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_pnl_pct: float
    annual_return_pct: float

    # İşlem metrikleri
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_usdt: float
    largest_loss_usdt: float
    avg_hold_hours: float
    max_hold_hours: float
    min_hold_hours: float

    # Risk metrikleri
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: float
    calmar_ratio: float
    profit_factor: float
    expectancy: float
    recovery_factor: float

    # Dağılım
    monthly_returns: Dict[str, float]
    trade_distribution: Dict[str, int]
    symbol_performance: Dict[str, float]


@dataclass
class BacktestResult:
    """Backtest sonucu."""
    symbol: str
    strategy_name: str
    timeframe: str
    metrics: BacktestMetrics
    trades: List[TradeRecord]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    monthly_returns: Dict[str, float]


class BacktestEngine:
    """Geriye dönük test motoru."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_pct: float = 0.0005,
        max_positions: int = 10,
        max_position_pct: float = 0.10,
    ) -> None:
        """BacktestEngine başlatır.

        Args:
            initial_capital: Başlangıç sermayesi (USDT).
            commission_rate: Komisyon oranı (ör. 0.001 = %0.1).
            slippage_pct: Slipaj yüzdesi.
            max_positions: Maksimum açık pozisyon sayısı.
            max_position_pct: Tek pozisyona ayrılan maksimum sermaye oranı.
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct

    def run(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, int], Optional[str]],
        symbol: str = "BTC/USDT",
        strategy_name: str = "unnamed",
        timeframe: str = "1h",
    ) -> BacktestResult:
        """Backtest çalıştırır.

        Args:
            data: OHLCV DataFrame (datetime index).
            strategy_fn: Sinyal üreten fonksiyon. Her barda çağrılır.
                Parametreler: (df, bar_index) -> "BUY", "SELL", veya None
            symbol: Sembol adı.
            strategy_name: Strateji adı.
            timeframe: Zaman dilimi.

        Returns:
            BacktestResult nesnesi.
        """
        logger.info(f"Backtest başlıyor: {symbol} | {strategy_name} | {timeframe}")
        logger.info(f"  Sermaye: ${self.initial_capital:,.2f} | Komisyon: %{self.commission_rate*100:.2f}")

        balance = self.initial_capital
        position = None  # {"side": "LONG"/"SHORT", "entry_price": float, "amount": float, "entry_time": datetime, "stop_loss": float, "take_profit": float}
        trades: List[TradeRecord] = []
        equity_curve = []
        timestamps = []

        for i in range(1, len(data)):
            current_bar = data.iloc[i]
            prev_bar = data.iloc[i - 1]
            current_price = float(current_bar["close"])
            current_time = data.index[i]

            # Pozisyon varsa PnL güncelle
            if position is not None:
                entry_price = position["entry_price"]
                amount = position["amount"]
                side = position["side"]

                if side == "LONG":
                    unrealized_pnl = (current_price - entry_price) * amount
                else:
                    unrealized_pnl = (entry_price - current_price) * amount

                # Stop loss / Take profit kontrolü
                exit_reason = None
                if side == "LONG":
                    if position.get("stop_loss") and current_bar["low"] <= position["stop_loss"]:
                        exit_reason = "STOP_LOSS"
                        current_price = position["stop_loss"]
                    elif position.get("take_profit") and current_bar["high"] >= position["take_profit"]:
                        exit_reason = "TAKE_PROFIT"
                        current_price = position["take_profit"]
                else:
                    if position.get("stop_loss") and current_bar["high"] >= position["stop_loss"]:
                        exit_reason = "STOP_LOSS"
                        current_price = position["stop_loss"]
                    elif position.get("take_profit") and current_bar["low"] <= position["take_profit"]:
                        exit_reason = "TAKE_PROFIT"
                        current_price = position["take_profit"]

                # Sinyal kontrolü
                if exit_reason is None:
                    signal = strategy_fn(data, i)
                    if signal == "SELL" and side == "LONG":
                        exit_reason = "SIGNAL"
                    elif signal == "BUY" and side == "SHORT":
                        exit_reason = "SIGNAL"

                # Pozisyon kapat
                if exit_reason:
                    if side == "LONG":
                        pnl = (current_price - entry_price) * amount
                    else:
                        pnl = (entry_price - current_price) * amount

                    commission = (entry_price * amount + current_price * amount) * self.commission_rate
                    net_pnl = pnl - commission
                    balance += net_pnl

                    hold_hours = (current_time - position["entry_time"]).total_seconds() / 3600

                    trade = TradeRecord(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        exit_price=current_price,
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        amount=amount,
                        pnl_usdt=net_pnl,
                        pnl_pct=(net_pnl / (entry_price * amount)) * 100,
                        commission=commission,
                        hold_duration_hours=hold_hours,
                        exit_reason=exit_reason,
                    )
                    trades.append(trade)
                    position = None

            # Pozisyon yoksa sinyal kontrolü
            if position is None:
                signal = strategy_fn(data, i)
                if signal in ("BUY", "SELL"):
                    # Pozisyon büyüklüğünü dengele: başlangıç sermayesine göre sınırla
                    max_position_value = min(
                        balance * self.max_position_pct,
                        self.initial_capital * self.max_position_pct * 2,  # Max 2x initial per trade
                    )
                    max_position_value = min(max_position_value, balance * 0.95)  # Bakiyenin %95'ini aşma
                    entry_price = current_price * (1 + self.slippage_pct if signal == "BUY" else 1 - self.slippage_pct)
                    amount = max_position_value / entry_price
                    commission = max_position_value * self.commission_rate
                    balance -= commission

                    # SL/TP hesapla (ATR bazlı)
                    atr = self._calculate_atr(data, i, period=14)
                    if signal == "BUY":
                        stop_loss = entry_price - 1.5 * atr
                        take_profit = entry_price + 3.0 * atr
                    else:
                        stop_loss = entry_price + 1.5 * atr
                        take_profit = entry_price - 3.0 * atr

                    position = {
                        "side": "LONG" if signal == "BUY" else "SHORT",
                        "entry_price": entry_price,
                        "amount": amount,
                        "entry_time": current_time,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    }

            # Equity kaydet
            if position:
                if position["side"] == "LONG":
                    unrealized = (current_price - position["entry_price"]) * position["amount"]
                else:
                    unrealized = (position["entry_price"] - current_price) * position["amount"]
                equity_curve.append(balance + unrealized)
            else:
                equity_curve.append(balance)
            timestamps.append(current_time)

        # Son pozisyonu kapat
        if position:
            last_price = float(data.iloc[-1]["close"])
            if position["side"] == "LONG":
                pnl = (last_price - position["entry_price"]) * position["amount"]
            else:
                pnl = (position["entry_price"] - last_price) * position["amount"]
            commission = (position["entry_price"] * position["amount"] + last_price * position["amount"]) * self.commission_rate
            net_pnl = pnl - commission
            balance += net_pnl

            hold_hours = (data.index[-1] - position["entry_time"]).total_seconds() / 3600
            trade = TradeRecord(
                symbol=symbol,
                side=position["side"],
                entry_price=position["entry_price"],
                exit_price=last_price,
                entry_time=position["entry_time"],
                exit_time=data.index[-1],
                amount=position["amount"],
                pnl_usdt=net_pnl,
                pnl_pct=(net_pnl / (position["entry_price"] * position["amount"])) * 100,
                commission=commission,
                hold_duration_hours=hold_hours,
                exit_reason="END_OF_DATA",
            )
            trades.append(trade)

        # Equity curve ve drawdown hesapla
        equity_series = pd.Series(equity_curve, index=timestamps)
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak * 100

        # Metrikleri hesapla
        metrics = self._calculate_metrics(
            equity_series, drawdown, trades, self.initial_capital
        )

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            timeframe=timeframe,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_series,
            drawdown_curve=drawdown,
            monthly_returns=metrics.monthly_returns,
        )

    def _calculate_atr(
        self,
        data: pd.DataFrame,
        current_idx: int,
        period: int = 14,
    ) -> float:
        """ATR hesaplar.

        Args:
            data: OHLCV DataFrame.
            current_idx: Mevcut bardaki indeks.
            period: ATR periyodu.

        Returns:
            ATR değeri.
        """
        start_idx = max(0, current_idx - period)
        high = data["high"].iloc[start_idx:current_idx + 1].values
        low = data["low"].iloc[start_idx:current_idx + 1].values
        close = data["close"].iloc[start_idx:current_idx + 1].values

        if len(close) < 2:
            return float(data["high"].iloc[current_idx] - data["low"].iloc[current_idx])

        tr_list = []
        for j in range(1, len(high)):
            tr = max(
                high[j] - low[j],
                abs(high[j] - close[j - 1]),
                abs(low[j] - close[j - 1]),
            )
            tr_list.append(tr)

        if not tr_list:
            return float(data["high"].iloc[current_idx] - data["low"].iloc[current_idx])

        return np.mean(tr_list[-period:])

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        drawdown: pd.Series,
        trades: List[TradeRecord],
        initial_capital: float,
    ) -> BacktestMetrics:
        """Kapsamlı metrikleri hesaplar.

        Args:
            equity_curve: Portföy değeri zaman serisi.
            drawdown: Drawdown zaman serisi.
            trades: İşlem listesi.
            initial_capital: Başlangıç sermayesi.

        Returns:
            BacktestMetrics nesnesi.
        """
        final_capital = float(equity_curve.iloc[-1])
        total_pnl = final_capital - initial_capital
        total_pnl_pct = (total_pnl / initial_capital) * 100

        # Süre hesaplama
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days <= 0:
            days = 1
        annual_return_pct = ((final_capital / initial_capital) ** (365 / days) - 1) * 100

        # İşlem metrikleri
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl_usdt > 0)
        losing_trades = sum(1 for t in trades if t.pnl_usdt <= 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        wins = [t for t in trades if t.pnl_usdt > 0]
        losses = [t for t in trades if t.pnl_usdt <= 0]

        avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
        largest_win_usdt = max([t.pnl_usdt for t in trades]) if trades else 0
        largest_loss_usdt = min([t.pnl_usdt for t in trades]) if trades else 0
        hold_hours = [t.hold_duration_hours for t in trades]
        avg_hold_hours = np.mean(hold_hours) if hold_hours else 0
        max_hold_hours = max(hold_hours) if hold_hours else 0
        min_hold_hours = min(hold_hours) if hold_hours else 0

        # Risk metrikleri
        daily_returns = equity_curve.pct_change().dropna()
        sharpe_ratio = self._calculate_sharpe(daily_returns)
        sortino_ratio = self._calculate_sortino(daily_returns)
        max_drawdown_pct = abs(float(drawdown.min()))
        max_dd_duration = self._calculate_max_dd_duration(drawdown)
        calmar_ratio = annual_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0

        gross_profit = sum(t.pnl_usdt for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_usdt for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win_usdt = np.mean([t.pnl_usdt for t in wins]) if wins else 0
        avg_loss_usdt = abs(np.mean([t.pnl_usdt for t in losses])) if losses else 0
        expectancy = (win_rate / 100 * avg_win_usdt) - ((1 - win_rate / 100) * avg_loss_usdt)
        recovery_factor = total_pnl / max_drawdown_pct if max_drawdown_pct > 0 else 0

        # Aylık getiriler
        monthly_returns = self._calculate_monthly_returns(equity_curve)

        # Sembol performansı
        symbol_perf = {}
        for t in trades:
            symbol_perf[t.symbol] = symbol_perf.get(t.symbol, 0) + t.pnl_usdt

        # İşlem dağılımı
        trade_dist = {"LONG": 0, "SHORT": 0}
        for t in trades:
            trade_dist[t.side] = trade_dist.get(t.side, 0) + 1

        return BacktestMetrics(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            annual_return_pct=annual_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win_usdt=largest_win_usdt,
            largest_loss_usdt=largest_loss_usdt,
            avg_hold_hours=avg_hold_hours,
            max_hold_hours=max_hold_hours,
            min_hold_hours=min_hold_hours,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration_days=max_dd_duration,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            expectancy=expectancy,
            recovery_factor=recovery_factor,
            monthly_returns=monthly_returns,
            trade_distribution=trade_dist,
            symbol_performance=symbol_perf,
        )

    def _calculate_sharpe(
        self,
        daily_returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Sharpe Ratio hesaplar."""
        if len(daily_returns) < 2 or daily_returns.std() == 0:
            return 0.0
        excess_returns = daily_returns - risk_free_rate / 365
        return float(np.sqrt(365) * excess_returns.mean() / excess_returns.std())

    def _calculate_sortino(
        self,
        daily_returns: pd.Series,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Sortino Ratio hesaplar."""
        if len(daily_returns) < 2:
            return 0.0
        excess_returns = daily_returns - risk_free_rate / 365
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0
        return float(np.sqrt(365) * excess_returns.mean() / downside_returns.std())

    def _calculate_max_dd_duration(self, drawdown: pd.Series) -> float:
        """Maksimum drawdown süresini hesaplar (gün cinsinden)."""
        is_dd = drawdown < 0
        if not is_dd.any():
            return 0.0

        dd_groups = (~is_dd).cumsum()
        dd_durations = is_dd.groupby(dd_groups).sum()
        max_days = dd_durations.max() / 24 if len(dd_durations) > 0 else 0
        return float(max_days)

    def _calculate_monthly_returns(
        self,
        equity_curve: pd.Series,
    ) -> Dict[str, float]:
        """Aylık getirileri hesaplar."""
        monthly = equity_curve.resample("M").last()
        monthly_returns = monthly.pct_change().dropna()
        return {
            str(date.strftime("%Y-%m")): float(ret * 100)
            for date, ret in monthly_returns.items()
        }


class MultiSymbolBacktest:
    """Çoklu sembol backtest motoru."""

    def __init__(self, engine: BacktestEngine) -> None:
        """MultiSymbolBacktest başlatır.

        Args:
            engine: BacktestEngine instance'ı.
        """
        self.engine = engine
        self.results: List[BacktestResult] = []

    def run_portfolio_backtest(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy_fn: Callable[[pd.DataFrame, int], Optional[str]],
        strategy_name: str = "portfolio_strategy",
        timeframe: str = "1h",
    ) -> Dict[str, BacktestResult]:
        """Portföy bazlı backtest çalıştırır.

        Args:
            data_dict: {symbol: DataFrame} sözlüğü.
            strategy_fn: Sinyal üreten fonksiyon.
            strategy_name: Strateji adı.
            timeframe: Zaman dilimi.

        Returns:
            {symbol: BacktestResult} sözlüğü.
        """
        results = {}
        for symbol, data in data_dict.items():
            if data.empty:
                continue
            result = self.engine.run(
                data=data,
                strategy_fn=strategy_fn,
                symbol=symbol,
                strategy_name=strategy_name,
                timeframe=timeframe,
            )
            results[symbol] = result
            self.results.append(result)

        return results

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Portföy özetini hesaplar."""
        if not self.results:
            return {}

        total_trades = sum(r.metrics.total_trades for r in self.results)
        total_pnl = sum(r.metrics.total_pnl for r in self.results)
        winning = sum(r.metrics.winning_trades for r in self.results)
        losing = sum(r.metrics.losing_trades for r in self.results)

        all_trades = []
        for r in self.results:
            all_trades.extend(r.trades)

        # En iyi ve en kötü semboller
        symbol_pnl = {}
        for r in self.results:
            symbol_pnl[r.symbol] = r.metrics.total_pnl

        best_symbol = max(symbol_pnl, key=symbol_pnl.get) if symbol_pnl else "N/A"
        worst_symbol = min(symbol_pnl, key=symbol_pnl.get) if symbol_pnl else "N/A"

        return {
            "total_symbols": len(self.results),
            "total_trades": total_trades,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": (winning / total_trades * 100) if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / self.engine.initial_capital) * 100,
            "best_symbol": best_symbol,
            "best_symbol_pnl": symbol_pnl.get(best_symbol, 0),
            "worst_symbol": worst_symbol,
            "worst_symbol_pnl": symbol_pnl.get(worst_symbol, 0),
            "avg_sharpe": np.mean([r.metrics.sharpe_ratio for r in self.results]),
            "avg_sortino": np.mean([r.metrics.sortino_ratio for r in self.results]),
            "avg_max_dd": np.mean([r.metrics.max_drawdown_pct for r in self.results]),
            "avg_win_rate": np.mean([r.metrics.win_rate for r in self.results]),
            "avg_profit_factor": np.mean([r.metrics.profit_factor for r in self.results if r.metrics.profit_factor != float("inf")]),
        }
