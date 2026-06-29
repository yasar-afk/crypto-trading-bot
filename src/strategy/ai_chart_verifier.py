# -*- coding: utf-8 -*-
"""
src/strategy/ai_chart_verifier.py — Grafik Tabanlı AI Sinyal Doğrulayıcı
Xiaomi MiMo v2.5 (multimodal, vision destekli) kullanır.

Her sinyal açılmadan önce:
  1. OHLCV verisinden mum grafiği oluşturur (matplotlib)
  2. Grafiği base64'e çevirir
  3. MiMo v2.5'e gönderir (OpenAI uyumlu API)
  4. TRADE / SKIP kararına göre işlem açılır veya atlanır

API: https://api.xiaomimimo.com/v1
Dok: api.xiaomimimo.com
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # GUI gerektirmeyen backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("ai_chart_verifier")

# ─── Karar dosyası (denetim izi) ──────────────────────────────────────────────
DECISIONS_LOG = "logs/ai_decisions.jsonl"

# ─── Zengin teknik indikatörlü prompt ────────────────────────────────────────
_PROMPT_TEMPLATE = """You are an expert crypto technical analyst. Analyze the chart image and the indicators below to decide if this trade setup is valid.

=== TRADE SETUP ===
Symbol     : {symbol}
Timeframe  : {tf}
Signal     : {direction}
Entry      : {price}
Stop-Loss  : {sl}
Take-Profit: {tp}
Risk/Reward: {rr:.2f}
Timestamp  : {timestamp}

=== TECHNICAL INDICATORS (calculated from latest candles) ===
RSI(14)        : {rsi:.1f}  {rsi_note}
ADX(14)        : {adx:.1f}  {adx_note}
EMA Cross      : {ema_cross}
EMA20 vs Price : {ema20_pos}  (price is {ema20_diff:+.2f}% from EMA20)
EMA50 vs Price : {ema50_pos}  (price is {ema50_diff:+.2f}% from EMA50)
Volume Change  : {vol_chg:+.1f}% vs 20-bar average
ATR(14) / Price: {atr_pct:.2f}%  (volatility)
Trend (20 bars): {trend_pct:+.2f}%  ({trend_note})
Market Structure: Support: {support_levels} | Resistance: {resistance_levels}
Last 3 candles :
  {last3}

=== MARKET SENTIMENT ===
Fear & Greed Index: {fear_greed}

=== DECISION RULES ===
DEFAULT BEHAVIOR: When in doubt, SKIP. Only TRADE when confidence is HIGH.

APPROVE (TRADE) ONLY IF ALL TRUE:
  BUY:
    - RSI between 30-65 (not overbought)
    - Price above EMA50 OR ADX > 20 with bullish momentum
    - Last 2 candles show upward momentum (higher lows)
    - Volume is stable or increasing
    - Price is in discount zone (below 50% of recent range)
  SELL:
    - RSI between 35-70 (not oversold)
    - Price below EMA50 OR ADX > 20 with bearish momentum
    - Last 2 candles show downward momentum (lower highs)
    - Volume is stable or increasing
    - Price is in premium zone (above 50% of recent range)

REJECT (SKIP) IF ANY TRUE:
  - RSI > 75 for BUY or RSI < 25 for SELL (extreme levels)
  - ADX < 15 AND ATR < 0.3% (no trend, no volatility)
  - Price below EMA50 for BUY or above EMA50 for SELL (trend misalignment)
  - Last 2 candles show strong reversal against signal direction
  - Volume collapsing (< -50% vs average)
  - Price dropped > 5% in last 10 candles (dumping, catch a falling knife)
  - Price pumped > 5% in last 10 candles (FOMO top, avoid chasing)
  - Choppy sideways action with no clear structure

CRITICAL WARNING — Low-priced coins ($<0.01):
  - ATR-based SL/TP calculations are unreliable at these prices
  - Default to SKIP unless ALL indicators align perfectly

General Rules:
  - If chart contradicts indicators, trust the indicators.
  - Do NOT reject based on TP distance alone.
  - REASON is MANDATORY — empty reason = automatic SKIP.

Reply in EXACTLY this format (nothing else). The REASON must be written in Turkish (min 10 characters):
DECISION: TRADE
CONFIDENCE: HIGH / MEDIUM / LOW
REASON: <10-20 words in Turkish explaining why>

or

DECISION: SKIP
CONFIDENCE: HIGH / MEDIUM / LOW
REASON: <10-20 words in Turkish explaining why>"""


class AIChartVerifier:
    """Xiaomi MiMo v2.5 vision modeli ile grafik tabanlı sinyal doğrulayıcı."""

    def __init__(self) -> None:
        # Önce pydantic settings üzerinden oku (get_settings .env'i otomatik yükler)
        # Fallback olarak os.getenv kullan
        try:
            from src.config.settings import get_settings
            cfg = get_settings()
            self._api_key  = cfg.mimo_api_key  or os.getenv("MIMO_API_KEY", "")
            self._model    = cfg.mimo_model    or os.getenv("MIMO_MODEL", "mimo-v2.5")
            self._base_url = cfg.mimo_base_url or os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
        except Exception:
            self._api_key  = os.getenv("MIMO_API_KEY", "")
            self._model    = os.getenv("MIMO_MODEL", "mimo-v2.5")
            self._base_url = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")

        self._enabled = bool(self._api_key)
        self._client  = None

        if self._enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
                logger.info(f"AIChartVerifier aktif | Model: {self._model} | URL: {self._base_url}")
            except ImportError:
                logger.warning("'openai' kutuphanesi yok. pip install openai")
                self._enabled = False
        else:
            logger.warning("MIMO_API_KEY bulunamadi — AI dogrulama devre disi, tum sinyaller gececek.")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ─────────────────────────────────────────────────────────────────────────
    # ANA DOĞRULAMA FONKSİYONU
    # ─────────────────────────────────────────────────────────────────────────

    def verify(
        self,
        symbol: str,
        direction: str,
        price: float,
        sl: float,
        tp: float,
        df: pd.DataFrame,
        timeframe: str = "1h",
        timestamp: str = None,
    ) -> dict:
        """
        Sinyali MiMo v2.5 ile doğrular.

        Returns:
            {
                "approved": True/False,
                "decision": "TRADE" / "SKIP",
                "confidence": "HIGH" / "MEDIUM" / "LOW",
                "reason": "...",
                "model": "mimo-v2.5",
                "skipped_ai": True/False
            }
        """
        if not self._enabled:
            return {"approved": True, "decision": "TRADE", "confidence": "MEDIUM",
                    "reason": "AI devre disi — otomatik onay", "skipped_ai": True}

        try:
            # 1. Grafik oluştur → base64
            image_b64 = self._generate_chart(df, symbol, direction, price, sl, tp, timeframe=timeframe)

            # 2. Teknik indikatörleri hesapla
            ind = self._calc_indicators(df, price)

            # 3. Prompt hazırla
            def _fmt(v):
                return f"${v:,.6f}".rstrip("0").rstrip(".")

            # Risk/Reward (R:R) hesabı
            denom = abs(price - sl)
            rr = abs(tp - price) / denom if denom > 0 else 0.0

            # Sinyal zaman damgası
            if timestamp is None:
                timestamp = pd.Timestamp.now().isoformat()

            # Fear & Greed Index
            fear_greed = "N/A"
            try:
                import requests
                fng_resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
                fng_data = fng_resp.json()["data"][0]
                fear_greed = f"{fng_data['value']} ({fng_data['value_classification']})"
            except Exception:
                pass

            prompt = _PROMPT_TEMPLATE.format(
                symbol=symbol,
                tf=timeframe,
                direction=direction,
                price=_fmt(price),
                sl=_fmt(sl),
                tp=_fmt(tp),
                rr=rr,
                timestamp=timestamp,
                fear_greed=fear_greed,
                **ind,
            )

            # 3. MiMo API'ye gönder
            response_text = self._call_api(prompt, image_b64)

            # 4. Yanıtı parse et
            result = self._parse_response(response_text)
            result["model"] = self._model
            result["skipped_ai"] = False

            # 5. Kararı logla
            self._log_decision(symbol, direction, result)

            emoji = "[TRADE]" if result["approved"] else "[SKIP]"
            logger.info(f"[MiMo] {symbol} {direction} -> {emoji} | {result['reason']}")
            return result

        except Exception as e:
            logger.error(f"[MiMo] {symbol} dogrulama hatasi: {e} — reddedildi (fail-close)")
            return {"approved": False, "decision": "SKIP",
                    "reason": f"AI hatasi: {e}", "skipped_ai": True}

    # ─────────────────────────────────────────────────────────────────────────
    # TEKNİK İNDİKATÖR HESAPLAMA
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_indicators(self, df: pd.DataFrame, price: float) -> dict:
        """
        Son mumlardan RSI, ADX, EMA, Hacim ve ATR hesaplar.
        Prompt template'e unpack edilmek uzere dict dondurur.
        """
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else None

        # ── RSI(14) ───────────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs    = gain / loss.replace(0, 1e-9)
        rsi   = float(100 - 100 / (1 + rs.iloc[-1]))

        if rsi >= 70:
            rsi_note = "(OVERBOUGHT - risky for BUY)"
        elif rsi <= 30:
            rsi_note = "(OVERSOLD - risky for SELL)"
        elif rsi >= 55:
            rsi_note = "(bullish zone)"
        elif rsi <= 45:
            rsi_note = "(bearish zone)"
        else:
            rsi_note = "(neutral)"

        # ── ADX(14) ───────────────────────────────────────────────────────
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.ewm(com=13, adjust=False).mean()

        up_move   = high.diff()
        down_move = -low.diff()
        dm_plus   = pd.Series(0.0, index=df.index)
        dm_minus  = pd.Series(0.0, index=df.index)
        dm_plus[up_move   > down_move] = up_move[up_move   > down_move].clip(lower=0)
        dm_minus[down_move > up_move]  = down_move[down_move > up_move].clip(lower=0)
        di_plus  = 100 * dm_plus.ewm(com=13, adjust=False).mean()  / atr14.replace(0, 1e-9)
        di_minus = 100 * dm_minus.ewm(com=13, adjust=False).mean() / atr14.replace(0, 1e-9)
        dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-9)
        adx      = float(dx.ewm(com=13, adjust=False).mean().iloc[-1])

        if adx >= 25:
            adx_note = "(STRONG TREND)"
        elif adx >= 15:
            adx_note = "(moderate trend)"
        else:
            adx_note = "(weak/no trend - caution)"

        # ── EMA 20 & 50 ───────────────────────────────────────────────────
        ema20_val = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50_val = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema20_diff = (price - ema20_val) / ema20_val * 100
        ema50_diff = (price - ema50_val) / ema50_val * 100
        ema20_pos  = "ABOVE EMA20" if ema20_diff > 0 else "BELOW EMA20"
        ema50_pos  = "ABOVE EMA50" if ema50_diff > 0 else "BELOW EMA50"

        # EMA Cross: EMA20 ve EMA50 birbirine göre durumu
        if ema20_val > ema50_val:
            ema_cross = "EMA20 above EMA50 - bullish"
        elif ema20_val < ema50_val:
            ema_cross = "EMA20 below EMA50 - bearish"
        else:
            ema_cross = "EMA20 equal to EMA50 - neutral"

        # ── Hacim ─────────────────────────────────────────────────────────
        if volume is not None and len(volume) >= 21:
            avg_vol = float(volume.iloc[-21:-1].mean())
            cur_vol = float(volume.iloc[-1])
            vol_chg = (cur_vol - avg_vol) / (avg_vol or 1) * 100
        else:
            vol_chg = 0.0

        # ── ATR (volatilite) ──────────────────────────────────────────────
        atr_pct = float(atr14.iloc[-1]) / price * 100

        # ── 20 Mum Trend ──────────────────────────────────────────────────
        if len(close) >= 20:
            old_price = float(close.iloc[-20])
            trend_pct = (price - old_price) / old_price * 100
        else:
            trend_pct = 0.0

        if trend_pct > 3:
            trend_note = "strong uptrend"
        elif trend_pct > 0.5:
            trend_note = "mild uptrend"
        elif trend_pct < -3:
            trend_note = "strong downtrend"
        elif trend_pct < -0.5:
            trend_note = "mild downtrend"
        else:
            trend_note = "sideways"

        # ── Market Structure: Son Önemli Destek/Direnç Seviyeleri ─────────
        supports = []
        resistances = []
        tail_df = df.tail(60)
        for i in range(len(tail_df) - 5):
            if i < 5:
                continue
            val_low = float(tail_df["low"].iloc[i])
            val_high = float(tail_df["high"].iloc[i])
            
            # i-5 ile i+5 arasındaki en düşük/en yüksek değer kontrolü
            if val_low == tail_df["low"].iloc[i-5:i+6].min():
                supports.append(val_low)
            if val_high == tail_df["high"].iloc[i-5:i+6].max():
                resistances.append(val_high)
        
        # Tekil değerleri al ve sırala
        supports = list(set(supports))
        resistances = list(set(resistances))
        
        sups_below = sorted([s for s in supports if s < price], reverse=True)
        ress_above = sorted([r for r in resistances if r > price])
        
        # Boş kalması durumunda son 60 mumun min/max değerlerini fallback yap
        if not sups_below:
            sups_below = [float(tail_df["low"].min())]
        if not ress_above:
            ress_above = [float(tail_df["high"].max())]
            
        support_levels = ", ".join([f"${s:,.6f}".rstrip("0").rstrip(".") for s in sups_below[:2]])
        resistance_levels = ", ".join([f"${r:,.6f}".rstrip("0").rstrip(".") for r in ress_above[:2]])

        # ── Son 3 mum özeti (Detaylı) ─────────────────────────────────────
        last3_desc = []
        tail3 = df.tail(3).copy()
        for idx, (_, row) in enumerate(tail3.iterrows(), 1):
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            direction_str = "Bullish" if c >= o else "Bearish"
            chg = (c - o) / o * 100
            lbl = "oldest" if idx == 1 else ("middle" if idx == 2 else "latest")
            last3_desc.append(
                f"Candle {idx} ({lbl}): O: {o:.6g}, H: {h:.6g}, L: {l:.6g}, C: {c:.6g} [{direction_str}, {chg:+.2f}%]"
            )
        last3 = "\n  ".join(last3_desc)

        return dict(
            rsi=rsi, rsi_note=rsi_note,
            adx=adx, adx_note=adx_note,
            ema20_pos=ema20_pos, ema20_diff=ema20_diff,
            ema50_pos=ema50_pos, ema50_diff=ema50_diff,
            ema_cross=ema_cross,
            vol_chg=vol_chg,
            atr_pct=atr_pct,
            trend_pct=trend_pct, trend_note=trend_note,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            last3=last3,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GRAFİK OLUŞTURMA
    # ─────────────────────────────────────────────────────────────────────────

    _TF_MAP = {"1m": "1", "5m": "5", "15m": "15", "30m": "30",
               "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W"}

    def _try_tradingview_chart(self, symbol: str, timeframe: str) -> Optional[str]:
        """TradingView'dan temiz grafik goruntusu cekmeyi dener (widget embed + Selenium)."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            import time, base64

            tv_symbol = symbol.replace("/", "")
            tf = self._TF_MAP.get(timeframe, "240")
            embed_url = (
                f"https://s.tradingview.com/widgetembed/"
                f"?symbol=BINANCE:{tv_symbol}&interval={tf}"
                f"&hidesidetoolbar=1&hideideas=1&theme=dark&style=1&timezone=exchange"
            )

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1280,720")

            driver = webdriver.Chrome(options=chrome_options)
            try:
                driver.get(embed_url)
                time.sleep(6)
                png = driver.get_screenshot_as_png()
                if len(png) > 5000:
                    b64 = base64.b64encode(png).decode("utf-8")
                    return f"data:image/png;base64,{b64}"
            finally:
                driver.quit()
        except Exception:
            pass
        return None


    def _generate_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        direction: str,
        price: float,
        sl: float,
        tp: float,
        last_n: int = 120,
        timeframe: str = "1h",
        use_tradingview: bool = True,
    ) -> str:
        """
        Grafik olusturur: Once TradingView dener, basarisiz olursa matplotlib ile olusturur.
        PNG -> base64 data URI dondurur.
        """
        if use_tradingview:
            tv_chart = self._try_tradingview_chart(symbol, timeframe)
            if tv_chart:
                return tv_chart

        df = df.tail(last_n).copy().reset_index(drop=True)
        n  = len(df)

        UP_COLOR   = "#26a69a"
        DOWN_COLOR = "#ef5350"
        BG_COLOR   = "#ffffff"
        PANEL_BG   = "#f8f9fa"
        GRID_COLOR = "#e0e3e8"
        TEXT_COLOR = "#1a1a2e"
        EMA20_CLR  = "#ff9800"
        EMA50_CLR  = "#2196f3"
        ENTRY_CLR  = "#1565c0"
        SL_CLR     = "#c62828"
        TP_CLR     = "#1b5e20"

        # ── 4:3 oran, 2 panel ────────────────────────────────────────────
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1,
            figsize=(12, 8),
            facecolor=BG_COLOR,
            gridspec_kw={"height_ratios": [5, 1.2], "hspace": 0.0},
        )
        ax_price.set_facecolor(PANEL_BG)
        ax_vol.set_facecolor(PANEL_BG)

        # ── Grid ─────────────────────────────────────────────────────────
        for _ax in (ax_price, ax_vol):
            _ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
            _ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.4, zorder=0)
            _ax.set_axisbelow(True)

        # ── Mumlar + Hacim ────────────────────────────────────────────────
        price_range_chart = df["high"].max() - df["low"].min()
        for idx, row in df.iterrows():
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            is_up = c >= o
            body_color = UP_COLOR if is_up else DOWN_COLOR
            edge_color = "#1b5e20" if is_up else "#b71c1c"
            body_h = max(abs(c - o), price_range_chart * 0.008)
            ax_price.bar(idx, body_h, bottom=min(o, c),
                         color=body_color, edgecolor=edge_color,
                         linewidth=0.5, width=0.75, zorder=2)
            ax_price.plot([idx, idx], [l, h], color=edge_color,
                          linewidth=1.2, zorder=2)

            # Hacim
            vol = float(row.get("volume", 0))
            ax_vol.bar(idx, vol, color=body_color, edgecolor=edge_color,
                       linewidth=0.3, width=0.75, alpha=0.8, zorder=2)

        # ── EMA 20 & 50 ───────────────────────────────────────────────────
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        xs    = range(n)
        ax_price.plot(xs, ema20.values, color=EMA20_CLR,
                      linewidth=1.4, label="EMA 20", zorder=3)
        ax_price.plot(xs, ema50.values, color=EMA50_CLR,
                      linewidth=1.6, label="EMA 50", zorder=3)

        # ── Entry / SL / TP yatay seritler ───────────────────────────────
        all_prices = [df["low"].min(), df["high"].max(), price, sl, tp]
        y_min = min(all_prices)
        y_max = max(all_prices)
        margin = (y_max - y_min) * 0.08
        y_min -= margin
        y_max += margin
        price_range = y_max - y_min or y_max * 0.01

        # TP bantı
        ax_price.axhspan(min(price, tp), max(price, tp),
                         alpha=0.07, color=TP_CLR, zorder=1)
        # SL bandı
        ax_price.axhspan(min(price, sl), max(price, sl),
                         alpha=0.07, color=SL_CLR, zorder=1)

        # Yatay çizgiler
        ax_price.axhline(price, color=ENTRY_CLR, linewidth=1.5,
                         linestyle="--", zorder=4, alpha=0.9)
        ax_price.axhline(sl, color=SL_CLR, linewidth=1.3,
                         linestyle="-.", zorder=4, alpha=0.9)
        ax_price.axhline(tp, color=TP_CLR, linewidth=1.3,
                         linestyle="-.", zorder=4, alpha=0.9)

        # Fiyat etiketleri (grafik icine, saga yasli)
        x_label = n + 0.5
        ax_price.text(
            x_label, price, f" ENTRY {price:.5g}",
            fontsize=7.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=ENTRY_CLR, ec="none", alpha=0.85),
            ha="left", va="center", zorder=6,
        )
        ax_price.text(
            x_label, sl, f" SL {sl:.5g}",
            fontsize=7.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=SL_CLR, ec="none", alpha=0.85),
            ha="left", va="center", zorder=6,
        )
        ax_price.text(
            x_label, tp, f" TP {tp:.5g}",
            fontsize=7.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=TP_CLR, ec="none", alpha=0.85),
            ha="left", va="center", zorder=6,
        )

        ax_price.set_ylim(y_min, y_max)

        # ── Başlık (sol üst — sadece sembol ve zaman dilimi) ─────────────
        ax_price.set_title(
            f"{symbol}  ·  {timeframe}",
            fontsize=13, fontweight="bold",
            color=TEXT_COLOR, loc="left", pad=10,
        )

        # ── BUY / SELL rozeti (sağ üst, büyük ve belirgin) ───────────────
        is_buy   = direction == "BUY"
        dir_text = "▲  BUY" if is_buy else "▼  SELL"
        dir_clr  = "#1b5e20" if is_buy else "#b71c1c"
        ax_price.text(
            0.993, 0.975, dir_text,
            transform=ax_price.transAxes,
            fontsize=18, fontweight="bold",
            color="white", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.55",
                      fc=dir_clr, ec="none", alpha=0.97),
            zorder=8,
        )

        # ── Giriş/Çıkış ok işaretleri ────────────────────────────────────
        entry_idx = n - 5
        if is_buy:
            ax_price.annotate(
                "", xy=(entry_idx, price), xytext=(entry_idx, price - price_range * 0.06),
                arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=2.5),
                zorder=7,
            )
            ax_price.text(entry_idx, price - price_range * 0.08, "GİRİŞ",
                          fontsize=7, color="#1b5e20", ha="center", fontweight="bold")
        else:
            ax_price.annotate(
                "", xy=(entry_idx, price), xytext=(entry_idx, price + price_range * 0.06),
                arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=2.5),
                zorder=7,
            )
            ax_price.text(entry_idx, price + price_range * 0.08, "GİRİŞ",
                          fontsize=7, color="#b71c1c", ha="center", fontweight="bold")

        # SL oku (kırmızı, aşağı yönlü)
        ax_price.annotate(
            "", xy=(entry_idx, sl), xytext=(entry_idx, sl + price_range * 0.03),
            arrowprops=dict(arrowstyle="->", color=SL_CLR, lw=1.8),
            zorder=7,
        )
        ax_price.text(entry_idx, sl + price_range * 0.05, "SL",
                      fontsize=6.5, color=SL_CLR, ha="center", fontweight="bold")

        # TP oku (yeşil, yukarı yönlü)
        ax_price.annotate(
            "", xy=(entry_idx, tp), xytext=(entry_idx, tp - price_range * 0.03),
            arrowprops=dict(arrowstyle="->", color=TP_CLR, lw=1.8),
            zorder=7,
        )
        ax_price.text(entry_idx, tp - price_range * 0.05, "TP",
                      fontsize=6.5, color=TP_CLR, ha="center", fontweight="bold")

        # ── Legend (EMA çizgileri için, rozeti çakışmadan göster) ─────────
        ax_price.legend(
            fontsize=9, loc="upper left",
            facecolor="white", edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR, framealpha=0.95,
            bbox_to_anchor=(0.0, 0.88),
        )

        # ── Eksen stilleri ────────────────────────────────────────────────
        for _ax in (ax_price, ax_vol):
            _ax.tick_params(colors=TEXT_COLOR, labelsize=8.5, length=3)
            for spine in _ax.spines.values():
                spine.set_edgecolor(GRID_COLOR)
                spine.set_linewidth(0.8)

        ax_price.set_xlim(-1, n + 8)
        ax_vol.set_xlim(-1, n + 8)
        ax_price.tick_params(labelbottom=False)
        ax_vol.set_yticks([])
        ax_vol.set_xlabel(
            "← Daha Eski" + " " * 50 + "Daha Yeni →",
            fontsize=8.5, color="#777", labelpad=5,
        )

        fig.subplots_adjust(left=0.09, right=0.92,
                            top=0.93, bottom=0.07, hspace=0.0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=BG_COLOR)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # ─────────────────────────────────────────────────────────────────────────
    # API ÇAĞRISI
    # ─────────────────────────────────────────────────────────────────────────

    def _call_api(self, prompt: str, image_b64: str) -> str:
        """MiMo v2.5'e grafik + prompt gönderir, ham yanıt döndürür."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1500,    # MiMo v2.5 akıl yürütme (reasoning) yaptığı için yüksek limit gerekir
            temperature=0.1,    # Tutarlı, deterministik karar
        )
        return response.choices[0].message.content.strip()

    # ─────────────────────────────────────────────────────────────────────────
    # YANIT PARSE
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> dict:
        """'DECISION: TRADE\nCONFIDENCE: HIGH\nREASON: ...' formatını parse eder."""
        approved = False
        reason   = text[:240]
        decision = "SKIP"
        confidence = "MEDIUM"

        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("DECISION:"):
                decision = line.split(":", 1)[1].strip().upper()
                approved = decision == "TRADE"
            elif line.upper().startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        if len(reason.strip()) < 10:
            logger.warning(f"[MiMo] Bos/kisa reason: '{reason}' — otomatik SKIP")
            approved = False
            decision = "SKIP"
            reason = f"Reason yetersiz: {reason}"

        return {
            "approved": approved,
            "decision": decision,
            "confidence": confidence,
            "reason": reason
        }

    # ─────────────────────────────────────────────────────────────────────────
    # AÇIK POZİSYON İNCELEME
    # ─────────────────────────────────────────────────────────────────────────

    _REVIEW_PROMPT = """You are reviewing an OPEN crypto trade position. Decide if it should be kept, closed early, or partially closed.

=== POSITION DETAILS ===
Symbol: {symbol} | Direction: {direction}
Entry: {entry_price} | Current price: {current_price} | Unrealized PnL: {pnl_pct}%
Stop-Loss: {sl} | Take-Profit: {tp}
Current Risk/Reward: {current_rr:.2f}
Held for: {held_hours}h
Timestamp: {timestamp}

=== TECHNICAL INDICATORS (calculated from latest candles) ===
RSI(14)        : {rsi:.1f}  {rsi_note}
ADX(14)        : {adx:.1f}  {adx_note}
EMA Cross      : {ema_cross}
EMA20 vs Price : {ema20_pos}  (price is {ema20_diff:+.2f}% from EMA20)
EMA50 vs Price : {ema50_pos}  (price is {ema50_diff:+.2f}% from EMA50)
Volume Change  : {vol_chg:+.1f}% vs 20-bar average
ATR(14) / Price: {atr_pct:.2f}%  (volatility)
Trend (20 bars): {trend_pct:+.2f}%  ({trend_note})
Market Structure: Support: {support_levels} | Resistance: {resistance_levels}
Last 3 candles :
  {last3}

=== DECISION RULES ===
MANDATORY CLOSE RULES (override everything):
  - PnL < -10%: ALWAYS CLOSE. No exceptions.
  - PnL < -7% AND held > 24h: CLOSE. Trade thesis invalidated.
  - PnL < -5% AND ADX < 15: CLOSE. No momentum to recover.
  - RSI < 20 for BUY or RSI > 80 for SELL: CLOSE. Extreme exhaustion.

PARTIAL_CLOSE RULES:
  - PnL > +8% AND near resistance/support: PARTIAL_CLOSE (lock 50%)
  - PnL > +15%: PARTIAL_CLOSE (lock profit, let rest run)
  - Price touched TP zone but reversed: PARTIAL_CLOSE immediately

HOLD RULES (strict — only if ALL true):
  - Trend direction matches position direction
  - Price above EMA50 (for BUY) or below EMA50 (for SELL)
  - RSI between 35-65 (neutral zone, room to move)
  - ADX > 18 (trend is strong enough)
  - No reversal candles in last 3 bars
  - Held < 36h (not overstaying)

DEFAULT: If unsure between HOLD and CLOSE, choose CLOSE. Protect capital first.

Time Context:
  - < 12h: Allow room unless structure breaks
  - 12h - 36h: Normal, evaluate trend
  - > 36h: Must show clear momentum or CLOSE

General Rules:
  - If chart contradicts indicators, trust the indicators.
  - REASON is MANDATORY — empty reason = automatic CLOSE.

Reply in EXACTLY this format (nothing else). The REASON must be written in Turkish (min 10 characters):
ACTION: HOLD / CLOSE / PARTIAL_CLOSE
CONFIDENCE: HIGH / MEDIUM / LOW
REASON: <10-20 words in Turkish explaining why>"""

    def review_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        current_price: float,
        sl: float,
        tp: float,
        df: pd.DataFrame,
        opened_at: str = "",
        timeframe: str = "1h",
    ) -> dict:
        """
        Açık bir pozisyonu MiMo v2.5 ile inceler.

        Returns:
            {
                "action":     "HOLD" / "CLOSE" / "PARTIAL_CLOSE",
                "should_close": True/False,
                "confidence": "HIGH" / "MEDIUM" / "LOW",
                "reason":     "...",
                "skipped_ai": True/False,
                "pnl_pct":    float
            }
        """
        if not self._enabled:
            return {"action": "HOLD", "should_close": False, "confidence": "MEDIUM",
                    "reason": "AI devre disi", "skipped_ai": True, "pnl_pct": 0.0}

        try:
            # Açık kalma süresi
            held_hours = 0
            if opened_at:
                try:
                    opened = pd.Timestamp(opened_at)
                    held_hours = int((pd.Timestamp.now() - opened).total_seconds() / 3600)
                except Exception:
                    pass

            # Unrealized PnL % ve Risk/Ödül Oranı
            if direction == "BUY":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                denom = abs(current_price - sl)
                current_rr = abs(tp - current_price) / denom if denom > 0 else 0.0
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                denom = abs(sl - current_price)
                current_rr = abs(current_price - tp) / denom if denom > 0 else 0.0

            # Grafik oluştur (entry ve current fiyatı işaretle)
            image_b64 = self._generate_chart(df, symbol, direction,
                                              current_price, sl, tp, timeframe=timeframe)

            # Teknik indikatörleri hesapla
            ind = self._calc_indicators(df, current_price)

            prompt = self._REVIEW_PROMPT.format(
                symbol=symbol,
                direction=direction,
                entry_price=f"${entry_price:,.6f}".rstrip("0").rstrip("."),
                current_price=f"${current_price:,.6f}".rstrip("0").rstrip("."),
                pnl_pct=f"{pnl_pct:+.2f}",
                sl=f"${sl:,.6f}".rstrip("0").rstrip("."),
                tp=f"${tp:,.6f}".rstrip("0").rstrip("."),
                current_rr=current_rr,
                held_hours=held_hours,
                timestamp=pd.Timestamp.now().isoformat(),
                **ind,
            )

            response_text = self._call_api(prompt, image_b64)

            # Parse ACTION, CONFIDENCE, REASON
            action = "HOLD"
            confidence = "MEDIUM"
            reason = response_text[:240]
            for line in response_text.splitlines():
                line = line.strip()
                if line.upper().startswith("ACTION:"):
                    raw_action = line.split(":", 1)[1].strip().upper()
                    if raw_action in ("PARTIAL CLOSE", "PARTIAL_CLOSE"):
                        action = "PARTIAL_CLOSE"
                    else:
                        action = raw_action
                elif line.upper().startswith("CONFIDENCE:"):
                    confidence = line.split(":", 1)[1].strip().upper()
                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            if len(reason.strip()) < 10:
                logger.warning(f"[MiMo Review] Bos/kisa reason: '{reason}' — otomatik CLOSE")
                action = "CLOSE"
                should_close = True
                reason = f"Reason yetersiz: {reason}"

            should_close = action == "CLOSE"
            result = {
                "action": action,
                "should_close": should_close,
                "confidence": confidence,
                "reason": reason,
                "model": self._model,
                "skipped_ai": False,
                "pnl_pct": round(pnl_pct, 2),
            }

            # Logla
            try:
                Path("logs").mkdir(exist_ok=True)
                entry_log = {
                    "ts":         pd.Timestamp.now().isoformat(),
                    "type":       "position_review",
                    "symbol":     symbol,
                    "direction":  direction,
                    "pnl_pct":    round(pnl_pct, 2),
                    "action":     action,
                    "confidence": confidence,
                    "reason":     reason,
                    "model":      self._model,
                }
                with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry_log, ensure_ascii=False) + "\n")
            except Exception:
                pass

            tag = f"[{action}]"
            logger.info(f"[MiMo] Pozisyon inceleme {symbol} {direction} "
                        f"PnL:{pnl_pct:+.2f}% -> {tag} | Conf: {confidence} | {reason}")
            return result

        except Exception as e:
            logger.error(f"[MiMo] {symbol} pozisyon inceleme hatasi: {e} — otomatik CLOSE")
            return {"action": "CLOSE", "should_close": True,
                    "reason": f"AI hatasi: {e}", "skipped_ai": True}

    # ─────────────────────────────────────────────────────────────────────────
    # KARAR LOGLAMA
    # ─────────────────────────────────────────────────────────────────────────

    def _log_decision(self, symbol: str, direction: str, result: dict) -> None:
        """Her MiMo kararını JSONL dosyasına kaydeder (denetim izi)."""
        try:
            Path("logs").mkdir(exist_ok=True)
            entry = {
                "ts":         pd.Timestamp.now().isoformat(),
                "symbol":     symbol,
                "direction":  direction,
                "decision":   result.get("decision"),
                "approved":   result.get("approved"),
                "confidence": result.get("confidence"),
                "reason":     result.get("reason"),
                "model":      result.get("model"),
            }
            with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

