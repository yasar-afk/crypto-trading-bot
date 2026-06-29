# ============================================================
# src/utils/telegram_listener.py — Telegram Command Listener
# Komutlar:
#   /durum veya "durum" → Açık pozisyonlar + K/Z
# ============================================================

import json
import urllib.request
import urllib.parse
import threading
import time
import sys
from typing import Any
from src.utils.logger import get_logger
from src.utils.telegram_notifier import send_telegram_notification

logger = get_logger(__name__)


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


class TelegramListener(threading.Thread):
    def __init__(self, token: str, chat_id: str, execution_engine: Any) -> None:
        super().__init__(daemon=True)
        self.token = token
        self.chat_id = str(chat_id)
        self.engine = execution_engine
        self.last_update_id = 0
        self.running = True

    def run(self) -> None:
        if "pytest" in sys.modules:
            return

        logger.info("Telegram listener baslatildi.")
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates?offset=-1&limit=1"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                if res.get("ok") and res.get("result"):
                    self.last_update_id = res["result"][0]["update_id"]
        except Exception:
            pass

        while self.running:
            try:
                url = f"https://api.telegram.org/bot{self.token}/getUpdates?offset={self.last_update_id + 1}&timeout=10"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res = json.loads(resp.read().decode('utf-8'))
                    if res.get("ok") and res.get("result"):
                        for update in res["result"]:
                            self.last_update_id = update["update_id"]
                            
                            msg = update.get("message") or update.get("edited_message")
                            if not msg:
                                continue
                            
                            text = msg.get("text", "").strip().lower()
                            chat_id_from_msg = str(msg.get("chat", {}).get("id"))
                            
                            if chat_id_from_msg != self.chat_id:
                                continue

                            # /durum komutu
                            if any(cmd in text for cmd in ["durum", "status", "portfoy", "pozisyon", "acik"]):
                                self._send_durum()
                                
            except Exception as e:
                if "409" in str(e):
                    # Birden fazla listener — bekle
                    time.sleep(30)
                else:
                    logger.error(f"Telegram listener hatasi: {e}")
                    time.sleep(10)
            
            time.sleep(2)

    def _send_durum(self) -> None:
        """Acik pozisyonlari ve K/Z'yi goster — hem V5 hem V7 key formatini destekler."""
        try:
            # Eger engine'in kendi get_durum_text metodu varsa kullan (V5)
            if hasattr(self.engine, 'get_durum_text'):
                send_telegram_notification(self.engine.get_durum_text())
                return

            # V7 formati
            balance  = getattr(self.engine, 'balance', 0)
            positions = getattr(self.engine, 'positions', {})
            initial  = getattr(self.engine, 'initial_capital', 10000)
            symbol_stats = getattr(self.engine, '_symbol_stats', {})

            total_pnl = sum(s.get('total_pnl', 0) for s in symbol_stats.values())
            getiri = (balance / initial - 1) * 100 if initial else 0

            lines = [
                f"🤖 [V7] DURUM",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"💰 Bakiye: {format_price(balance)}",
                f"📊 Getiri: %{getiri:+.1f}",
                f"📈 Toplam K/Z: ${total_pnl:+,.2f}",
                f"🔓 Açık Pozisyon: {len(positions)}/{getattr(self.engine, 'max_positions', '?')}",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ]

            if not positions:
                lines.append("  Açık pozisyon yok")
            else:
                for sym, pos in positions.items():
                    # V7 key formatı
                    direction = pos.get('direction', pos.get('t', '?'))
                    entry     = pos.get('entry_price', pos.get('e', 0))
                    sl        = pos.get('stop_loss',   pos.get('sl', 0))
                    tp        = pos.get('take_profit', pos.get('tp', 0))
                    notional  = pos.get('notional',    pos.get('n', 0))
                    sz        = pos.get('size',        pos.get('sz', 0))

                    try:
                        ticker  = self.engine.fetcher.exchange.fetch_ticker(sym)
                        current = float(ticker['last'])
                    except Exception:
                        current = entry

                    if direction == 'BUY':
                        pnl_pct = (current - entry) / entry * 100 if entry else 0
                    else:
                        pnl_pct = (entry - current) / entry * 100 if entry else 0
                    pnl_usd = notional * pnl_pct / 100 * 5

                    yon_emoji = "🟢" if direction == "BUY" else "🔴"
                    kz_emoji  = "📈" if pnl_usd >= 0 else "📉"

                    lines.append(f"  {yon_emoji} {sym} | {direction}")
                    lines.append(f"     Giriş: {format_price(entry)} → Şimdi: {format_price(current)}")
                    lines.append(f"     {kz_emoji} K/Z: ${pnl_usd:+.2f} (%{pnl_pct:+.1f})")
                    lines.append(f"     SL: {format_price(sl)} | TP: {format_price(tp)}")
                    lines.append(f"     Büyüklük: {sz:.4g} adet | Değer: {format_price(notional)}")
                    lines.append("")

            send_telegram_notification("\n".join(lines))

        except Exception as e:
            logger.error(f"Durum raporu hatasi: {e}")


def start_telegram_listener(execution_engine: Any) -> None:
    """Telegram command listener'ini baslatir."""
    cfg = execution_engine.settings
    token = cfg.telegram_bot_token
    chat_id = cfg.telegram_chat_id

    if not token or not chat_id:
        return

    if "your_telegram_bot" in token or "your_telegram_chat" in chat_id:
        return

    listener = TelegramListener(token, chat_id, execution_engine)
    listener.start()
