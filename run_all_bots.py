# -*- coding: utf-8 -*-
"""run_all_bots.py — V5 PA + V7 PA tek pencerede, tek Telegram listener"""
import sys
import os
import threading
import time
import signal
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

running = True
_bots = {}  # {"v5": bot_v5, "v7": bot_v7} — threadler arasi paylasim

def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)


def run_v5():
    global running
    try:
        from live_v5 import LiveV5Bot
        bot = LiveV5Bot(initial_capital=10000.0, max_positions=5, position_pct=0.02,
                        is_live=True, top_n=100)
        _bots['v5'] = bot
        while running:
            try:
                bot.run_cycle()
            except Exception as e:
                print(f"[V5] Hata: {e}")
            time.sleep(900)
    except Exception as e:
        print(f"[V5] Kritik: {e}")


def run_v7():
    global running
    try:
        from live_v7 import LiveV7Bot
        bot = LiveV7Bot(initial_capital=10000.0, max_positions=10, position_pct=0.02,
                        is_live=True, top_n=200)
        _bots['v7'] = bot
        while running:
            try:
                bot.run_cycle()
            except Exception as e:
                print(f"[V7] Hata: {e}")
            time.sleep(900)
    except Exception as e:
        print(f"[V7] Kritik: {e}")


def format_price(price):
    if price is None or price == 0: return "-"
    if price >= 100:     return f"${price:,.2f}"
    elif price >= 1.0:   return f"${price:,.4f}"
    elif price >= 0.0001: return f"${price:,.6f}"
    else:                return f"${price:,.8f}"


def _build_bot_durum(label, bot):
    """Bir botun durum metnini olusturur."""
    balance  = getattr(bot, 'balance', 0)
    initial  = getattr(bot, 'initial_capital', 10000)
    positions = getattr(bot, 'positions', {})
    max_pos  = getattr(bot, 'max_positions', '?')
    getiri   = (balance / initial - 1) * 100 if initial else 0

    lines = [
        f"🤖 [{label}] DURUM",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"💰 Bakiye: {format_price(balance)}",
        f"📊 Getiri: %{getiri:+.1f}",
        f"🔓 Açık Pozisyon: {len(positions)}/{max_pos}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not positions:
        lines.append("  Açık pozisyon yok")
    else:
        for sym, pos in positions.items():
            # V5 key'leri: t, e, sl, tp, sz, n  |  V7: direction, entry_price, ...
            direction = pos.get('direction', pos.get('t', '?'))
            entry     = pos.get('entry_price', pos.get('e', 0))
            sl        = pos.get('stop_loss',   pos.get('sl', 0))
            tp        = pos.get('take_profit', pos.get('tp', 0))
            notional  = pos.get('notional',    pos.get('n', 0))
            sz        = pos.get('size',        pos.get('sz', 0))

            # Anlik fiyat al
            try:
                ticker  = bot.fetcher.exchange.fetch_ticker(sym)
                current = float(ticker['last'])
            except Exception:
                current = entry

            if direction == 'BUY':
                pnl_usd = (current - entry) * sz * 5
                pnl_pct = (current - entry) / entry * 100 if entry else 0
            else:
                pnl_usd = (entry - current) * sz * 5
                pnl_pct = (entry - current) / entry * 100 if entry else 0

            yon_emoji = "🟢" if direction == "BUY" else "🔴"
            kz_emoji  = "📈" if pnl_usd >= 0 else "📉"

            lines.append(f"  {yon_emoji} {sym} | {direction}")
            lines.append(f"     Giriş: {format_price(entry)} → Şimdi: {format_price(current)}")
            lines.append(f"     {kz_emoji} K/Z: ${pnl_usd:+.2f} (%{pnl_pct:+.1f})")
            lines.append(f"     SL: {format_price(sl)} | TP: {format_price(tp)}")
            lines.append(f"     Büyüklük: {sz:.4g} adet | Değer: {format_price(notional)}")
            lines.append("")

    # Toplam anlık K/Z
    total_unrealized = 0.0
    for sym, pos in positions.items():
        direction = pos.get('direction', pos.get('t', '?'))
        entry     = pos.get('entry_price', pos.get('e', 0))
        sz        = pos.get('size',        pos.get('sz', 0))
        try:
            ticker  = bot.fetcher.exchange.fetch_ticker(sym)
            current = float(ticker['last'])
        except Exception:
            current = entry
        if direction == 'BUY':
            total_unrealized += (current - entry) * sz * 5
        else:
            total_unrealized += (entry - current) * sz * 5

    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📊 Açık Pozisyon K/Z: {format_price(total_unrealized)}")
    lines.append(f"💼 Toplam: {format_price(balance + total_unrealized)}")

    return "\n".join(lines)


def start_combined_listener():
    """Her iki botu bilen tek Telegram listener."""
    import urllib.request
    from src.utils.telegram_notifier import send_telegram_notification
    from src.config.settings import get_settings

    try:
        cfg = get_settings()
        token   = cfg.telegram_bot_token
        chat_id = str(cfg.telegram_chat_id)
        if not token or not chat_id or "your_telegram" in token:
            return
    except Exception:
        return

    def _get_updates(last_id):
        url = f"https://api.telegram.org/bot{token}/getUpdates?offset={last_id+1}&timeout=10"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    # Baslangic offset
    last_update_id = 0
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates?offset=-1&limit=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            res = json.loads(r.read().decode())
            if res.get("ok") and res.get("result"):
                last_update_id = res["result"][0]["update_id"]
    except Exception:
        pass

    print("[LISTENER] Birlesik Telegram listener baslatildi.")

    while running:
        try:
            res = _get_updates(last_update_id)
            if res.get("ok") and res.get("result"):
                for update in res["result"]:
                    last_update_id = update["update_id"]
                    msg = update.get("message") or update.get("edited_message")
                    if not msg:
                        continue
                    if str(msg.get("chat", {}).get("id")) != chat_id:
                        continue
                    text = msg.get("text", "").strip().lower()

                    if any(cmd in text for cmd in ["durum", "status", "portfoy", "pozisyon", "acik"]):
                        # V5 ve V7 icin ayri mesajlar gonder
                        for label, key in [("V5", "v5"), ("V7", "v7")]:
                            bot = _bots.get(key)
                            if bot:
                                try:
                                    send_telegram_notification(_build_bot_durum(label, bot))
                                    time.sleep(0.5)
                                except Exception as ex:
                                    print(f"[LISTENER] {label} durum hatasi: {ex}")
                            else:
                                send_telegram_notification(f"🤖 [{label}] henüz başlatılmadı...")
        except Exception as e:
            if "409" not in str(e):
                print(f"[LISTENER] Hata: {e}")
            time.sleep(10)
        time.sleep(2)


def main():
    global running

    print("=" * 60)
    print("  ANTIGRAVITI -- V5 + V7 (TEK PENCERE)")
    print("=" * 60)
    print("  V5: Price Action, filtresiz, %2 risk, 100 coin")
    print("  V7: Price Action, filtreli,  %2 risk, 100 coin")
    print("  Durdurmak icin: Ctrl+C")
    print("=" * 60)

    try:
        from src.utils.telegram_notifier import send_telegram_notification
        send_telegram_notification(
            "🤖 V5 + V7 BAŞLATILDI\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "V5: PA Filtresiz, %2 risk, 100 coin\n"
            "V7: PA Filtreli, %2 risk, 100 coin\n"
            "/durum → anlık pozisyonlar"
        )
    except Exception:
        pass

    threads = [
        threading.Thread(target=run_v5, name="V5", daemon=True),
        threading.Thread(target=run_v7, name="V7", daemon=True),
    ]
    for t in threads:
        t.start()
        time.sleep(5)

    # Tek birlesik listener
    listener_thread = threading.Thread(target=start_combined_listener, name="Listener", daemon=True)
    listener_thread.start()

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
    print("\nDurduruldu.")


if __name__ == "__main__":
    main()
