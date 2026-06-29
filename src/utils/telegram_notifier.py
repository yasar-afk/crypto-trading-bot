# ============================================================
# src/utils/telegram_notifier.py — Telegram Notification Utility
# Sadece onemli olaylar bildirilir:
#   - Bot baslatma
#   - Yeni pozisyon acma
#   - Pozisyon kapama (k/z ile)
#   - /durum komutu
# ============================================================

import urllib.request
import urllib.parse
import json
from typing import Optional
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def send_telegram_notification(message: str) -> None:
    """Onemli olaylari Telegram'a bildirir."""
    import sys
    if "pytest" in sys.modules:
        return

    try:
        cfg = get_settings()
        
        import os
        if os.getenv("TELEGRAM_ENABLED", "true").lower() == "false":
            return

        token = cfg.telegram_bot_token
        chat_id = cfg.telegram_chat_id

        if not token or not chat_id:
            return

        if "your_telegram_bot" in token or "your_telegram_chat" in chat_id:
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status != 200:
                logger.error(f"Telegram basarisiz: {response.status}")
    except Exception as e:
        logger.error(f"Telegram hatasi: {e}")


def get_telegram_updates() -> list:
    """Telegram'dan gelen mesajlari okur."""
    try:
        cfg = get_settings()
        token = cfg.telegram_bot_token
        chat_id = cfg.telegram_chat_id

        if not token or not chat_id:
            return []

        url = f"https://api.telegram.org/bot{token}/getUpdates?offset=-1&limit=5"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data.get("ok"):
                return data.get("result", [])
    except Exception:
        pass
    return []
