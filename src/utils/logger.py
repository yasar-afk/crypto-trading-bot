# ============================================================
# src/utils/logger.py — ANTIGRAVITI Trading Bot
# Amaç : Tüm sistem genelinde tek tip, merkezi loglama altyapısı.
#         Her modül bu dosyadan logger alır; direkt print() kullanmaz.
# Tarih: 2026-06-03
#
# KULLANIM (herhangi bir modülden):
#   from src.utils.logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("Mesaj")
#
# ÖZELLİKLER:
#   - Rotating file handler: log dosyası büyüdükçe otomatik rotate
#   - Renkli konsol çıktısı: seviyeye göre farklı renk
#   - Hem dosyaya hem ekrana aynı anda yazar
#   - Thread-safe: birden fazla modül aynı anda kullanabilir
# ============================================================

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


# ─── Sabitler ────────────────────────────────────────────────
DEFAULT_LOG_DIR = os.getenv("LOG_DIR", "logs")
DEFAULT_LOG_FILE = "antigraviti.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
DEFAULT_BACKUP_COUNT = 5
DEFAULT_LEVEL = logging.INFO
DEFAULT_LOG_LEVEL = DEFAULT_LEVEL   # Alias — setup_logger imzasında kullanılır

# Renk haritası (colorlog kullanılıyorsa)
LOG_COLORS = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}

# ─── Global tracker (aynı logger'ı iki kez kurma) ────────────
_configured_loggers: set[str] = set()


def setup_logger(
    name: str,
    log_dir: str = DEFAULT_LOG_DIR,
    log_file: str = DEFAULT_LOG_FILE,
    level: int = DEFAULT_LOG_LEVEL,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console_output: bool = True,
) -> logging.Logger:
    """Logger'ı ilk kez kurar ve yapılandırır.

    Args:
        name: Logger adı (genellikle __name__ kullanılır).
        log_dir: Log dosyalarının yazılacağı klasör.
        log_file: Log dosyasının adı.
        level: Loglama seviyesi (logging.DEBUG, INFO, vb.).
        max_bytes: Rotate öncesi maksimum dosya boyutu (byte).
        backup_count: Tutulacak eski log dosyası sayısı.
        console_output: True ise terminale de yazar.

    Returns:
        Yapılandırılmış logging.Logger nesnesi.
    """
    logger = logging.getLogger(name)

    # Aynı logger'ı iki kez yapılandırma
    if name in _configured_loggers:
        return logger

    logger.setLevel(level)

    # Log klasörünü oluştur (yoksa)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    # ── Dosya Handler (Rotating) ────────────────────────────
    file_formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | "
            "%(funcName)-20s | %(lineno)4d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # ── Konsol Handler ──────────────────────────────────────
    if console_output:
        if COLORLOG_AVAILABLE:
            console_formatter = colorlog.ColoredFormatter(
                fmt=(
                    "%(log_color)s%(asctime)s | %(levelname)-8s | "
                    "%(name)-25s | %(message)s%(reset)s"
                ),
                datefmt="%H:%M:%S",
                log_colors=LOG_COLORS,
            )
        else:
            console_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
                datefmt="%H:%M:%S",
            )

        # Windows'ta emoji/unicode karakterler CP1254 ile yazılamaz.
        # Konsol stream'ini UTF-8'e çevir; bilinmeyen karakterler '?' ile değiştirilir.
        try:
            if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
                import io as _io
                _safe_stdout = _io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
                )
            else:
                _safe_stdout = sys.stdout
        except Exception:
            _safe_stdout = sys.stdout

        console_handler = logging.StreamHandler(_safe_stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)


    # Üst logger'a iletmeyi engelle (duplicate log önleme)
    logger.propagate = False
    _configured_loggers.add(name)

    return logger


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_dir: str = DEFAULT_LOG_DIR,
) -> logging.Logger:
    """Modüllerin kullanacağı tek erişim noktası.

    Eğer logger daha önce kurulmamışsa ilk kez kurar.
    Kurulmuşsa mevcut logger'ı döner.

    Args:
        name: Logger adı (genellikle __name__).
        level: Opsiyonel log seviyesi string'i ("DEBUG", "INFO", vb.).
               None ise ortam değişkeni veya varsayılan kullanılır.
        log_dir: Log klasörü yolu.

    Returns:
        Kullanıma hazır logging.Logger nesnesi.
    """
    # Seviye belirleme: parametre > ENV > varsayılan
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, DEFAULT_LOG_LEVEL)

    return setup_logger(
        name=name,
        log_dir=log_dir,
        level=numeric_level,
    )


# ─── Sistem Geneli Ana Logger ─────────────────────────────────
# Bu logger'ı doğrudan import edebilirsin:
#   from src.utils.logger import system_logger
system_logger = get_logger("antigraviti.system")
