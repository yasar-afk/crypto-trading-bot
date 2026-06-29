# ============================================================
# src/config/settings.py — ANTIGRAVITI Trading Bot
# Amaç : config.yaml ve .env dosyalarını birleştirerek tip-güvenli,
#         merkezi bir ayar nesnesi üretir. Tüm modüller ayarlarını
#         buradan alır; dosyaları direkt okumazlar.
# Tarih: 2026-06-03
#
# KULLANIM:
#   from src.config.settings import get_settings
#   cfg = get_settings()
#   print(cfg.exchange.symbol)
#
# MİMARİ NOT:
#   Pydantic BaseSettings kullandık çünkü:
#   1. .env dosyasını otomatik yükler
#   2. Tip doğrulaması yapar (yanlış değer → hemen hata)
#   3. Ortam değişkenleri config.yaml'ı override edebilir
#      (prod/dev farkı için kullanışlı)
# ============================================================

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# ─── Alt Konfigürasyon Modelleri ─────────────────────────────

class ExchangeConfig(BaseModel):
    """Borsa bağlantı parametreleri."""

    name: str = "binance"
    symbol: str = "BTC/USDT"
    sandbox: bool = True            # Testnet modunu etkinleştir
    default_type: str = "future"    # spot veya future
    rate_limit: bool = True
    recv_window: int = 5000
    request_timeout: int = 30

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Sembol formatını doğrula: 'BASE/QUOTE' formatında olmalı."""
        if "/" not in v:
            raise ValueError(
                f"Geçersiz sembol formatı: '{v}'. "
                "Beklenen format: 'BTC/USDT', 'ETH/USDT', vb."
            )
        return v.upper()


class PollingConfig(BaseModel):
    """REST polling aralıkları (saniye cinsinden)."""

    intervals: Dict[str, int] = Field(
        default={"4h": 14400, "1d": 86400},
        description="Her timeframe için polling aralığı (saniye)",
    )


class ValidationConfig(BaseModel):
    """Veri doğrulama eşikleri."""

    max_missing_candles: int = 5
    min_volume_threshold: float = 0.0
    price_spike_factor: float = 0.10


class DataConfig(BaseModel):
    """Veri toplama parametreleri."""

    timeframes: List[str] = ["4h", "1d"]
    limit: int = Field(default=500, ge=1, le=1000)
    primary_timeframe: str = "4h"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    polling: Dict[str, int] = {"4h": 14400, "1d": 86400}
    validation: ValidationConfig = ValidationConfig()

    @field_validator("primary_timeframe")
    @classmethod
    def primary_must_be_in_timeframes(cls, v: str, info) -> str:
        """Ana timeframe, timeframes listesinde olmalı."""
        # Pydantic v2'de diğer alanlara erişim için info kullanılır
        timeframes = info.data.get("timeframes", [])
        if timeframes and v not in timeframes:
            raise ValueError(
                f"primary_timeframe '{v}', timeframes listesinde yok: {timeframes}"
            )
        return v


class RiskConfig(BaseModel):
    """Risk yönetimi parametreleri (Faz 2'de aktif kullanılacak)."""

    max_position_pct: float = Field(default=0.02, ge=0.001, le=0.10)
    max_daily_drawdown_pct: float = Field(default=0.05, ge=0.01, le=0.20)
    min_risk_reward_ratio: float = Field(default=2.0, ge=1.0)
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5)
    normal_risk_pct: float = Field(default=0.10, ge=0.01, le=0.50)
    high_risk_pct: float = Field(default=0.20, ge=0.01, le=0.50)


class LoggingConfig(BaseModel):
    """Loglama yapılandırması."""

    level: str = "INFO"
    log_dir: str = "logs"
    max_bytes: int = 10_485_760   # 10 MB
    backup_count: int = 5
    console_output: bool = True


class PaperTradingConfig(BaseModel):
    """Paper trading / simülasyon ayarları."""

    enabled: bool = True
    initial_balance_usdt: float = Field(default=10_000.0, ge=100.0)
    commission_rate: float = Field(default=0.001, ge=0.0, le=0.01)


class StrategyConfig(BaseModel):
    """Strateji versiyon ayarları."""
    version: str = "v2"
    entry_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    exit_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    max_concurrent_positions: int = Field(default=10, ge=1)
    min_volume_ratio: float = Field(default=0.8, ge=0.0)
    sweep_window: int = Field(default=30, ge=5)
    max_hold_sweep: int = Field(default=5, ge=1, le=20)
    require_trend: bool = Field(default=False)
    displacement_atr_mult: float = Field(default=1.5, ge=0.5)
    use_premium_discount: bool = Field(default=True)
    use_session_filter: bool = Field(default=True)
    use_ote_filter: bool = Field(default=False)
    use_volume_filter: bool = Field(default=False)
    use_funding_oi_filter: bool = Field(default=False)
    trend_ema: int = Field(default=200, ge=10)


class TechnicalConfig(BaseModel):
    """Teknik indikatör parametreleri."""
    adx_period: int = Field(default=14, ge=2)
    adx_threshold: float = Field(default=20.0, ge=0.0)



class ExecutionConfig(BaseModel):
    """Emir yürütme/yönlendirme ayarları."""
    order_type: str = "market"
    limit_timeout_minutes: int = Field(default=15, ge=1)
    leverage: int = Field(default=5, ge=1, le=125)
    margin_mode: str = "ISOLATED"


# ─── Ana Ayar Sınıfı ─────────────────────────────────────────

class Settings(BaseSettings):
    """ANTIGRAVITI bot için tam konfigürasyon.

    Öncelik sırası (yüksekten düşüğe):
    1. Ortam değişkenleri (ENV)
    2. .env dosyası
    3. config.yaml
    4. Pydantic varsayılanları
    """

    # API anahtarları — yalnızca .env'den gelir
    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="google/gemma-4-31b-it:free", alias="OPENROUTER_MODEL")
    use_ai_signal_verification: bool = Field(default=True, alias="USE_AI_SIGNAL_VERIFICATION")
    # Xiaomi MiMo v2.5 Vision AI
    mimo_api_key: str = Field(default="", alias="MIMO_API_KEY")
    mimo_model: str = Field(default="mimo-v2.5", alias="MIMO_MODEL")
    mimo_base_url: str = Field(default="https://api.xiaomimimo.com/v1", alias="MIMO_BASE_URL")


    # Trading modu: "paper" | "live"
    trading_mode: str = Field(default="paper", alias="TRADING_MODE")

    # Telegram (Faz 3)
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")

    # Alt konfigürasyonlar (config.yaml'dan doldurulur)
    strategy: StrategyConfig = StrategyConfig()
    exchange: ExchangeConfig = ExchangeConfig()
    data: DataConfig = DataConfig()
    risk: RiskConfig = RiskConfig()
    logging: LoggingConfig = LoggingConfig()
    paper_trading: PaperTradingConfig = PaperTradingConfig()
    execution: ExecutionConfig = ExecutionConfig()
    technical: TechnicalConfig = TechnicalConfig()


    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @field_validator("trading_mode")
    @classmethod
    def validate_trading_mode(cls, v: str) -> str:
        """Trading modu yalnızca 'paper' veya 'live' olabilir."""
        allowed = {"paper", "live"}
        if v.lower() not in allowed:
            raise ValueError(f"trading_mode '{v}' geçersiz. Seçenekler: {allowed}")
        return v.lower()

    @property
    def is_paper_trade(self) -> bool:
        """Sistemin paper trade modunda çalışıp çalışmadığını döner."""
        return self.trading_mode == "paper" or self.paper_trading.enabled

    @property
    def has_api_credentials(self) -> bool:
        """API key'lerinin yapılandırılmış olup olmadığını kontrol eder."""
        return bool(self.binance_api_key and self.binance_api_secret)


# ─── Config Yükleyici ────────────────────────────────────────

def _load_yaml_config(config_path: str = "config.yaml") -> dict:
    """YAML konfigürasyon dosyasını yükler.

    Args:
        config_path: config.yaml dosyasının yolu.

    Returns:
        YAML içeriğini dict olarak döner. Dosya yoksa boş dict.
    """
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_settings_from_yaml(yaml_data: dict) -> Settings:
    """YAML verilerinden Settings nesnesi oluşturur.

    Args:
        yaml_data: _load_yaml_config() çıktısı.

    Returns:
        Tip-doğrulanmış Settings nesnesi.
    """
    kwargs: dict = {}

    if "strategy" in yaml_data:
        kwargs["strategy"] = StrategyConfig(**yaml_data["strategy"])

    if "exchange" in yaml_data:
        kwargs["exchange"] = ExchangeConfig(**yaml_data["exchange"])

    if "data" in yaml_data:
        d = yaml_data["data"].copy()
        if "validation" in d:
            d["validation"] = ValidationConfig(**d["validation"])
        kwargs["data"] = DataConfig(**d)

    if "risk" in yaml_data:
        kwargs["risk"] = RiskConfig(**yaml_data["risk"])

    if "logging" in yaml_data:
        kwargs["logging"] = LoggingConfig(**yaml_data["logging"])

    if "paper_trading" in yaml_data:
        kwargs["paper_trading"] = PaperTradingConfig(**yaml_data["paper_trading"])

    if "execution" in yaml_data:
        kwargs["execution"] = ExecutionConfig(**yaml_data["execution"])

    if "technical" in yaml_data:
        kwargs["technical"] = TechnicalConfig(**yaml_data["technical"])

    if "use_ai_signal_verification" in yaml_data:
        kwargs["use_ai_signal_verification"] = yaml_data["use_ai_signal_verification"]
    if "trading_mode" in yaml_data:
        kwargs["trading_mode"] = yaml_data["trading_mode"]

    return Settings(**kwargs)



@lru_cache(maxsize=4)
def get_settings(config_path: Optional[str] = None) -> Settings:
    """Singleton settings nesnesi döner.

    İlk çağrıda YAML + .env birleştirerek Settings oluşturur.
    Sonraki çağrılar önbellekten aynı nesneyi döner.

    Args:
        config_path: config.yaml dosyasının yolu. None ise CONFIG_FILE env veya varsayılan kullanılır.

    Returns:
        Tam yapılandırılmış Settings nesnesi.

    Example:
        >>> cfg = get_settings()
        >>> print(cfg.exchange.symbol)
        'BTC/USDT'
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_FILE", "config.yaml")
    yaml_data = _load_yaml_config(config_path)
    settings = _build_settings_from_yaml(yaml_data)
    return settings
