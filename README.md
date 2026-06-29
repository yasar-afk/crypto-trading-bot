<div align="center">

# 🤖 ANTIGRAVITI

### AI Destekli Kripto Trading Botu

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Binance](https://img.shields.io/badge/Binance-F0B90B?style=flat&logo=binance&logoColor=black)
![OpenAI](https://img.shields.io/badge/MiMo_v2.5-412991?style=flat&logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Binance Futures üzerinde çalışan, yapay zeka ile grafik tabanlı sinyal doğrulama yapan
otomatik kripto para trading sistemi.

</div>

---

## 📋 Özellikler

| Özellik | Açıklama |
|---------|----------|
| 🧠 **AI Doğrulama** | MiMo v2.5 ile grafik analizi ve sinyal onayı |
| 📊 **Price Action** | SMC Liquidity Sweep stratejisi |
| 📈 **Dinamik RR** | ADX bazlı Risk/Ödül oranı (3.0-4.0) |
| 🛡️ **Risk Yönetimi** | Drawdown, trailing stop, ardisik kayip korumasi |
| 🔄 **Adaptif Ogrenme** | Gunluk otomatik parametre optimizasyonu |
| 📱 **Telegram** | Bildirimler ve uzaktan komut |
| 🕐 **Coklu Zaman Dilimi** | 15m + 1h + 4h konfirmasyon |

---

## 🏗️ Mimari

```
ANTIGRAVITI
├── live_v7.py              ← Ana bot
├── config_v7.yaml          ← Konfigurasyon
├── .env                    ← API anahtarlari (gizli)
│
├── src/
│   ├── strategy/
│   │   ├── v7_pa_strategy.py      ← Fiyat eylem stratejisi
│   │   ├── ai_chart_verifier.py   ← MiMo grafik dogrulama
│   │   └── adaptive_learner.py    ← Adaptif ogrenme
│   ├── risk/
│   │   └── engine.py              ← Risk yonetimi
│   └── utils/
│       └── telegram_notifier.py   ← Telegram bildirimleri
│
└── data/                   ← Fiyat verileri
```

---

## 🚀 Kurulum

```bash
# 1. Depoyu klonla
git clone https://github.com/yasar-afk/antigraviti-trading-bot.git
cd antigraviti-trading-bot

# 2. Sanal ortam olustur
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Bagimliliklari yukle
pip install -r requirements.txt

# 4. .env dosyasini olustur
cp .env.example .env
# .env dosyasini duzenle

# 5. Botu calistir
python live_v7.py --paper  # Paper trading
python live_v7.py --live   # Canli borsa
```

---

## ⚙️ Konfigurasyon

```yaml
# config_v7.yaml
strategy:
  sweep_window: 100        # Swing high/low tarama
  trend_ema: 180           # Trend EMA
  min_sl_pct: 0.02         # Min SL %2
  max_sl_pct: 0.08         # Max SL %8

risk:
  max_position_pct: 0.02   # Pozisyon basina risk %2
  max_daily_drawdown_pct: 0.05  # Gunluk max drawdown %5
  min_risk_reward_ratio: 3.0

execution:
  leverage: 5              # Kaldirac
  margin_mode: ISOLATED
```

---

## 📊 Strateji

### Price Action — SMC Liquidity Sweep

1. **Sweep Tespiti:** 100 mum penceresinde swing high/low kirilimi
2. **Giris Onayi:** Son 3 mumda kapanis onayi + EMA trend filtresi
3. **Dinamik RR:** ADX >25 → RR 4.0 | ADX 15-25 → RR 3.0
4. **AI Dogrulama:** MiMo v2.5 ile grafik analizi
5. **Risk Kontrolu:** Drawdown, korelasyon, sektor limiti

---

## 🛡️ Risk Yonetimi

| Kural | Deger |
|-------|-------|
| Pozisyon basina risk | %2 |
| Gunluk max drawdown | %5 |
| Kaldirac | 5x izole |
| Cooldown (SL sonrasi) | 24 saat |
| Ardisik kayip (2. kayip) | Boyut %50 |
| Ardisik kayip (3. kayip) | Boyut %25 |
| Blacklist | 2+ kayipta aninda |

---

## 📁 Proje Yapisi

| Dosya | Aciklama |
|-------|----------|
| `live_v7.py` | Ana bot döngüsü |
| `config_v7.yaml` | Strateji ve borsa ayarlari |
| `src/strategy/v7_pa_strategy.py` | Price Action stratejisi |
| `src/strategy/ai_chart_verifier.py` | MiMo grafik dogrulama |
| `src/strategy/adaptive_learner.py` | Adaptif ogrenme |
| `src/risk/engine.py` | Risk motoru |
| `src/utils/telegram_notifier.py` | Telegram bildirimleri |

---

## 🔒 Guvenlik

- `.env` dosyasi `.gitignore`'da — API anahtarlari paylasilmiyor
- Paper trading varsayilan mod
- Izole marjin ile calisir
- Gunluk drawdown limiti aktif

---

## 📜 Lisans

MIT License

---

<div align="center">

**ANTIGRAVITI** — Yapay zeka ile gelecege yatirim

</div>
