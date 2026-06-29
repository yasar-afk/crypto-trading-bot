# ============================================================
# src/utils/ai_analyzer.py — Trading Bot Trading Bot
#
# AMAÇ:
#   Kapalı pozisyonların (kâr veya zarar) durumunu ve giriş anındaki
#   indikatör koşullarını Gemini AI modeline göndererek otomatik,
#   profesyonel bir "Hata & Gelişim Analizi" (Post-Mortem) üretir.
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v1.0 | İlk Yapay Zeka entegrasyonu
# ============================================================

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Dict, Optional, Tuple, List

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_post_mortem_analysis(
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    take_profit: float,
    pnl_usdt: float,
    pnl_pct: float,
    reason: str,
    indicator_summary: Optional[Dict] = None,
) -> str:
    """Gemini API kullanarak bir işlemin hata/kazanç analizini üretir.

    API Key girilmediyse veya hata oluşursa, temel kurallara göre statik analiz döner.
    """
    settings = get_settings()
    
    use_openrouter = bool(settings.openrouter_api_key)
    api_key = settings.openrouter_api_key if use_openrouter else settings.gemini_api_key

    # 1. API Anahtarı Yoksa Statik Yedek Analiz Döner
    if not api_key:
        logger.debug("GEMINI_API_KEY veya OPENROUTER_API_KEY bulunamadı. Şablon analizi kullanılıyor.")
        return _generate_fallback_analysis(pnl_usdt, exit_price, stop_loss, take_profit)

    # 2. Yapay Zeka Promptunu Oluştur
    indicators_str = ""
    if indicator_summary:
        indicators_str = "\n".join([f"  - {k}: {v}" for k, v in indicator_summary.items()])
    else:
        indicators_str = "  - Belirtilmemiş"

    prompt = (
        "Sen kıdemli bir Kripto Para Algoritmik Ticaret Uzmanı ve Risk Yöneticisisin.\n"
        "Aşağıda tamamlanmış (kapatılmış) bir işlemin teknik detayları verilmiştir.\n"
        "Bu işlemin neden kârla veya zararla bittiğini teknik göstergeler ışığında analiz et.\n"
        "Gelecekteki benzer işlemler için çıkarılacak dersi ve neyi geliştirmemiz gerektiğini belirt.\n"
        "Cevabını maksimum 2-3 cümleyle, kısa, net ve profesyonel bir Türkçe analiz olarak yaz. "
        "Analizinde doğrudan sonuca odaklan, gereksiz giriş cümleleri kurma.\n\n"
        "İŞLEM DETAYLARI:\n"
        f"- Sembol: {symbol}\n"
        f"- Yön: {side.upper()}\n"
        f"- Giriş Fiyatı: ${entry_price:,.4f}\n"
        f"- Çıkış Fiyatı: ${exit_price:,.4f}\n"
        f"- Stop-Loss (SL): ${stop_loss:,.4f}\n"
        f"- Take-Profit (TP): ${take_profit:,.4f}\n"
        f"- Net Kâr/Zarar: ${pnl_usdt:+,.2f} USDT ({pnl_pct*100:+.2f}%)\n"
        f"- Alış Gerekçesi: {reason}\n"
        f"Giriş Anındaki Teknik İndikatör Durumları:\n{indicators_str}"
    )

    # 3. API İsteği Gönder (urllib standart kütüphanesi ile)
    headers = {"Content-Type": "application/json"}
    if use_openrouter:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        headers["HTTP-Referer"] = "https://github.com/antigravity-trading-bot"
        headers["X-Title"] = "Antigravity Trading Bot"
        
        payload = {
            "model": settings.openrouter_model or "google/gemini-2.5-flash:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.4
        }
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 2048,
                "temperature": 0.4
            }
        }

    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data_bytes,
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            
            # Response parsing
            if use_openrouter:
                choices = res_data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    ai_text = message.get("content", "").strip()
                    logger.info(f"[AI] [{symbol}] OpenRouter Yapay Zeka Analizi başarıyla oluşturuldu.")
                    return ai_text
            else:
                candidates = res_data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        ai_text = parts[0].get("text", "").strip()
                        logger.info(f"[AI] [{symbol}] Gemini Yapay Zeka Analizi başarıyla oluşturuldu.")
                        return ai_text
            
            raise ValueError("API yanıt formatı geçersiz.")

    except urllib.error.HTTPError as e:
        logger.error(f"AI API HTTP Hatası: {e.code} - {e.read().decode('utf-8', errors='ignore')}")
    except Exception as e:
        logger.error(f"AI API Bağlantı Hatası: {e}")

    # Hata durumunda yedeğe düş
    return _generate_fallback_analysis(pnl_usdt, exit_price, stop_loss, take_profit)


def _generate_fallback_analysis(
    pnl_usdt: float,
    exit_price: float,
    stop_loss: float,
    take_profit: float,
) -> str:
    """API anahtarı eksik veya hatalıysa dönecek olan statik analiz şablonu."""
    if pnl_usdt < 0:
        if exit_price <= stop_loss:
            return "ZARARLA KAPANDI: Fiyat stop-loss seviyesine ulaşarak pozisyonu kapattı. Sistem risk sınırları korundu."
        else:
            return f"ZARARLA KAPANDI: Manuel veya strateji çıkışı ile pozisyon kapatıldı. Net Kayıp: ${abs(pnl_usdt):.2f}"
    else:
        if exit_price >= take_profit:
            return "KÂRLA KAPANDI: Fiyat take-profit hedefine ulaştı. Başarılı kâr realizasyonu yapıldı."
        else:
            return f"KÂRLA KAPANDI: Manuel veya strateji sinyaliyle pozisyon kârda kapatıldı. Net Kazanım: ${pnl_usdt:.2f}"


def verify_signal_with_ai(
    symbol: str,
    signal_type: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    indicator_summary: Dict,
    reasons: List[str]
) -> Tuple[bool, str]:
    """Sinyali yapay zekaya doğrulatır. (Approved, Reason) döndürür."""
    settings = get_settings()
    
    use_openrouter = bool(settings.openrouter_api_key)
    api_key = settings.openrouter_api_key if use_openrouter else settings.gemini_api_key

    # API Anahtarı Yoksa Varsayılan Olarak İşlemi Onayla (Teknik sinyal bloke edilmesin)
    if not api_key:
        logger.debug("AI API anahtarı bulunamadı. AI Sinyal Doğrulama devredışı bırakıldı (İşlem Onaylandı).")
        return True, "API Anahtarı girilmediği için işlem teknik sinyale göre onaylandı."

    # 1. Prompt Hazırla
    # Fibonacci seviyelerini ayıkla ve özel analiz hazırla
    fib_context = ""
    if indicator_summary and "fib_high" in indicator_summary and "fib_low" in indicator_summary:
        fib_high = indicator_summary.get("fib_high", 0.0)
        fib_low = indicator_summary.get("fib_low", 0.0)
        fib_236 = indicator_summary.get("fib_236", 0.0)
        fib_382 = indicator_summary.get("fib_382", 0.0)
        fib_500 = indicator_summary.get("fib_500", 0.0)
        fib_618 = indicator_summary.get("fib_618", 0.0)
        fib_786 = indicator_summary.get("fib_786", 0.0)
        
        # En yakın seviyeyi bul
        levels = {
            "0.236": fib_236,
            "0.382": fib_382,
            "0.500 (Golden Pocket - 0.5)": fib_500,
            "0.618 (Golden Pocket - 0.618)": fib_618,
            "0.786 (Golden Pocket - 0.786)": fib_786,
            "1.000 (Swing High)": fib_high,
            "0.000 (Swing Low)": fib_low
        }
        
        closest_level = None
        min_dist_pct = 999.0
        for name, val in levels.items():
            if val > 0:
                dist_pct = abs(entry_price - val) / val * 100
                if dist_pct < min_dist_pct:
                    min_dist_pct = dist_pct
                    closest_level = (name, val, dist_pct)
        
        fib_context = (
            f"- FIBONACCI YAPISI (Son 500 Mum):\n"
            f"  - En Yüksek (Swing High): ${fib_high:,.4f}\n"
            f"  - En Düşük (Swing Low): ${fib_low:,.4f}\n"
            f"  - Önemli Seviyeler: %23.6=${fib_236:,.4f}, %38.2=${fib_382:,.4f}, %50.0=${fib_500:,.4f}, %61.8=${fib_618:,.4f}, %78.6=${fib_786:,.4f}\n"
        )
        if closest_level:
            name, val, dist = closest_level
            fib_context += f"  - Fiyat Konumu: Giriş fiyatı (${entry_price:,.4f}), en yakın Fibonacci seviyesi olan %{name} (${val:,.4f}) seviyesine %{dist:.2f} uzaklıktadır.\n"

    # Formasyon bilgilerini ayıkla ve özel olarak formatla
    patterns_context = ""
    if indicator_summary and "active_patterns" in indicator_summary:
        active_pat = indicator_summary.get("active_patterns")
        if active_pat:
            patterns_context = f"- TESPİT EDİLEN MUM VE GRAFİK FORMASYONLARI:\n  - {active_pat}\n"

    filtered_indicators = {k: v for k, v in indicator_summary.items() if not k.startswith("fib") and k != "active_patterns"} if indicator_summary else {}
    indicators_str = "\n".join([f"  - {k}: {v}" for k, v in filtered_indicators.items()]) if filtered_indicators else "  - Belirtilmemiş"
    reasons_str = "\n".join([f"  - {r}" for r in reasons]) if reasons else "  - Belirtilmemiş"

    prompt = (
        "Sen kıdemli bir Kripto Para Algoritmik Ticaret Uzmanı ve Risk Yöneticisisin.\n"
        "Sistemimiz teknik analiz indikatörleri doğrultusunda yeni bir işlem sinyali üretti.\n"
        "Bu sinyalin detaylarını ve piyasa koşullarını inceleyerek, bu işleme girmenin mantıklı olup olmadığını değerlendir.\n"
        "İşleme girmeyi onaylıyor musun yoksa riskli görüp reddediyor musun?\n\n"
        "SİNYAL BİLGİLERİ:\n"
        f"- Sembol: {symbol}\n"
        f"- Sinyal Tipi: {signal_type}\n"
        f"- Giriş Fiyatı: ${entry_price:,.4f}\n"
        f"- Stop-Loss (SL): ${stop_loss:,.4f}\n"
        f"- Take-Profit (TP): ${take_profit:,.4f}\n"
        f"- Gerekçeler:\n{reasons_str}\n"
        f"- Teknik İndikatörler:\n{indicators_str}\n"
        f"{fib_context}"
        f"{patterns_context}\n"
        "Cevabını kesinlikle aşağıdaki JSON şablonuna tam olarak uyacak şekilde ver. Başka hiçbir şey yazma (açıklama ekleme):\n"
        "{\n"
        '  "approved": true veya false,\n'
        '  "reason": "Kararının gerekçesini açıklayan 1-2 Türkçe cümle"\n'
        "}"
    )

    # 2. API İsteği Gönder (urllib standart kütüphanesi ile)
    headers = {"Content-Type": "application/json"}
    if use_openrouter:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        headers["HTTP-Referer"] = "https://github.com/antigravity-trading-bot"
        headers["X-Title"] = "Antigravity Trading Bot"
        
        payload = {
            "model": settings.openrouter_model or "google/gemma-4-31b-it:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2
        }
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "temperature": 0.2
            }
        }

    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data_bytes,
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            
            # Response parsing
            ai_text = ""
            if use_openrouter:
                choices = res_data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    ai_text = message.get("content", "").strip()
            else:
                candidates = res_data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        ai_text = parts[0].get("text", "").strip()
            
            if not ai_text:
                raise ValueError("API boş yanıt döndü.")

            # JSON temizle ve yükle
            cleaned_text = ai_text.strip()
            if cleaned_text.startswith("```"):
                lines = cleaned_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned_text = "\n".join(lines).strip()

            parsed = json.loads(cleaned_text)
            approved = bool(parsed.get("approved", True))
            reason = str(parsed.get("reason", "Onaylandı."))
            return approved, reason

    except Exception as e:
        logger.error(f"AI sinyal doğrulama hatası: {e}")
        # Hata durumunda işlemi engellemiyoruz, teknik kararla devam ediyoruz.
        return True, f"AI doğrulama hatası nedeniyle teknik kararla devam ediliyor: {e}"
