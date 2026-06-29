# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import ccxt

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows console output encoding fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = PROJECT_ROOT / "data" / "historical" / "1h"
DATA_DIR.mkdir(parents=True, exist_ok=True)

exchange = ccxt.binance({
    "enableRateLimit": True,
    "timeout": 10000,
    "options": {"defaultType": "future"},
})

def get_top_100_symbols():
    """Fetches Binance Futures tickers and returns the top 100 USDT symbols by 24h volume."""
    print("Binance Futures pazarları yükleniyor...")
    exchange.load_markets()
    tickers = exchange.fetch_tickers()
    
    usdt_tickers = []
    for sym, ticker in tickers.items():
        # Only USDT markets, must be active
        is_usdt = (sym.endswith("/USDT") or sym.endswith("/USDT:USDT"))
        if is_usdt and ticker.get("active", True) != False:
            base_vol = ticker.get("baseVolume") or 0.0
            close_price = ticker.get("close") or 0.0
            vol = ticker.get("quoteVolume") or (base_vol * close_price)
            if vol and vol > 0:
                usdt_tickers.append((sym, vol))
                
    # Sort by 24h quote volume descending
    usdt_tickers.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [x[0] for x in usdt_tickers[:100]]
    return top_symbols

def clean_filename(symbol):
    """Replaces slashes and colons with underscores to make a clean filename."""
    return symbol.replace("/", "_").replace(":", "_")

def download_or_update_symbol(symbol, limit=1000, force=False):
    """Downloads last 1000 hourly candles or updates them incrementally."""
    file_name = clean_filename(symbol) + "_1h.csv"
    file_path = DATA_DIR / file_name
    
    df_existing = None
    since_ms = None
    
    if file_path.exists() and not force:
        try:
            df_existing = pd.read_csv(file_path)
            if not df_existing.empty and "timestamp" in df_existing.columns:
                last_ts = int(df_existing["timestamp"].iloc[-1])
                # Fetch starting from the next hour
                since_ms = last_ts + (60 * 60 * 1000)
                print(f"[{symbol}] Dosya mevcut. Son tarih: {df_existing['datetime'].iloc[-1]} → Güncelleniyor...")
        except Exception as e:
            print(f"  [{symbol}] Mevcut dosya okuma hatası: {e}. Yeniden indiriliyor...")
            df_existing = None

    # If file doesn't exist or we force download
    if since_ms is None:
        print(f"[{symbol}] Son {limit} mum sıfırdan indiriliyor...")
        # To get 1000 candles, we can pass since as None and limit=limit
        since_ms = None
        
    try:
        # Fetch OHLCV
        # For a clean 1000 limit, Binance allows up to 1000 per request.
        # If since_ms is provided, we fetch from since_ms up to now.
        candles = exchange.fetch_ohlcv(symbol, "1h", since=since_ms, limit=1000)
        if not candles:
            print(f"  [{symbol}] Yeni veri bulunamadı.")
            return True
            
        df_new = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_new["datetime"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
        
        if df_existing is not None:
            # Append new data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Remove duplicates based on timestamp
            df_combined = df_combined.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        else:
            df_combined = df_new
            
        # Keep only the last 1500 candles to avoid files growing infinitely, 
        # but making sure we have at least 1000 candles.
        if len(df_combined) > 1500:
            df_combined = df_combined.tail(1500).reset_index(drop=True)
            
        df_combined.to_csv(file_path, index=False)
        print(f"  [{symbol}] Başarılı! Toplam mum: {len(df_combined)} | Son mum: {df_combined['datetime'].iloc[-1]}")
        return True
    except Exception as e:
        print(f"  [{symbol}] HATA: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Binance Futures 1h Tarihsel Veri İndirici")
    parser.add_argument("--symbols", type=str, default=None, help="Virgülle ayrılmış özel sembol listesi (Örn: BTC/USDT,ETH/USDT)")
    parser.add_argument("--limit", type=int, default=1000, help="Sıfırdan çekilecek mum sayısı")
    parser.add_argument("--force", action="store_true", help="Mevcut dosyaları ez ve sıfırdan indir")
    args = parser.parse_args()
    
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = get_top_100_symbols()
        
    print(f"Tarama yapılacak toplam sembol: {len(symbols)}")
    print(f"Veri klasörü: {DATA_DIR}")
    print("-" * 50)
    
    success_count = 0
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] ", end="")
        ok = download_or_update_symbol(symbol, limit=args.limit, force=args.force)
        if ok:
            success_count += 1
        time.sleep(0.1) # Sleep to respect CCXT rate limit
        
    print("-" * 50)
    print(f"Tamamlandı! {success_count}/{len(symbols)} sembol başarıyla güncellendi.")

if __name__ == "__main__":
    main()
