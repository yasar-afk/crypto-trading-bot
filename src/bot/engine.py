# ============================================================
# src/bot/engine.py — Trading Bot Trading Bot
#
# AMAÇ:
#   Trading botunun tüm katmanlarını (DataCollector,
#   TechnicalEngine, SignalGenerator, SignalJournal, CandleScheduler,
#   RiskEngine, ExecutionEngine) bir araya getiren ve periyodik
#   döngüyü işleten ana orkestratör.
#
# DEĞİŞİKLİK GEÇMİŞİ:
#   2026-06-04 | v2.0 | Risk ve Execution modüllerinin tam entegrasyonu
# ============================================================

from __future__ import annotations

import time
from typing import Dict, List, Optional
import sys

from src.config.settings import Settings, get_settings
from src.data.collector import DataCollector
from src.technical.engine import TechnicalEngine
from src.signal.generator import SignalGenerator
from src.signal.journal import SignalJournal
from src.bot.scheduler import CandleScheduler
from src.risk.engine import RiskEngine
from src.execution.engine import ExecutionEngine
from src.execution.models import OrderStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BotEngine:
    """Tüm alt bileşenleri koordine eden ana ticaret motoru.

    Rolü:
      Borsaya bağlan → Mum zamanlarını izle → Veri Çek → Analiz Et → Sinyal Üret 
      → Risk Değerlendir → Emri İlet → Pozisyonları İzle
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        is_paper_trade: Optional[bool] = None,
    ) -> None:
        """BotEngine sınıfını başlatır.

        Args:
            settings: Sistem ayarları nesnesi. None ise varsayılan yüklenir.
            is_paper_trade: Kağıt işlem modu ayarı (belirtilmezse settings'den alınır).
        """
        self.settings = settings or get_settings()

        # Paper trade modu belirleme
        if is_paper_trade is not None:
            self.is_paper_trade = is_paper_trade
        else:
            self.is_paper_trade = self.settings.is_paper_trade

        # Bileşenleri başlat
        self.collector = DataCollector(
            settings=self.settings,
            is_paper_trade=self.is_paper_trade
        )

        self.technical_engine = TechnicalEngine(
            atr_multiplier=self.settings.risk.atr_stop_multiplier,
            rr_ratio=self.settings.risk.min_risk_reward_ratio,
            adx_period=self.settings.technical.adx_period,
            adx_threshold=self.settings.technical.adx_threshold,
        )

        self.signal_generator = SignalGenerator(
            min_rr_ratio=self.settings.risk.min_risk_reward_ratio,
            min_volume_ratio=self.settings.strategy.min_volume_ratio,
            adx_threshold=self.settings.technical.adx_threshold,
            is_paper_trade=self.is_paper_trade,
        )
        # Strateji versiyonunu generator'a enjekte et (lru_cache bypass için)
        self.signal_generator._strategy_version = self.settings.strategy.version

        self.journal = SignalJournal(
            journal_dir=self.settings.logging.log_dir
        )

        self.scheduler = CandleScheduler(
            timeframes=self.settings.data.timeframes
        )

        # Risk ve Execution Motorlarının Tanımlanması
        self.risk_engine = RiskEngine(
            settings=self.settings,
            initial_balance=self.settings.paper_trading.initial_balance_usdt
        )

        self.execution_engine = ExecutionEngine(
            settings=self.settings,
            risk_engine=self.risk_engine,
            exchange=None,  # Borsa bağlantısı connect() sonrası atanır
            state_dir=self.settings.logging.log_dir
        )

        self._keep_running = False

        logger.info(
            f"BotEngine başarıyla yüklendi | "
            f"İşlem Modu: {'PAPER TRADE' if self.is_paper_trade else 'CANLI'} | "
            f"Sembol: {self.settings.exchange.symbol} | "
            f"Timeframes: {self.settings.data.timeframes}"
        )

    def run(
        self,
        single_run: bool = False,
        symbols: Optional[List[str]] = None,
        top_50: bool = False,
        top_100: bool = False,
        top_n: Optional[int] = None,
    ) -> None:
        """Bot motorunu çalıştırır.

        Args:
            single_run: True ise döngüye girmeden bir kez tarar ve çıkar.
            symbols: Manuel taranacak sembol listesi (None ise varsayılan sembol kullanılır).
            top_50: True ise Binance'taki en yüksek 24h hacimli 50 USDT paritesini dinamik tarar.
            top_100: True ise Binance'taki en yüksek 24h hacimli 100 USDT paritesini dinamik tarar.
            top_n: Belirtilen sayıdaki en yüksek 24h hacimli USDT paritesini dinamik tarar.
        """
        logger.info("=" * 60)
        logger.info("[START] Trading Bot Algo Trading Bot Calismaya Basliyor")
        logger.info("=" * 60)

        # Borsaya baglan
        if not self.collector.connect():
            logger.error("[HATA] Borsaya baglanti kurulamadi. Motor baslatilmiyor.")
            return

        # Canlı borsa nesnesini ExecutionEngine'e bağla
        self.execution_engine.exchange = self.collector.exchange

        # Hedef sembol listesini belirle
        target_symbols: List[str] = []

        limit_n = None
        if top_n is not None:
            limit_n = top_n
        elif top_100:
            limit_n = 100
        elif top_50:
            limit_n = 50

        if limit_n is not None:
            try:
                logger.info(f"Binance en yüksek 24 saatlik hacme sahip {limit_n} coin çekiliyor...")
                tickers = self.collector.exchange.fetch_tickers()
                usdt_tickers = []
                for sym, ticker in tickers.items():
                    # Hem Spot (BASE/USDT) hem de Futures (BASE/USDT:USDT) sembollerini kabul et
                    is_usdt = sym.endswith("/USDT") or sym.endswith("/USDT:USDT")
                    if is_usdt and ticker.get("active", True) != False:
                        base_vol = ticker.get("baseVolume") or 0.0
                        close_price = ticker.get("close") or 0.0
                        vol = ticker.get("quoteVolume") or (base_vol * close_price)
                        if vol and vol > 0:
                            clean_sym = sym.split(":")[0]  # BASE/USDT formatına dönüştür
                            usdt_tickers.append((clean_sym, vol))
                # Hacme göre sırala
                usdt_tickers.sort(key=lambda x: x[1], reverse=True)
                
                # Tekrar eden sembolleri engelle
                seen_symbols = set()
                target_symbols = []
                for sym, vol in usdt_tickers:
                    if sym not in seen_symbols:
                        seen_symbols.add(sym)
                        target_symbols.append(sym)
                        if len(target_symbols) >= limit_n:
                            break
                            
                logger.info(f"🔥 Dinamik Top {limit_n} Sembol Listesi Alındı: {target_symbols[:10]}... (Toplam {len(target_symbols)} adet)")
            except Exception as e:
                logger.error(f"Top {limit_n} coin listesi alınamadı: {e}. Varsayılan sembole dönülüyor.")
                target_symbols = [self.settings.exchange.symbol]
        elif symbols:
            target_symbols = symbols
        else:
            target_symbols = [self.settings.exchange.symbol]

        # Sembolleri doğrula
        self.symbols = []
        for sym in target_symbols:
            if self.collector.validate_symbol(sym):
                self.symbols.append(sym)
            else:
                logger.warning(f"⚠️ Sembol geçersiz olduğu için listeden çıkarıldı: {sym}")

        if not self.symbols:
            logger.error("❌ Taranacak geçerli sembol kalmadı! Bot durduruluyor.")
            self.collector.disconnect()
            return

        logger.info(f"🎯 Tarama yapılacak Semboller (Toplam {len(self.symbols)}): {self.symbols[:10]}...")

        # Tek seferlik analiz modu (Single Run)
        if single_run:
            logger.info("⚡ Tek seferlik tarama modu etkin (Single Run).")
            self._execute_pass(self.symbols, self.settings.data.timeframes)
            self._print_portfolio_status()
            self.collector.disconnect()
            logger.info("⚡ Tek seferlik tarama tamamlandı. Bağlantı kesildi.")
            return

        # Sonsuz periyodik döngü
        self._keep_running = True
        logger.info("Periyodik tarama dongusu baslatildi. Durdurmak icin Ctrl+C'ye basin.")
        logger.info(self.scheduler.status_report())

        last_report_time = time.time()
        report_interval = 300  # 5 dakikada bir durum raporu yazdır

        # BAGIMIZ SL/TP IZLEME SAYACI
        # Mum kapanmasını beklemeksizin her N saniyede bir fiyat güncelle
        last_price_check_time = 0
        price_check_interval = 60  # Saniye (her 1 dakikada bir)

        try:
            while self._keep_running:
                now = time.time()

                # ── BAGIMIZ FIYAT IZLEME (SL/TP Kontrolü) ─────────────────
                # Her 60 saniyede bir, mum beklenmeksizin anlık fiyat çekilir
                # ve SL/TP kontrolleri yapılır. Açık pozisyon yoksa da watch.py için state kaydedilir (kalp atışı).
                if now - last_price_check_time >= price_check_interval:
                    if self.execution_engine.open_positions:
                        open_syms = list(self.execution_engine.open_positions.keys())
                        try:
                            live_prices = self.collector.get_current_prices(open_syms)
                            if live_prices:
                                closed = self.execution_engine.monitor_positions(live_prices)
                                if closed:
                                    logger.info(
                                        f"Fiyat izleme: {len(closed)} pozisyon kapatildi."
                                    )
                                else:
                                    # Sadece SL'ye yakin pozisyonlari uyar
                                    for sym, pos in self.execution_engine.open_positions.items():
                                        active_sl = pos.get_trailing_stop_level()
                                        cur = live_prices.get(sym, 0)
                                        if active_sl > 0 and cur > 0:
                                            sl_dist_pct = (cur - active_sl) / cur * 100
                                            if sl_dist_pct < 3.0:  # SL'ye %3'ten yakin
                                                logger.warning(
                                                    f"UYARI [{sym}] SL'ye cok yakin! "
                                                    f"Fiyat: ${cur:.4f} | SL: ${active_sl:.4f} | "
                                                    f"Mesafe: %{sl_dist_pct:.1f}"
                                                )
                        except Exception as e:
                            logger.error(f"Fiyat izleme hatasi: {e}")
                    else:
                        # Açık pozisyon yoksa watch.py'nin botun aktif olduğunu bilmesi için kalp atışı (state save) yap
                        try:
                            self.execution_engine.save_state()
                        except Exception as e:
                            logger.error(f"Kalp atisi state kaydetme hatasi: {e}")
                    last_price_check_time = now
                # ─────────────────────────────────────────────────────────────

                # Zamanı gelmiş zaman dilimlerini al
                due_tfs = self.scheduler.get_due_timeframes()

                if due_tfs:
                    logger.info(f"Yeni mum tetiklendi. Analiz ediliyor: {due_tfs}")
                    self._execute_pass(self.symbols, due_tfs)

                    # Portföy ve Zamanlayıcı raporlarını yazdır
                    self._print_portfolio_status()
                    logger.info(self.scheduler.status_report())
                
                # Periyodik durum güncellemesi
                if now - last_report_time >= report_interval:
                    self._print_portfolio_status()
                    logger.info(self.scheduler.status_report())
                    last_report_time = now

                # Kısa bir süre uyu ve tekrar kontrol et
                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("\nKullanici tarafindan durdurma istegi alindi (Ctrl+C).")
        finally:
            self._keep_running = False
            self.collector.disconnect()
            logger.info("Bot temiz bir sekilde kapatildi.")


    def _execute_pass(self, symbols: List[str], timeframes: List[str]) -> None:
        """Belirtilen zaman dilimleri için veri çekim, analiz ve sinyal süreçlerini işletir.

        Args:
            symbols: İşlem yapılacak sembollerin listesi.
            timeframes: İşlenecek zaman dilimleri.
        """
        # ── Açık Pozisyonlar İçin Toplu Anlık Fiyat Güncellemesi ──────────
        # OHLCV önbelleğinden bağımsız olarak, gerçek zamanlı ticker verisi çek.
        # Bu şekilde SL/TP kontrolleri her zaman anlık fiyatla yapılır.
        if self.execution_engine.open_positions:
            open_syms = list(self.execution_engine.open_positions.keys())
            logger.info(f"📡 {len(open_syms)} açık pozisyon için anlık fiyat güncelleniyor: {open_syms}")
            try:
                live_prices = self.collector.get_current_prices(open_syms)
                if live_prices:
                    self.execution_engine.monitor_positions(live_prices)
                    logger.info(f"✅ Anlık fiyat güncellemesi tamamlandı: {live_prices}")
                else:
                    logger.warning("⚠️ Batch ticker boş döndü. Tekil çekime geçiliyor...")
                    for sym in open_syms:
                        p = self.collector.get_current_price(sym)
                        if p is not None:
                            self.execution_engine.monitor_positions({sym: p})
            except Exception as e:
                logger.error(f"❌ Anlık fiyat güncellemesi hatası: {e}")
        # Açık pozisyonları da taranacak semboller listesine ekle (böylece strateji çıkış sinyalleri taranabilir)
        open_pos_symbols = list(self.execution_engine.open_positions.keys())
        combined_symbols = list(symbols)
        for sym in open_pos_symbols:
            if sym not in combined_symbols:
                combined_symbols.append(sym)

        # BTC Makro Trend Kontrolü (Death Cross check on 4h for V2 and V2.1)
        btc_trend_bearish = False
        strategy_ver = self.settings.strategy.version
        if strategy_ver in ("v2", "v2.1"):
            try:
                btc_res = self.collector.fetch("BTC/USDT", "4h")
                if btc_res.success and btc_res.data:
                    btc_df = self.collector.to_dataframe(btc_res.data)
                    btc_enriched = self.technical_engine.enrich_dataframe(btc_df)
                    if "ema50" in btc_enriched.columns and "ema200" in btc_enriched.columns:
                        last_btc = btc_enriched.iloc[-1]
                        btc_trend_bearish = float(last_btc["ema50"]) < float(last_btc["ema200"])
                        if btc_trend_bearish:
                            if strategy_ver == "v2.1":
                                logger.info("📡 [v2.1 MAKRÖ] BTC 4h Death Cross aktif → BUY bloke, SHORT pozisyonlar TEŞVİK edildi.")
                            else:
                                logger.info("📡 [BTC MAKRÖ] BTC 4h Death Cross aktif, altcoin alımları durdurulacak.")
            except Exception as e:
                logger.error(f"BTC makro trend kontrolü hatası: {e}")

        for tf in timeframes:
            for symbol in combined_symbols:
                try:
                    logger.info(f"[{symbol}@{tf}] Veri çekiliyor...")
                    # Veri çek (Önbellek kullanılabilir)
                    result = self.collector.fetch(
                        symbol=symbol,
                        timeframe=tf,
                        limit=self.settings.data.limit
                    )

                    if not result.success or not result.data:
                        logger.error(
                            f"[{symbol}@{tf}] Veri çekilemedi: {result.error_message}"
                        )
                        continue
                    # DataFrame'e dönüştür
                    df = self.collector.to_dataframe(result.data)

                    
                    # Anlık fiyat ve ticker bilgisi al (F5 kontrolü için)
                    ticker_info = None
                    live_price = None
                    try:
                        ticker_info = self.collector.exchange.fetch_ticker(symbol)
                        live_price = ticker_info.get("last") or ticker_info.get("close")
                        if live_price is not None:
                            live_price = float(live_price)
                    except Exception as e:
                        logger.warning(f"Ticker çekilemedi ({symbol}): {e}")
                    
                    current_price = live_price if live_price is not None else float(df["close"].iloc[-1])
 
                    # ── SL/TP Pozisyon Kontrolleri ───────────────────
                    # Tekil sembol için de anlık fiyatla kontrol et
                    self.execution_engine.monitor_positions({symbol: current_price})

                    # Teknik analiz motoruna gönder ve indikatörleri hesapla
                    ind_set = self.technical_engine.get_latest_indicators(
                        df=df,
                        symbol=symbol,
                        timeframe=tf
                    )

                    if not ind_set:
                        logger.warning(
                            f"[{symbol}@{tf}] İndikatör yorum seti oluşturulamadı."
                        )
                        continue

                    # Sinyal motorunda değerlendir
                    signal = self.signal_generator.evaluate(
                        ind_set,
                        btc_trend_bearish=btc_trend_bearish,
                        ticker_info=ticker_info
                    )
                    
                    # ── Pump & Dump Dedektörü (V2 ve V2.1 BUY için) ──
                    from src.signal.models import SignalType as ST
                    _ver = self.settings.strategy.version
                    if signal.signal_type == ST.BUY and _ver in ("v2", "v2.1"):
                        try:
                            logger.info(f"🔍 [PUMP DEDEKTÖRÜ] {symbol} için 5 dakikalık mumlar inceleniyor...")
                            res_5m = self.collector.fetch(symbol, "5m", limit=3, use_cache=False)
                            if res_5m.success and res_5m.data and len(res_5m.data.candles) >= 2:
                                candles = res_5m.data.candles
                                price_10m_ago = float(candles[0].open)
                                price_now = float(candles[-1].close)
                                price_change_pct = (price_now - price_10m_ago) / price_10m_ago * 100
                                vol_ratio = ind_set.volume.volume_ratio if ind_set.volume else 0.0
                                
                                if price_change_pct >= 5.0 and vol_ratio >= 4.0:
                                    logger.warning(
                                        f"⚠️ [PUMP BLOKE] {symbol} son 10 dakikada %{price_change_pct:.2f} yükseldi "
                                        f"ve hacim oranı {vol_ratio:.2f}x. Alış bloke edildi!"
                                    )
                                    signal = self.signal_generator._build_no_signal(
                                        symbol=symbol,
                                        timeframe=tf,
                                        price=current_price,
                                        score=signal.weighted_score,
                                        rejection=[f"PUMP_AND_DUMP_BLOCKED (%{price_change_pct:.1f} rise, {vol_ratio:.1f}x vol)"]
                                    )
                        except Exception as e:
                            logger.error(f"Pump dedektörü kontrolü hatası: {e}")

                    # ── Yapay Zeka Sinyal Doğrulama Filtresi (Pre-Trade AI Filter) ──
                    if signal.is_actionable and self.settings.use_ai_signal_verification:
                        from src.utils.ai_analyzer import verify_signal_with_ai
                        from dataclasses import replace
                        from src.signal.models import SignalType as ST
                        
                        logger.info(f"🔮 [{symbol}] {signal.signal_type.value} sinyali için AI doğrulaması başlatılıyor...")
                        approved, ai_reason = verify_signal_with_ai(
                            symbol=signal.symbol,
                            signal_type=signal.signal_type.value,
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            indicator_summary=signal.indicator_summary,
                            reasons=list(signal.reasons)
                        )
                        
                        if not approved:
                            logger.warning(
                                f"🚫 [AI REDDİ] [{symbol}] Yapay zeka işlemi riskli bulduğu için ENGELLEDİ. "
                                f"Gerekçe: {ai_reason}"
                            )
                            signal = replace(
                                signal,
                                signal_type=ST.NO_SIGNAL,
                                rejection_reasons=signal.rejection_reasons + (f"AI_REJECTED ({ai_reason})",)
                            )
                        else:
                            logger.info(f"✅ [AI ONAYI] [{symbol}] Yapay zeka işlemi onayladı: {ai_reason}")

                    # Sinyal defterine kaydet
                    self.journal.record(signal)

                    # Scheduler üzerinde bu timeframe'i güncellendi olarak işaretle
                    self.scheduler.mark_fetched(tf)

                    # ── Emir İletim ve Risk Yönetim Entegrasyonu ──────
                    if signal.is_actionable:
                        logger.info(f"⚡ [{symbol}] Aktif sinyal: {signal.signal_type.value} | Fiyat: ${current_price:,.4f} | Skor: {signal.weighted_score:.3f} → Risk değlendiriliyor...")

                        # ── MAX_CONCURRENT_POSITIONS kontrolü ──────────
                        max_pos = self.settings.strategy.max_concurrent_positions
                        current_open = len(self.execution_engine.open_positions)

                        # BUY sinyali için açık pozisyon limitini kontrol et
                        from src.signal.models import SignalType as ST
                        if signal.signal_type == ST.BUY and current_open >= max_pos:
                            logger.warning(
                                f"⛔ [{symbol}] Açık pozisyon limiti doldu! "
                                f"Mevcut: {current_open} / Limit: {max_pos}. "
                                f"Yeni alış engellendi."
                            )
                        else:
                            # 1. Risk değerlendirmesi al
                            assessment = self.risk_engine.assess(
                                signal=signal,
                                portfolio_balance=self.execution_engine.total_portfolio_value
                            )

                            if not assessment.is_approved:
                                logger.warning(
                                    f"🚫 [{symbol}] Risk motoru reddetti → POZISYON ACILMADI. "
                                    f"Neden: {getattr(assessment, 'rejection_reason', 'Bilinmiyor')}"
                                )
                            else:
                                # 2. Sinyali ve risk onayını ExecutionEngine'e ilet
                                order = self.execution_engine.execute_signal(
                                    signal=signal,
                                    assessment=assessment
                                )

                                if order:
                                    if order.status == OrderStatus.FILLED:
                                        logger.info(f"✅ POZISYON ACILDI: ID {order.order_id} | {symbol} | Yön: {order.side.upper()} | Miktar: {order.amount:.6f} | Fiyat: ${order.price:,.4f} | SL: ${order.stop_loss:,.4f} | TP: ${order.take_profit:,.4f}")
                                    elif order.status == OrderStatus.PENDING:
                                        logger.info(f"⏳ LiMIT EMRi YERLESTIRILDI: ID {order.order_id} | {symbol} | Limit Fiyat: ${order.price:,.4f} | SL: ${order.stop_loss:,.4f} | TP: ${order.take_profit:,.4f}")
                                    elif order.status == OrderStatus.FAILED:
                                        logger.warning(f"❌ Emir başarısız: {order.error_message}")


                except Exception as e:
                    logger.error(
                        f"❌ [{symbol}@{tf}] Beklenmeyen çevrim hatası: {e}",
                        exc_info=True
                    )

    def _print_portfolio_status(self) -> None:
        """Cüzdan bakiyesi ve açık pozisyon durumunu ekrana yazdırır."""
        initial_balance = 10000.0
        current_balance = self.execution_engine.usdt_balance
        net_profit = current_balance - initial_balance
        
        logger.info("\n" + "=" * 80)
        logger.info("💳 PORTFÖY DURUM RAPORU")
        logger.info("=" * 80)
        logger.info(f"  Boştaki Nakit (USDT): ${current_balance:,.2f} USDT  (Başlangıç: ${initial_balance:,.2f} | Net Kâr: +${net_profit:,.2f})")
        logger.info(f"  Açık Pozisyon Sayısı: {len(self.execution_engine.open_positions)} adet")
        
        total_pnl = 0.0
        total_cost = 0.0
        
        if self.execution_engine.open_positions:
            logger.info("-" * 80)
            logger.info(f"  {'Sembol':<10} | {'Yön':<5} | {'Giriş':<10} | {'Güncel':<10} | {'SL':<10} | {'TP':<10} | {'Anlık PnL ($ / %)'}")
            logger.info("-" * 80)
            for sym, pos in self.execution_engine.open_positions.items():
                pnl_usdt = pos.pnl_usdt
                pnl_pct = pos.pnl_pct * 100
                total_pnl += pnl_usdt
                total_cost += pos.cost_usdt
                
                logger.info(
                    f"  • {sym:<8} | "
                    f"{pos.side.value.upper():<5} | "
                    f"${pos.entry_price:<9,.4f} | "
                    f"${pos.current_price:<9,.4f} | "
                    f"${pos.stop_loss:<9,.4f} | "
                    f"${pos.take_profit:<9,.4f} | "
                    f"${pnl_usdt:+,.2f} ({pnl_pct:+.2f}%)"
                )
            logger.info("-" * 80)
            logger.info(f"  Toplam Pozisyon Maliyeti : ${total_cost:,.2f} USDT")
            logger.info(f"  Toplam Anlık Kar / Zarar : ${total_pnl:+,.2f} USDT (%{total_pnl/total_cost*100:+.2f}%)" if total_cost > 0 else f"  Toplam Anlık Kar / Zarar : ${total_pnl:+,.2f} USDT")
            
            total_value = self.execution_engine.usdt_balance + total_cost + total_pnl
            logger.info(f"  Toplam Portföy Değeri    : ${total_value:,.2f} USDT")
            
        logger.info("=" * 80 + "\n")
