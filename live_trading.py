"""
Live Trading Modul für Oanda Demo Account
Macht alle 15 Minuten Predictions und platziert optional Orders
"""
import time
import json
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import config
from oanda_client import OandaClient
from data_loader import DataManager
from feature_engineering import FeatureEngineer
from models import TradingModels
from explainability import ModelExplainer
from sentiment import SentimentAnalyzer
from fear_greed import FearGreedScraper
from risk_engine import RiskEngine
from fred_data import load_fred_data


class LiveTrader:
    """Live Trading mit Random Forest"""
    
    def __init__(self):
        self.oanda_client = OandaClient()
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        self.models = TradingModels()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.fear_greed_scraper = FearGreedScraper()
        
        if not self.models.load_models():
            raise ValueError("Models konnten nicht geladen werden! Trainiere zuerst die Models.")
        
        self.explainer = ModelExplainer(self.models)
        self.risk_engine = RiskEngine()
        self.trade_log = []
    
    def get_latest_features(self) -> Dict:
        """
        Lädt neueste Daten und erstellt Features
        
        Returns:
            Dict mit feature_df und latest_candle
        """
        # Update alle Datenquellen
        all_data = self.data_manager.load_all_data(force_reload=False)
        
        # Load FRED macro data
        fred_data = load_fred_data(force_reload=False)
        
        # Erstelle Feature-Matrix (ohne Labels für Live)
        df = self.feature_engineer.create_feature_matrix(
            candle_data=all_data['gold_candles'][-500:],  # Nur letzte 500 Candles für Performance
            macro_data={
                'oil_candles': all_data['oil_candles'][-500:],
                'silver_candles': all_data['silver_candles'][-500:],
                'vix': all_data['vix'],
                'yields': all_data['yields'],
                'fear_greed': all_data['fear_greed'],
                'macro': all_data['macro']
            },
            sentiment_data=[],  # TODO: Live Sentiment
            fred_data=fred_data,
            generate_labels=False
        )
        
        # Neueste Zeile
        latest_features = df.iloc[-1:].copy()
        
        return {
            'features': latest_features,
            'timestamp': latest_features['timestamp'].values[0],
            'close_price': latest_features['close'].values[0]
        }
    
    def get_live_sentiment(self, text: str = None) -> Dict:
        """
        Gets live sentiment from recent news or provided text
        
        Returns:
            Dict with sentiment info
        """
        if text is None:
            text = "Gold market sentiment analysis"
        
        sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        return sentiment
    
    def get_fear_greed(self) -> Optional[Dict]:
        """
        Gets current Fear & Greed Index from CNN
        
        Returns:
            Dict with fear_greed value and classification
        """
        return self.fear_greed_scraper.scrape_with_classification()
    
    def make_prediction(self, features_dict: Dict) -> Dict:
        """
        Macht Prediction für aktuellen Candle mit Sentiment und Fear & Greed
        
        Returns:
            Prediction mit Explanation, Sentiment und Fear & Greed
        """
        features = features_dict['features']
        
        ml_prediction = self.models.predict(features)
        
        live_sentiment = self.get_live_sentiment()
        fear_greed = self.get_fear_greed()
        
        explanation = self.explainer.explain_prediction(
            features=features,
            prediction=ml_prediction,
            top_n=10
        )
        
        return {
            'timestamp': str(features_dict['timestamp']),
            'close_price': features_dict['close_price'],
            'prediction': ml_prediction,
            'sentiment': live_sentiment,
            'fear_greed': fear_greed,
            'explanation': explanation
        }
    
    def execute_trade(self, prediction_result: Dict):
        """
        Führt Trade aus mit ML, Sentiment und Fear & Greed
        
        Args:
            prediction_result: Dict mit Prediction, Sentiment, Fear & Greed
        """
        ml_pred = prediction_result['prediction']
        sentiment = prediction_result.get('sentiment', {})
        fear_greed = prediction_result.get('fear_greed', {})
        close_price = prediction_result['close_price']
        
        print("\n" + "="*70)
        print("[*] TRADING DECISION ANALYSIS")
        print("="*70)
        
        print(f"\n[ML] Signal: {ml_pred['entry_signal_name']}")
        print(f"     Confidence: {ml_pred['confidence']:.2%}")
        print(f"     Risk %: {ml_pred['risk_pct']:.2f}%")
        print(f"     SL ATR Mult: {ml_pred['sl_atr_multiplier']:.2f}")
        print(f"     TP ATR Mult: {ml_pred['tp_atr_multiplier']:.2f}")
        
        print(f"\n[SENTIMENT] Score: {sentiment.get('sentiment_score', 0):.3f}")
        print(f"            Label: {sentiment.get('sentiment_label', 0)}")
        print(f"            Confidence: {sentiment.get('confidence', 0):.2%}")
        
        if fear_greed:
            print(f"\n[FEAR & GREED] Value: {fear_greed.get('fear_greed', 'N/A')}")
            print(f"               Classification: {fear_greed.get('classification', 'N/A')}")
        
        if ml_pred['entry_signal'] == 0:
            print("\n⏸ FLAT - No trade signal")
            print("="*70)
            return
        
        if not config.AUTO_TRADE:
            print("\n[WARN] AUTO_TRADE disabled - Trade NOT executed")
            print("       Set AUTO_TRADE=True in config.py to enable")
            print("="*70)
            return
        
        open_positions = self.oanda_client.get_open_positions()
        if len(open_positions) >= config.MAX_OPEN_POSITIONS:
            print(f"\n[WARN] Max positions ({config.MAX_OPEN_POSITIONS}) reached")
            print("="*70)
            return
        
        account_info = self.oanda_client.get_account_info()
        current_capital = float(account_info.get('balance', 10000))
        
        features_row = prediction_result.get('features', pd.DataFrame())
        atr = features_row['atr'].values[0] if 'atr' in features_row.columns else close_price * 0.01
        
        direction = "LONG" if ml_pred['entry_signal'] == 1 else "SHORT"
        
        risk_params = self.risk_engine.calculate_position(
            capital=current_capital,
            entry_price=close_price,
            atr=atr,
            direction=direction,
            ml_risk_pct=ml_pred['risk_pct'],
            ml_sl_atr_mult=ml_pred['sl_atr_multiplier'],
            ml_tp_atr_mult=ml_pred['tp_atr_multiplier']
        )
        
                # Oanda requires integer units for XAU_USD (1 unit = 1 oz)
                # Round to integer, minimum 1 unit
        position_size = max(1, int(round(risk_params['position_size'])))
        
                # Determine units (1 unit = 1 oz for XAU/USD)
        units = max(1, int(round(risk_params['position_size'])))

        if ml_pred['entry_signal'] == 1:     # LONG
            units = units
        else:                                # SHORT
            units = -units

        # ================================
        # FIX: Enforce Oanda precision
        # ================================
        # XAU/USD typically allows 1-2 decimals (safe: use 2)
        stop_loss = round(risk_params['stop_loss'], 2)
        take_profit = round(risk_params['take_profit'], 2)

        # ================================
        # FIX: Ensure SL/TP are VALID for Oanda
        # ================================
        if units > 0:      # LONG
            # SL MUST be below entry
            if stop_loss >= close_price:
                stop_loss = round(close_price - 0.20, 2)
            # TP MUST be above entry
            if take_profit <= close_price:
                take_profit = round(close_price + 0.20, 2)

        else:              # SHORT
            # SL MUST be above entry
            if stop_loss <= close_price:
                stop_loss = round(close_price + 0.20, 2)
            # TP MUST be below entry
            if take_profit >= close_price:
                take_profit = round(close_price - 0.20, 2)

        
        print(f"\n[EXECUTE] Signal: {ml_pred['entry_signal_name']}")
        print(f"          Position: {abs(units):.2f} oz")
        print(f"          Entry: ${close_price:.2f}")
        print(f"          Stop Loss: ${stop_loss:.2f}")
        print(f"          Take Profit: ${take_profit:.2f}")
        print(f"          Risk/Reward: {risk_params['risk_reward_ratio']:.2f}")
        
        order_result = self.oanda_client.place_market_order(
            instrument=config.GOLD_INSTRUMENT,
            units=units,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if order_result:
            trade_log_entry = {
                'timestamp': prediction_result['timestamp'],
                'signal': ml_pred['entry_signal_name'],
                'units': units,
                'entry_price': close_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'ml_prediction': ml_pred,
                'sentiment': sentiment,
                'fear_greed': fear_greed,
                'risk_params': risk_params,
                'order_result': str(order_result)
            }
            
            self.trade_log.append(trade_log_entry)
            self._save_trade_log()
            print(f"\n✅ {direction} Trade executed")
        
        print("="*70)
    
    def _save_trade_log(self):
        """Speichert Trade-Log"""
        with open(config.LIVE_TRADES_LOG_FILE, 'w') as f:
            json.dump(self.trade_log, f, indent=2)
    
    def run_live_loop(self):
        """
        Hauptschleife für Live Trading
        Läuft alle 15 Minuten
        """
        print("\n" + "="*70)
        print("[*] LIVE TRADING GESTARTET")
        print("="*70)
        print(f"Instrument: {config.GOLD_INSTRUMENT}")
        print(f"Interval: {config.LIVE_TRADING_INTERVAL}s ({config.LIVE_TRADING_INTERVAL//60} Minuten)")
        print(f"Auto-Trade: {'AKTIV' if config.AUTO_TRADE else 'INAKTIV'}")
        print("="*70)
        print("\nDrücke Ctrl+C zum Beenden\n")
        
        try:
            while True:
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 70)
                
                print("[DOWNLOAD] Lade aktuelle Daten...")
                features_dict = self.get_latest_features()
                
                print("🤖 Mache Prediction...")
                prediction_result = self.make_prediction(features_dict)
                prediction_result['features'] = features_dict['features']
                
                print(prediction_result['explanation'])
                
                self.execute_trade(prediction_result)
                
                # Warte bis zum nächsten Interval
                print(f"\n⏳ Warte {config.LIVE_TRADING_INTERVAL}s bis zur nächsten Analyse...")
                time.sleep(config.LIVE_TRADING_INTERVAL)
        
        except KeyboardInterrupt:
            print("\n\n⏹[*]  Live Trading gestoppt")
            print(f"Trades durchgeführt: {len(self.trade_log)}")
    
    def run_once(self):
        """
        Führt eine einzelne Prediction aus (für Testing)
        """
        print("\n" + "="*70)
        print("[*] SINGLE PREDICTION")
        print("="*70)
        
        features_dict = self.get_latest_features()
        
        prediction_result = self.make_prediction(features_dict)
        prediction_result['features'] = features_dict['features']
        
        print(prediction_result['explanation'])
        
        self.execute_trade(prediction_result)
        
        print("\n[OK] Prediction abgeschlossen")
    
    def test_trade_signal(self, signal_type: str = "LONG"):
        """
        Testet die Trading-Pipeline mit einem simulierten Signal
        
        Args:
            signal_type: "LONG" oder "SHORT"
        """
        print("\n" + "="*70)
        print(f"[TEST] SIMULIERE {signal_type} SIGNAL")
        print("="*70)
        print("[INFO] Dies ist ein TEST - kein echtes Signal!")
        print("="*70)
        
        # Hole aktuelle Features
        features_dict = self.get_latest_features()
        
        # Erstelle Test-Prediction
        features = features_dict['features']
        
        # Simuliere ein starkes Signal
        test_prediction = {
            "entry_signal": 1 if signal_type == "LONG" else -1,
            "entry_signal_name": signal_type,
            "confidence": 0.85,  # 85% Konfidenz
            "risk_pct": 1.0,  # 1% Risiko
            "sl_atr_multiplier": 2.0,
            "tp_atr_multiplier": 3.0,
            "entry_proba": {
                "SHORT": 0.05,
                "FLAT": 0.10,
                "LONG": 0.85
            } if signal_type == "LONG" else {
                "SHORT": 0.85,
                "FLAT": 0.10,
                "LONG": 0.05
            }
        }
        
        # Hole Live-Daten
        live_sentiment = self.get_live_sentiment()
        fear_greed = self.get_fear_greed()
        
        # Erstelle Test-Result
        prediction_result = {
            'timestamp': str(features_dict['timestamp']),
            'close_price': features_dict['close_price'],
            'prediction': test_prediction,
            'sentiment': live_sentiment,
            'fear_greed': fear_greed,
            'features': features,
            'explanation': f"\n[TEST MODE] Simuliertes {signal_type} Signal mit 85% Konfidenz\n"
        }
        
        print(prediction_result['explanation'])
        
        # Führe Trade aus (wird nur ausgeführt wenn AUTO_TRADE=True)
        self.execute_trade(prediction_result)
        
        print("\n[OK] Test abgeschlossen")
        print("[INFO] Um den Trade tatsächlich auszuführen, setze AUTO_TRADE=True in config.py")


if __name__ == "__main__":
    trader = LiveTrader()
    
    # Einzelne Prediction zum Testen
    trader.run_once()
    
    # Für dauerhaftes Live Trading:
    # trader.run_live_loop()
