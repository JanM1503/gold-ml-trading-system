"""
Zentrale Konfigurationsdatei für das Gold Trading Framework
Alle API-Keys werden aus Environment-Variablen geladen
"""
import os
from datetime import datetime


OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_API_URL = "https://api-fxpractice.oanda.com/v3"

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")

# ====================================
# DATEN-KONFIGURATION
# ====================================
DATA_START_DATE = "2005-01-01"
TRAIN_START_DATE = "2005-01-01"
TRAIN_END_DATE = "2014-12-31"
TEST_START_DATE = "2015-01-01"
TEST_END_DATE = "latest"

# Legacy datetime objects for backward compatibility
_DATA_START_DATETIME = datetime.strptime(DATA_START_DATE, "%Y-%m-%d")
_TRAIN_START_DATETIME = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
_TRAIN_END_DATETIME = datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
_TEST_START_DATETIME = datetime.strptime(TEST_START_DATE, "%Y-%m-%d")
START_DATE = DATA_START_DATE  # String format for compatibility
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

# Instrument-Konfiguration
GOLD_INSTRUMENT = "XAU_USD"
OIL_INSTRUMENT = "BCO_USD"  # Brent Crude Oil (Oanda)
SILVER_INSTRUMENT = "XAG_USD"
GRANULARITY = "M15"  # 15-Minuten-Candles

# Daten-Dateien
GOLD_CANDLES_FILE = f"{DATA_DIR}/gold_candles.json"
OIL_CANDLES_FILE = f"{DATA_DIR}/oil_candles.json"
SILVER_CANDLES_FILE = f"{DATA_DIR}/silver_candles.json"
VIX_FILE = f"{DATA_DIR}/vix.json"
YIELDS_FILE = f"{DATA_DIR}/yields_10y.json"
FEAR_GREED_FILE = f"{DATA_DIR}/fear_greed.json"
PIZZA_FILE = f"{DATA_DIR}/pentagon_pizza.json"
MACRO_FILE = f"{DATA_DIR}/macro_data.json"
SENTIMENT_FILE = f"{DATA_DIR}/sentiment.json"

# ====================================
# FEATURE ENGINEERING
# ====================================
# Technische Indikatoren
SMA_PERIODS = [20, 50, 200]
RSI_PERIODS = [14, 28]  # Both RSI-14 and RSI-28
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14

# Sentiment
SENTIMENT_ROLLING_WINDOWS = [4, 16, 96]  # 1h, 4h, 1d (bei 15min Candles)

# ====================================
# MACHINE LEARNING
# ====================================
# Random Forest Parameter
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 9  # REDUZIERT von 15 -> weniger Overfitting
RF_MIN_SAMPLES_SPLIT = 20  # ERHÖHT von 10 -> robuster
RF_MIN_SAMPLES_LEAF = 10  # ERHÖHT von 5 -> weniger Overfitting
RF_MAX_FEATURES = 'sqrt'  # NEU: Feature Subsampling
RF_CLASS_WEIGHT = 'balanced'  # NEU: Ausgleich für Klassenungleichgewicht
RF_RANDOM_STATE = 42

# Label-Generierung
FUTURE_RETURN_PERIODS = [4, 8, 16]  # 1h, 2h, 4h ahead
ENTRY_THRESHOLD_PCT = 0.2  # 0.15% for 15-min gold candles (was 0.7% - too high!)

# Mindest-Wahrscheinlichkeit für LONG/SHORT Trades (sonst FLAT)
MIN_TRADE_PROBA = 0.6  # z.B. 60%: darunter -> kein Trade

# Model-Dateien
ENTRY_MODEL_FILE = f"{MODELS_DIR}/entry_model.pkl"
RISK_MODEL_SIZE_FILE = f"{MODELS_DIR}/risk_model_size.pkl"
RISK_MODEL_SL_FILE = f"{MODELS_DIR}/risk_model_sl.pkl"
RISK_MODEL_TP_FILE = f"{MODELS_DIR}/risk_model_tp.pkl"
FEATURE_COLUMNS_FILE = f"{MODELS_DIR}/feature_columns.json"

# ====================================
# RISK MANAGEMENT (KOMPLETT ÜBERARBEITET - OHNE HEBEL)
# ====================================
# Kapitalgebundenes Risk Management OHNE Hebel
# FIXED: Reduced from 1.0% to 0.5% for realistic, sustainable returns
RISK_PER_TRADE_PCT = 0.8  # 0.5% des Kapitals pro Trade riskieren (realistic for gold trading)
MAX_LEVERAGE = 1.0  # KEIN Hebel - nur mit eigenem Kapital handeln
MAX_POSITION_PCT = 0.15  # NEW: Maximum 15% of capital per trade (prevents exponential growth)
MIN_CAPITAL_FOR_TRADING = 1000  # Minimum Kapital um zu traden

# Position Size Constraints (in Units, nicht CHF)
# Bei Gold ~2000 CHF/oz und 10k Kapital:
# - 1% Risiko = 100 CHF
# - Bei 2% Stop = 50 CHF SL-Distanz
# - Max Position Size = 100/50 = 2 Units (Bruchteile möglich)
MIN_POSITION_SIZE = 0.01  # Mindestens 0.01 Units (Micro-Lots)
MAX_POSITION_SIZE = 15  # Maximum 10 Units (bei 2000 CHF/oz = 20'000 CHF Notional)

# Stop Loss / Take Profit Defaults (werden von RiskEngine überschrieben)
DEFAULT_SL_ATR_MULTIPLIER = 2.0
DEFAULT_TP_ATR_MULTIPLIER = 3.0
MIN_RISK_REWARD = 2

# ====================================
# BACKTESTING
# ====================================
INITIAL_CAPITAL = 10000

# ====================================
# TRADING COSTS (SPREAD + SLIPPAGE)
# ====================================
# NO COMMISSIONS - All costs via spread + slippage
COMMISSION_PER_TRADE = 0.0  # Zero commission

# Fixed Spread Model (constant spread per ounce)
SPREAD_PER_OUNCE = 0.30  # $0.30 per ounce constant spread
# 0.1 oz -> $0.03, 1.0 oz -> $0.30, 5.0 oz -> $1.50

# ATR-Based Slippage Model
ATR_SLIPPAGE_MULT = 0.008  # Slippage = ATR * 0.02
# ATR=5 -> slippage=0.10, ATR=10 -> slippage=0.20, ATR=15 -> slippage=0.30

# ====================================
# LIVE TRADING
# ====================================
AUTO_TRADE = True  # ACTIVATED FOR TESTING - Set to False to disable
LIVE_TRADING_INTERVAL = 900  # Sekunden (15 Minuten)
MAX_OPEN_POSITIONS = 1

# ====================================
# LOGGING
# ====================================
BACKTEST_RESULTS_FILE = f"{LOGS_DIR}/backtest_results.json"
LIVE_TRADES_LOG_FILE = f"{LOGS_DIR}/live_trades.json"
SYSTEM_LOG_FILE = f"{LOGS_DIR}/system.log"

# ====================================
# EXTERNE DATEN-URLS
# ====================================
# Fear & Greed Index - scraped live from CNN
FEAR_GREED_URL = "https://edition.cnn.com/markets/fear-and-greed"

# NewsAPI Konfiguration
NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_QUERY_TERMS = ["gold", "gold price", "XAU", "precious metals"]
NEWS_LOOKBACK_DAYS = 7
NEWS_MAX_RESULTS = 100

# FinBERT Model
FINBERT_MODEL = "ProsusAI/finbert"

print(f"[OK] Konfiguration geladen")
print(f"[DATA] Daten-Startdatum: {DATA_START_DATE}")
print(f"[TRAIN] Train: {TRAIN_START_DATE} bis {TRAIN_END_DATE}")
print(f"[TEST] Test: {TEST_START_DATE} bis {TEST_END_DATE}")
print(f"[GOLD] Gold-Instrument: {GOLD_INSTRUMENT}")
print(f"[TRADE] Auto-Trading: {'AKTIV' if AUTO_TRADE else 'INAKTIV'}")


