# 🥇 Gold Trading Framework

Ein vollwertiges Python-Framework für algorithmisches Gold-Trading mit Machine Learning, Backtesting und Live-Demo-Trading über Oanda.

## 📋 Features

- **Random Forest ML Models** für Entry-Signale (LONG/SHORT/FLAT) und Risk Management
- **Umfangreiche Datenquellen**: Oanda API (Gold, Oil, Silver), VIX, Yields, Makrodaten, Sentiment
- **Inkrementelles Daten-Management**: Effizientes Caching, nur neue Daten werden nachgeladen
- **Feature Engineering**: 50+ technische Indikatoren, Makro-Features, Sentiment-Analyse mit FinBERT
- **Explainable AI**: Feature Importances und detaillierte Erklärungen für jede Entscheidung
- **Backtesting Engine**: Vollständige Performance-Analyse mit Equity Curve, Win Rate, Sharpe Ratio
- **Live Trading**: Automatisches oder manuelles Trading über Oanda Demo Account

## 🗂️ Projektstruktur

```
gold_trading_framework/
├── config.py                 # Zentrale Konfiguration
├── data_loader.py            # Inkrementelles Daten-Management
├── oanda_client.py           # Oanda API Client
├── feature_engineering.py    # Feature-Berechnung
├── sentiment.py              # NewsAPI + FinBERT Sentiment
├── models.py                 # Random Forest Training & Prediction
├── explainability.py         # Model-Erklärungen
├── backtest_optimized.py     # Backtesting Engine (schnelle, optimierte Version)
├── live_trading.py           # Live-Trading-Modul
├── run.py                    # Haupteinstiegspunkt
├── requirements.txt          # Python-Dependencies
├── data/                     # JSON-Datenfiles (wird automatisch erstellt)
├── models/                   # Trainierte Models (wird automatisch erstellt)
└── logs/                     # Logs & Ergebnisse (wird automatisch erstellt)
```

## 🚀 Setup

### 1. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 2. API-Keys konfigurieren

Setze folgende Environment-Variablen:

```bash
# Oanda Demo Account
export OANDA_API_KEY="dein_oanda_api_key"
export OANDA_ACCOUNT_ID="dein_oanda_account_id"

# NewsAPI (optional, für Sentiment)
export NEWS_API_KEY="dein_newsapi_key"
```

**Windows (PowerShell):**
```powershell
$env:OANDA_API_KEY="dein_oanda_api_key"
$env:OANDA_ACCOUNT_ID="dein_oanda_account_id"
$env:NEWS_API_KEY="dein_newsapi_key"
```

### 3. Oanda API-Keys erhalten

1. Registriere dich bei [Oanda fxTrade Practice](https://www.oanda.com/forex-trading/account-types/demo-account/)
2. Gehe zu "Manage API Access"
3. Erstelle einen neuen API Token (für Practice Account)
4. Kopiere API Key und Account ID

### 4. NewsAPI Key erhalten (optional)

1. Registriere dich bei [NewsAPI.org](https://newsapi.org)
2. Free-Tier ist ausreichend (100 Requests/Tag)
3. Kopiere den API Key

## 📊 Usage

### Modus 1: Kompletter Workflow (Empfohlen für ersten Start)

```bash
python run.py --mode full
```

Das führt automatisch durch:
1. ✅ Daten laden (inkrementell)
2. ✅ Features erstellen
3. ✅ Models trainieren
4. ✅ Backtest durchführen

### Modus 2: Nur Daten laden

```bash
python run.py --mode data
```

Lädt alle Datenquellen und cached sie als JSON.

**Force Reload** (alle Daten neu laden):
```bash
python run.py --mode data --force-reload
```

### Modus 3: Models trainieren

```bash
python run.py --mode train
```

Trainiert Random Forest Models auf den vorhandenen Daten.

### Modus 4: Backtest

```bash
python run.py --mode backtest
```

Führt Backtest durch und zeigt Performance-Metriken.

### Modus 5: Live Trading (Demo)

**Single Prediction** (einmalig):
```bash
python run.py --mode live
```

Holt aktuelle Marktdaten, macht eine Prediction und zeigt die Explanation.

**Kontinuierliches Live Trading** (alle 15 Minuten):
```bash
python run.py --mode live --live-loop
```

⚠️ **Auto-Trading aktivieren:**
Um automatisch Orders zu platzieren, setze in `config.py`:
```python
AUTO_TRADE = True
```

## 🎯 Beispiel-Output

### Backtest Performance

```
======================================================================
📊 BACKTEST PERFORMANCE
======================================================================
Initial Capital:    $10,000.00
Final Capital:      $12,450.00
Total Return:       24.50%

Trades:             42
Winning:            26 (61.9%)
Losing:             16
Avg Win:            $215.50
Avg Loss:           $-128.30
Profit Factor:      2.15

Max Drawdown:       -8.45%
Sharpe Ratio:       1.82
======================================================================
```

### Live Prediction mit Explanation

```
======================================================================
🔍 RANDOM FOREST ENTSCHEIDUNG
======================================================================

📊 SIGNAL: LONG
   Confidence: 72.3%
   Proba: LONG=72.3% | FLAT=18.2% | SHORT=9.5%

💰 RISK MANAGEMENT:
   Position Size: 150 units
   Stop Loss: 2015.45 (ATR Mult: 2.00)
   Take Profit: 2030.75 (ATR Mult: 3.00)
   Risk/Reward: 1:1.50

🎯 WICHTIGSTE EINFLUSSFAKTOREN (Top 10):

   1. dist_to_sma_200
      Importance: 0.1245 | Aktueller Wert: 1.25%
   2. rsi
      Importance: 0.0892 | Aktueller Wert: 45.3
   3. sentiment_score
      Importance: 0.0765 | Aktueller Wert: 0.234
   ...

📈 MARKTKONTEXT:
   RSI: 45.3 (neutral)
   Preis über 200-SMA: 1.25%
   Bollinger Band Position: 0.58 (im mittleren Bereich)
   Volatilität (ATR): 0.45%
   Sentiment: 0.234 (bullish)
   VIX: 15.2 (niedrig)

======================================================================
```

## 🔧 Konfiguration

Alle wichtigen Parameter befinden sich in `config.py`:

### Daten-Konfiguration
- `DATA_START_DATE`: Ab wann Daten geladen werden (default: 2005-01-01)
- `GOLD_INSTRUMENT`, `OIL_INSTRUMENT`, `SILVER_INSTRUMENT`: Instrumente
- `GRANULARITY`: Candle-Größe (default: M15 = 15 Minuten)

### Random Forest Parameter
- `RF_N_ESTIMATORS`: Anzahl Trees (default: 200)
- `RF_MAX_DEPTH`: Maximale Tiefe (default: 9)
- `RF_MIN_SAMPLES_SPLIT`, `RF_MIN_SAMPLES_LEAF`: Pruning-Parameter
- `RF_CLASS_WEIGHT`: Klassen-Gewichtung (default: "balanced")

### Entry-Threshold
- `ENTRY_THRESHOLD_PCT`: Mindest-Return für LONG/SHORT Signal (default: 0.2%)

### Risk Management (kapitalgebunden)
- `RISK_PER_TRADE_PCT`: Anteil des Kapitals, der pro Trade riskiert wird (default: 0.5%)
- `MAX_LEVERAGE`: Maximaler Hebel (default: 1.0 → kein Hebel)
- `MAX_POSITION_PCT`: Maximaler Anteil des Kapitals pro Position (default: 15%)
- `MIN_POSITION_SIZE` / `MAX_POSITION_SIZE`: Untere/obere Grenze für Positionsgröße in Units
- `DEFAULT_SL_ATR_MULTIPLIER` / `DEFAULT_TP_ATR_MULTIPLIER`: Fallback-Multiplikatoren für SL/TP in ATR

### Backtesting
- `INITIAL_CAPITAL`: Startkapital (default: $10,000)

### Live Trading
- `AUTO_TRADE`: Automatisches Trading aktivieren (default: False – aus Sicherheitsgründen)
- `LIVE_TRADING_INTERVAL`: Interval in Sekunden (default: 900 = 15min)
- `MAX_OPEN_POSITIONS`: Max. gleichzeitige Positionen (default: 1)

## 📦 Daten-Management

Das Framework verwendet **inkrementelles Daten-Update**:

1. **Erster Lauf**: Lädt alle Daten ab 2005-01-01
2. **Folgende Läufe**: Lädt nur neue Daten ab dem letzten Timestamp
3. **Force Reload**: Mit `--force-reload` Flag werden alle Daten neu geladen

Alle Daten werden als JSON in `data/` gespeichert:
- `gold_candles.json`: Gold OHLCV-Daten
- `oil_candles.json`: Oil OHLCV-Daten
- `silver_candles.json`: Silver OHLCV-Daten
- `sentiment.json`: News-Artikel mit FinBERT-Sentiment
- `vix.json`, `yields_10y.json`, etc.: Makrodaten (Placeholder)

## 🧠 Machine Learning Pipeline

### 1. Feature Engineering
- **Technische Indikatoren**: SMA (20/50/200), RSI, Bollinger Bands, ATR, Returns, Momentum
- **Makro-Features**: Oil, Silver, VIX, Yields, Fear & Greed, CPI, NFP, etc.
- **Sentiment-Features**: FinBERT-Score, Rolling Sentiment (1h/4h/1d)
- **Event-Dummies**: NFP Day, CPI Day, FOMC Week, US Trading Hours

### 2. Label-Generierung
- **LONG (1)**: Future Return >= 0.3%
- **SHORT (-1)**: Future Return <= -0.3%
- **FLAT (0)**: Sonst

### 3. Model Training
- **Entry Model**: RandomForestClassifier (LONG/SHORT/FLAT)
- **Risk Models**: 3x RandomForestRegressor (Position Size, SL, TP)

### 4. Explainability
- **Feature Importances**: Welche Features sind global wichtig?
- **Lokale Erklärungen**: Warum wurde diese spezifische Entscheidung getroffen?

## 📈 Erweiterte Nutzung

### Eigene Makrodaten einpflegen

Makrodaten (VIX, Yields, etc.) sind derzeit Placeholders. Um echte Daten zu nutzen:

1. Implementiere API-Calls in `data_loader.py` (z.B. FRED API)
2. Oder: Importiere CSV-Dateien manuell ins `data/` Verzeichnis
3. Format: `{"data": [{"timestamp": "...", "vix": ...}, ...]}`

### Sentiment erweitern

Für kontinuierliches Sentiment:
1. Aktiviere NewsAPI-Integration (API Key setzen)
2. Für Live-Trading: Sentiment wird automatisch nachgeladen
3. Für historisches Sentiment: Nutze `--force-reload` regelmäßig

### Custom Features hinzufügen

In `feature_engineering.py`:
1. Erstelle neue Berechnungsmethode
2. Füge sie zu `calculate_technical_features()` oder `create_feature_matrix()` hinzu
3. Features werden automatisch in Models integriert

## ⚠️ Wichtige Hinweise

### Risiken
- **Demo-Account verwenden**: Dieses Framework ist für Lern- und Forschungszwecke
- **Keine Finanzberatung**: Models können falsch liegen
- **Backtesting ≠ Live Performance**: Slippage, Latenz, etc. nicht simuliert

### Performance
- **Erste Ausführung**: Kann lange dauern (Daten ab 2005, FinBERT-Download)
- **FinBERT**: Sentiment-Analyse ist CPU/GPU-intensiv
- **Oanda API**: Rate Limits beachten (max 120 Requests/Minute)

### Datenqualität
- **Oanda Historical Data**: Begrenzte History (typisch 3-5 Jahre)
- **Makrodaten**: Placeholder - echte Daten benötigen zusätzliche APIs
- **Sentiment**: NewsAPI Free Tier = letzte 30 Tage

## 🛠️ Troubleshooting

### "Models nicht gefunden"
→ Führe zuerst Training aus: `python run.py --mode train`

### "API Key nicht konfiguriert"
→ Setze Environment-Variablen (siehe Setup)

### "FinBERT Download-Fehler"
→ Erste Ausführung lädt FinBERT (~500MB). Bei Netzwerkproblemen: Retry

### "Keine Daten verfügbar"
→ Oanda Demo Account hat begrenzte History. Nutze kleineres `DATA_START_DATE`

### Memory-Fehler
→ Reduziere Datenumfang oder nutze kleinere `RF_N_ESTIMATORS`

## 📚 Ressourcen

- [Oanda API Dokumentation](https://developer.oanda.com/rest-live-v20/introduction/)
- [NewsAPI Dokumentation](https://newsapi.org/docs)
- [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- [Sklearn RandomForest Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)

## 📄 Lizenz

Dieses Projekt ist für Bildungs- und Forschungszwecke. Keine Garantie für Profitabilität.

## 🤝 Beitragen

Verbesserungen und Erweiterungen sind willkommen:
- Zusätzliche Datenquellen integrieren
- Weitere ML-Modelle testen
- Bessere Risk Management Strategien
- Performance-Optimierungen

## 📞 Support

Bei Fragen oder Problemen:
1. Prüfe die Console-Ausgabe auf Fehlermeldungen
2. Validiere API-Keys und Konfiguration
3. Checke `logs/system.log` für Details

---

**Viel Erfolg beim Trading! 🚀📈**


