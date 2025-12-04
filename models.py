"""
Machine Learning Models: Random Forest für Entry und Risk Management
"""
import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import config


class TradingModels:
    """Random Forest Models für Entry-Signale und Risk Management
    
    - Entry Model: Vorhersage von LONG/SHORT/FLAT
    - Risk Models: Vorhersage von Risk-% (1-3%), SL und TP ATR-Multiplikatoren
    - RiskEngine validiert und begrenzt die ML-Predictions mit Kapital-Checks
    """
    
    def __init__(self):
        self.entry_model = None
        self.risk_model_pct = None  # Sagt Risiko-% vorher (1-3%)
        self.risk_model_sl = None   # Sagt SL ATR-Multiplikator vorher
        self.risk_model_tp = None   # Sagt TP ATR-Multiplikator vorher
        self.feature_columns = None
    
    def train_entry_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> RandomForestClassifier:
        """
        Trainiert Entry-Model (LONG/SHORT/FLAT)
        
        Args:
            X_train: Features
            y_train: Labels (-1, 0, 1)
        
        Returns:
            Trainiertes Model
        
        ÄNDERUNGEN:
        - class_weight='balanced' für Klassenungleichgewicht
        - max_features='sqrt' für Feature Subsampling
        - Konservativere Hyperparameter gegen Overfitting
        """
        print("\n[TOOL] Trainiere Entry-Model...")
        print(f"  Klassenverteilung: {dict(y_train.value_counts().sort_index())}")
        
        model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            max_features=config.RF_MAX_FEATURES,
            class_weight=config.RF_CLASS_WEIGHT,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1,
            verbose=0  # GEÄNDERT: Kein verbose Output mehr
        )
        
        model.fit(X_train, y_train)
        
        print(f"[OK] Entry-Model trainiert")
        
        return model
    
    def train_risk_models(
        self, 
        X_train: pd.DataFrame, 
        y_risk_pct: pd.Series,
        y_sl: pd.Series,
        y_tp: pd.Series
    ) -> Tuple[RandomForestRegressor, RandomForestRegressor, RandomForestRegressor]:
        """
        Trainiert Risk-Models (Risk-%, SL, TP)
        
        Diese Models sagen Risiko-Parameter vorher, die dann von der
        RiskEngine mit Kapital-Checks validiert werden.
        
        Returns:
            Tuple von (risk_pct_model, sl_model, tp_model)
        """
        print("\n[TOOL] Trainiere Risk-Management-Models...")
        
        # Risk-% Model (1-3% vom Kapital)
        print("  → Risk-% Model...")
        risk_pct_model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            max_features=config.RF_MAX_FEATURES,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        risk_pct_model.fit(X_train, y_risk_pct)
        print("     ✓ Trainiert")
        
        # Stop Loss ATR-Multiplikator Model
        print("  → Stop Loss ATR Model...")
        sl_model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            max_features=config.RF_MAX_FEATURES,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        sl_model.fit(X_train, y_sl)
        print("     ✓ Trainiert")
        
        # Take Profit ATR-Multiplikator Model
        print("  → Take Profit ATR Model...")
        tp_model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            max_features=config.RF_MAX_FEATURES,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        tp_model.fit(X_train, y_tp)
        print("     ✓ Trainiert")
        
        print(f"[OK] Risk-Models trainiert")
        
        return risk_pct_model, sl_model, tp_model
    
    def evaluate_entry_model(
        self, 
        model: RandomForestClassifier, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ):
        """Evaluiert Entry-Model"""
        print("\n[DATA] EVALUIERE ENTRY-MODEL")
        print("="*60)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['SHORT', 'FLAT', 'LONG']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("="*60)
    
    def train_and_save_models(
        self, 
        df: pd.DataFrame,
        feature_columns: List[str]
    ):
        """
        Trainiert Entry-Model und speichert es
        
        Args:
            df: Feature-Matrix mit Labels
            feature_columns: Liste der Feature-Spalten
        """
        print("\n" + "="*70)
        print("[*] TRAINIERE ALLE MODELS - OUT-OF-SAMPLE EVALUATION")
        print("="*70)
        
        # Train-Test-Split nach Datum - STRICT TIME-BASED SPLIT
        train_start = pd.to_datetime(config.TRAIN_START_DATE)
        train_end = pd.to_datetime(config.TRAIN_END_DATE)
        test_start = pd.to_datetime(config.TEST_START_DATE)
        
        train_df = df[(df['timestamp'] >= train_start) & 
                      (df['timestamp'] <= train_end)].copy()
        test_df = df[df['timestamp'] >= test_start].copy()
        
        print(f"\n{'='*70}")
        print(f"TRAINING ON: {config.TRAIN_START_DATE} to {config.TRAIN_END_DATE}")
        print(f"  Samples: {len(train_df):,}")
        print(f"  Period:  {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"\nTESTING ON:  {config.TEST_START_DATE} to {config.TEST_END_DATE}")
        print(f"  Samples: {len(test_df):,}")
        print(f"  Period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        print(f"{'='*70}")
        
        # Verify no overlap
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("[ERROR] Train or test set is empty! Check date ranges.")
        
        if train_df['timestamp'].max() >= test_df['timestamp'].min():
            raise ValueError("[ERROR] Train and test sets overlap! Data leakage detected.")
        
        # Features und Labels extrahieren
        X_train = train_df[feature_columns].copy()
        X_test = test_df[feature_columns].copy()
        
        # KRITISCH: NaN und Infinite Values behandeln
        print(f"\n[DATA] Bereinige Features...")
        print(f"  NaN vor Bereinigung - Train: {X_train.isna().sum().sum()}, Test: {X_test.isna().sum().sum()}")
        
        # Ersetze Infinite durch NaN, dann fillna mit 0
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"  NaN nach Bereinigung - Train: {X_train.isna().sum().sum()}, Test: {X_test.isna().sum().sum()}")
        print(f"  Features: {len(feature_columns)}")
        
        y_entry_train = train_df['label_entry']
        y_entry_test = test_df['label_entry']
        
        y_risk_pct_train = train_df['label_risk_pct']
        y_sl_train = train_df['label_sl_atr']
        y_tp_train = train_df['label_tp_atr']
        
        # Prüfe ob Labels gültig sind
        print(f"\n[DATA] Labels-Validierung...")
        print(f"  Entry Labels gültig: {y_entry_train.isna().sum() == 0}")
        print(f"  Risk-% Range: {y_risk_pct_train.min():.2f} - {y_risk_pct_train.max():.2f}")
        print(f"  SL ATR Range: {y_sl_train.min():.2f} - {y_sl_train.max():.2f}")
        print(f"  TP ATR Range: {y_tp_train.min():.2f} - {y_tp_train.max():.2f}")
        
        # Analyze label distribution before training
        print(f"\n[DATA] Training Label Distribution:")
        train_label_counts = y_entry_train.value_counts().sort_index()
        for label, count in train_label_counts.items():
            pct = count / len(y_entry_train) * 100
            label_name = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}[label]
            print(f"  {label_name:6s} ({label:2d}): {count:6d} ({pct:5.1f}%)")
        
        # Warn if severe imbalance
        long_pct = train_label_counts.get(1, 0) / len(y_entry_train) * 100
        short_pct = train_label_counts.get(-1, 0) / len(y_entry_train) * 100
        flat_pct = train_label_counts.get(0, 0) / len(y_entry_train) * 100
        
        if long_pct < 5 or short_pct < 5:
            print(f"\n  [WARN] SEVERE CLASS IMBALANCE!")
            print(f"        LONG: {long_pct:.1f}%  |  SHORT: {short_pct:.1f}%  |  FLAT: {flat_pct:.1f}%")
            print(f"        This will cause the model to barely predict the minority class!")
            print(f"        Consider adjusting ENTRY_THRESHOLD_PCT in config.py")
        
        # Trainiere Entry-Model
        self.entry_model = self.train_entry_model(X_train, y_entry_train)
        self.evaluate_entry_model(self.entry_model, X_test, y_entry_test)
        
        # Trainiere Risk-Models
        self.risk_model_pct, self.risk_model_sl, self.risk_model_tp = self.train_risk_models(
            X_train, y_risk_pct_train, y_sl_train, y_tp_train
        )
        
        # Speichere Feature-Columns
        self.feature_columns = feature_columns
        
        # Speichere Models
        print("\n[SAVE] Speichere Models...")
        
        with open(config.ENTRY_MODEL_FILE, 'wb') as f:
            pickle.dump(self.entry_model, f)
        print(f"  [*] {config.ENTRY_MODEL_FILE}")
        
        with open(config.RISK_MODEL_SIZE_FILE, 'wb') as f:
            pickle.dump(self.risk_model_pct, f)
        print(f"  [*] {config.RISK_MODEL_SIZE_FILE} (Risk-%)")
        
        with open(config.RISK_MODEL_SL_FILE, 'wb') as f:
            pickle.dump(self.risk_model_sl, f)
        print(f"  [*] {config.RISK_MODEL_SL_FILE}")
        
        with open(config.RISK_MODEL_TP_FILE, 'wb') as f:
            pickle.dump(self.risk_model_tp, f)
        print(f"  [*] {config.RISK_MODEL_TP_FILE}")
        
        with open(config.FEATURE_COLUMNS_FILE, 'w') as f:
            json.dump(feature_columns, f, indent=2)
        print(f"  [*] {config.FEATURE_COLUMNS_FILE}")
        
        print("\n[OK] ALLE MODELS TRAINIERT UND GESPEICHERT")
        print("[INFO] ML-Models sagen Risk-%, SL, TP vorher")
        print("[INFO] RiskEngine validiert Predictions mit Kapital-Checks")
        print("="*60)
    
    def load_models(self):
        """Lädt trainierte Models"""
        print("[*] Lade trainierte Models...")
        
        try:
            with open(config.ENTRY_MODEL_FILE, 'rb') as f:
                self.entry_model = pickle.load(f)
            
            with open(config.RISK_MODEL_SIZE_FILE, 'rb') as f:
                self.risk_model_pct = pickle.load(f)
            
            with open(config.RISK_MODEL_SL_FILE, 'rb') as f:
                self.risk_model_sl = pickle.load(f)
            
            with open(config.RISK_MODEL_TP_FILE, 'rb') as f:
                self.risk_model_tp = pickle.load(f)
            
            with open(config.FEATURE_COLUMNS_FILE, 'r') as f:
                self.feature_columns = json.load(f)
            
            print(f"[OK] Alle Models geladen ({len(self.feature_columns)} Features)")
            return True
        
        except Exception as e:
            print(f"[ERROR] Fehler beim Laden der Models: {e}")
            return False
    
    def predict(
        self, 
        features: pd.DataFrame
    ) -> Dict:
        """
        Macht Prediction für Entry und Risk Management
        
        Args:
            features: DataFrame mit Features (1 Zeile)
        
        Returns:
            Dict mit Entry-Signal und Risk-Parametern
            - risk_pct: ML-vorhergesagtes Risiko in % (1-3%)
            - sl_atr_multiplier: ML-vorhergesagter SL ATR-Mult
            - tp_atr_multiplier: ML-vorhergesagter TP ATR-Mult
        """
        if not self.entry_model:
            raise ValueError("Models nicht geladen! Rufe load_models() auf.")
        
        # Features vorbereiten
        X = features[self.feature_columns].fillna(0)
        
        # Entry Prediction
        entry_pred = self.entry_model.predict(X)[0]
        entry_proba = self.entry_model.predict_proba(X)[0]

        # Model liefert direkt unsere Labels (-1=SHORT, 0=FLAT, 1=LONG)
        entry_signal = int(entry_pred)
        
        # Confidence
        confidence = entry_proba.max()
        
        # Risk Predictions (nur wenn nicht FLAT)
        if entry_signal != 0:
            # ML-Model sagt Risiko-% (1-3%) vorher
            risk_pct = self.risk_model_pct.predict(X)[0]
            risk_pct = np.clip(risk_pct, 1.0, 3.0)  # Begrenze auf 1-3%
            
            # ML-Model sagt SL/TP ATR-Multiplikatoren vorher
            sl_atr_mult = self.risk_model_sl.predict(X)[0]
            tp_atr_mult = self.risk_model_tp.predict(X)[0]
            
            # Clipping für Sicherheit
            sl_atr_mult = np.clip(sl_atr_mult, 0.5, 5.0)
            tp_atr_mult = np.clip(tp_atr_mult, 1.0, 10.0)
        else:
            risk_pct = 0
            sl_atr_mult = 0
            tp_atr_mult = 0
        
        return {
            "entry_signal": entry_signal,  # -1, 0, 1
            "entry_signal_name": {-1: "SHORT", 0: "FLAT", 1: "LONG"}[entry_signal],
            "confidence": float(confidence),
            "risk_pct": float(risk_pct),  # 1-3% vom Kapital
            "sl_atr_multiplier": float(sl_atr_mult),
            "tp_atr_multiplier": float(tp_atr_mult),
            "entry_proba": {
                "SHORT": float(entry_proba[0]),
                "FLAT": float(entry_proba[1]),
                "LONG": float(entry_proba[2])
            }
        }


if __name__ == "__main__":
    # Test Model Training
    print("Dieser Test erfordert Feature-Matrix mit Labels.")
    print("Rufe zuerst data_loader.py und feature_engineering.py auf.")
