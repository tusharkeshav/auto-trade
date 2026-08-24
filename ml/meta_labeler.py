# ─────────────────────────────────────────────────────────────────
#  ml/meta_labeler.py
#  De Prado Two-Stage Machine Learning (XGBoost Meta-Labeling).
#
#  Mathematical & Architectural Basis (Marcos Lopez de Prado):
#    - Primary Model (UnifiedCrossScorer): Proposes candidate BUY signal.
#    - Secondary Model (XGBoostMetaLabeler): Evaluates macro regime and
#      technical feature vector to predict probability of profit (P_win).
#    - Regime Layer: Integrates 3-state GaussianHMM (Accumulation, Range, Crash).
#    - Sizing: Kelly-inspired sizing scalar based on P_win threshold.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed. XGBoostMetaLabeler will operate in fallback passthrough mode.")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_MODEL_TYPE = "GaussianHMM"
except ImportError:
    try:
        from sklearn.mixture import GaussianMixture
        HMM_MODEL_TYPE = "GaussianMixture"
    except ImportError:
        HMM_MODEL_TYPE = "None"


FEATURE_NAMES = [
    "hmm_state",
    "hmm_crash_score",
    "rs_slope_60d",
    "adx_14",
    "vix",
    "rsi_14",
    "atr_pct",
]


def add_hmm_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit 3-state GaussianHMM or GMM on [ret, range_pct] of master index.
    Adds 'hmm_state' (0=Safe/Bull, 1=Neutral/Range, 2=Crash/Dist) and 'hmm_crash_score',
    plus posterior state probabilities: 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2'.
    """
    df = df.copy()
    if "ret" not in df.columns:
        df["ret"] = df["close"].pct_change() * 100.0
    if "range_pct" not in df.columns:
        df["range_pct"] = ((df["high"] - df["low"]) / df["close"]) * 100.0

    df_clean = df.dropna(subset=["ret", "range_pct"])
    if len(df_clean) < 30 or HMM_MODEL_TYPE == "None":
        df["hmm_state"] = 0
        df["hmm_crash_score"] = 0.0
        df["hmm_prob_0"] = 1.0
        df["hmm_prob_1"] = 0.0
        df["hmm_prob_2"] = 0.0
        return df

    X = df_clean[["ret", "range_pct"]].values

    try:
        if HMM_MODEL_TYPE == "GaussianHMM":
            model = GaussianHMM(n_components=3, covariance_type="full", min_covar=1e-3, n_iter=500, random_state=42)
            model.fit(X)
            states = model.predict(X)
            probs = model.predict_proba(X)
        else:
            model = GaussianMixture(n_components=3, covariance_type="full", reg_covar=1e-3, max_iter=500, random_state=42)
            model.fit(X)
            states = model.predict(X)
            probs = model.predict_proba(X)
    except Exception as e:
        logger.warning(f"HMM fit failed ({e}), using default accumulation state")
        df["hmm_state"] = 0
        df["hmm_crash_score"] = 0.0
        df["hmm_prob_0"] = 1.0
        df["hmm_prob_1"] = 0.0
        df["hmm_prob_2"] = 0.0
        return df

    df_clean = df_clean.assign(raw_state=states)

    state_stats = []
    for s in range(3):
        sub = df_clean[df_clean["raw_state"] == s]
        m_ret = sub["ret"].mean() if len(sub) > 0 else 0.0
        m_vol = sub["range_pct"].mean() if len(sub) > 0 else 0.0
        crash_score = m_vol - (2.0 * m_ret)
        state_stats.append({"state": s, "crash_score": crash_score})

    state_stats_sorted = sorted(state_stats, key=lambda x: x["crash_score"])
    state_map = {st["state"]: i for i, st in enumerate(state_stats_sorted)}
    score_map = {st["state"]: st["crash_score"] for st in state_stats_sorted}

    # Find raw state indices corresponding to mapped states 0, 1, 2
    raw_0 = state_stats_sorted[0]["state"]
    raw_1 = state_stats_sorted[1]["state"]
    raw_2 = state_stats_sorted[2]["state"]

    df_clean = df_clean.assign(
        hmm_state=df_clean["raw_state"].map(state_map),
        hmm_crash_score=df_clean["raw_state"].map(score_map),
        hmm_prob_0=probs[:, raw_0],
        hmm_prob_1=probs[:, raw_1],
        hmm_prob_2=probs[:, raw_2],
    )

    df["hmm_state"] = df_clean["hmm_state"]
    df["hmm_crash_score"] = df_clean["hmm_crash_score"]
    df["hmm_prob_0"] = df_clean["hmm_prob_0"]
    df["hmm_prob_1"] = df_clean["hmm_prob_1"]
    df["hmm_prob_2"] = df_clean["hmm_prob_2"]

    df["hmm_state"] = df["hmm_state"].ffill().fillna(0).astype(int)
    df["hmm_crash_score"] = df["hmm_crash_score"].ffill().fillna(0.0)
    df["hmm_prob_0"] = df["hmm_prob_0"].ffill().fillna(1.0)
    df["hmm_prob_1"] = df["hmm_prob_1"].ffill().fillna(0.0)
    df["hmm_prob_2"] = df["hmm_prob_2"].ffill().fillna(0.0)

    return df


class XGBoostMetaLabeler:
    """
    De Prado secondary ML model. Evaluates primary scorer trade candidates.
    Predicts probability of win (P_win) and assigns sizing scalar.
    """

    def __init__(self, min_train_trades: int = 15):
        self.min_train_trades = min_train_trades
        self.model: Any = None
        self.is_fitted: bool = False
        self.feature_names = FEATURE_NAMES
        self._init_model()

    def _init_model(self):
        if not XGB_AVAILABLE:
            return
        self.model = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=50,
            learning_rate=0.05,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            eval_metric="logloss",
        )

    def fit(self, trades: List[Any]) -> bool:
        """
        Fit XGBoost classifier on historical trades with feature dicts.
        Returns True if newly fitted or updated, False if skipped.
        """
        if not XGB_AVAILABLE:
            return False

        valid_trades = [t for t in trades if hasattr(t, "features") and t.features and len(t.features) >= len(self.feature_names)]
        if len(valid_trades) < self.min_train_trades:
            return False

        X_rows = []
        y_rows = []
        for t in valid_trades:
            row = [float(t.features.get(f, 0.0)) for f in self.feature_names]
            X_rows.append(row)
            y_rows.append(1 if t.net_pnl > 0 else 0)

        X = np.array(X_rows)
        y = np.array(y_rows)

        # Require at least one winner and one loser to train binary classifier
        if len(np.unique(y)) < 2:
            return False

        try:
            self.model.fit(X, y)
            self.is_fitted = True
            return True
        except Exception as e:
            logger.warning(f"XGBoost fit failed: {e}")
            return False

    def predict_size_scalar(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict win probability and assign sizing scalar.
        Returns (scalar, win_prob).

        Scalar rules (De Prado):
          - P_win < 0.45:  0.0 (Veto / Block)
          - 0.45 <= P_win < 0.55: 0.5 (Half Conviction)
          - P_win >= 0.55: 1.0 (Full Conviction)
        """
        if not self.is_fitted or not XGB_AVAILABLE:
            return 1.0, 0.50

        try:
            row = np.array([[float(features.get(f, 0.0)) for f in self.feature_names]])
            prob = float(self.model.predict_proba(row)[0, 1])

            if prob < 0.45:
                scalar = 0.0
            elif prob < 0.55:
                scalar = 0.5
            else:
                scalar = 1.0

            return scalar, prob
        except Exception as e:
            logger.warning(f"XGBoost inference fault: {e}")
            return 1.0, 0.50
