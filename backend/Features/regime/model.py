import json
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


REGIME_LABELS = ["trending", "mean_reverting", "high_volatility", "low_volatility", "news_driven"]


@dataclass
class RegimeOutput:
    regime: str
    confidence: dict
    state_id: int
    features_snapshot: dict = field(default_factory=dict)


class RegimeCNN(nn.Module):
    """1D-CNN fallback for hard-to-separate regimes."""

    def __init__(self, n_features: int = 8, n_classes: int = 5, seq_len: int = 30):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(128 * 8, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class RegimeDetectionModel:
    """
    Primary: Gaussian HMM (5 states).
    Fallback: 1D-CNN when HMM confidence is low (<0.60).
    """

    def __init__(self, n_components: int = 5, lookback: int = 60):
        self.n_components = n_components
        self.lookback = lookback
        self.hmm: Optional[GaussianHMM] = None
        self.cnn: Optional[RegimeCNN] = None
        self.scaler = StandardScaler()
        self.state_map: dict[int, str] = {}
        self.version: str = "v0"

    # ─── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> None:
        """
        X: raw feature matrix, shape (n_samples, n_features).
        Training is done on a chronological split — caller must pass only
        the training portion (first 80% by time).
        """
        X_scaled = self.scaler.fit_transform(X)

        self.hmm = GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=42,
        )
        self.hmm.fit(X_scaled)

        # Label states by their statistical characteristics
        state_sequences = self.hmm.predict(X_scaled)
        self._label_states(X_scaled, state_sequences)

    def _label_states(self, X: np.ndarray, states: np.ndarray) -> None:
        """
        Map latent HMM state IDs to named regimes using state statistics.
        Feature indices: 0=realised_vol, 2=ret_zscore, 5=hurst, 6=spread_proxy, 7=sentiment_var
        """
        state_stats = {}
        for s in range(self.n_components):
            mask = states == s
            if mask.sum() == 0:
                continue
            state_X = X[mask]
            state_stats[s] = {
                "mean_vol": float(state_X[:, 0].mean()),
                "mean_ret_z": float(np.abs(state_X[:, 2]).mean()),
                "mean_hurst": float(state_X[:, 5].mean()),
                "mean_spread": float(state_X[:, 6].mean()),
                "mean_sentiment_var": float(state_X[:, 7].mean()),
            }

        # Rank states and assign labels
        sorted_by_vol = sorted(state_stats.items(), key=lambda x: x[1]["mean_vol"])
        sorted_by_hurst = sorted(state_stats.items(), key=lambda x: x[1]["mean_hurst"], reverse=True)
        sorted_by_sentiment = sorted(state_stats.items(), key=lambda x: x[1]["mean_sentiment_var"], reverse=True)

        assigned: set[int] = set()
        label_map: dict[int, str] = {}

        # High volatility = highest vol state
        hv_state = sorted_by_vol[-1][0]
        label_map[hv_state] = "high_volatility"
        assigned.add(hv_state)

        # Low volatility = lowest vol state
        lv_state = sorted_by_vol[0][0]
        label_map[lv_state] = "low_volatility"
        assigned.add(lv_state)

        # Trending = highest Hurst exponent among remaining
        for s, _ in sorted_by_hurst:
            if s not in assigned:
                label_map[s] = "trending"
                assigned.add(s)
                break

        # News-driven = highest sentiment variance among remaining
        for s, _ in sorted_by_sentiment:
            if s not in assigned:
                label_map[s] = "news_driven"
                assigned.add(s)
                break

        # Mean-reverting = remaining
        for s in state_stats:
            if s not in assigned:
                label_map[s] = "mean_reverting"

        self.state_map = label_map

    # ─── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> RegimeOutput:
        """
        Online inference using the last `lookback` rows of X.
        X: recent feature matrix, shape (n_recent, n_features).
        """
        if self.hmm is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        window = X[-self.lookback:] if len(X) >= self.lookback else X
        X_scaled = self.scaler.transform(window)

        state_sequence = self.hmm.predict(X_scaled)
        current_state = int(state_sequence[-1])

        # Posterior probabilities for current bar
        log_posteriors = self.hmm.predict_proba(X_scaled)
        probs = log_posteriors[-1]

        confidence = {
            self.state_map.get(i, f"state_{i}"): float(probs[i])
            for i in range(self.n_components)
        }
        max_conf = float(probs[current_state])
        regime = self.state_map.get(current_state, "mean_reverting")

        # Fallback to CNN if HMM is uncertain
        if max_conf < 0.60 and self.cnn is not None:
            regime = self._cnn_predict(X_scaled)

        return RegimeOutput(
            regime=regime,
            confidence=confidence,
            state_id=current_state,
        )

    def _cnn_predict(self, X_scaled: np.ndarray) -> str:
        self.cnn.eval()
        seq = torch.FloatTensor(X_scaled[-30:]).unsqueeze(0)
        with torch.no_grad():
            logits = self.cnn(seq)
            pred = int(torch.argmax(logits, dim=1).item())
        return REGIME_LABELS[pred]

    # ─── PSI Drift Detection ───────────────────────────────────────────────────

    def compute_psi(self, X_reference: np.ndarray, X_current: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index. PSI > 0.25 triggers retrain."""
        psi_total = 0.0
        for col in range(X_reference.shape[1]):
            ref = X_reference[:, col]
            cur = X_current[:, col]
            breaks = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
            breaks[0] -= 1e-8
            breaks[-1] += 1e-8

            ref_pct = np.histogram(ref, bins=breaks)[0] / len(ref)
            cur_pct = np.histogram(cur, bins=breaks)[0] / len(cur)

            ref_pct = np.clip(ref_pct, 1e-8, None)
            cur_pct = np.clip(cur_pct, 1e-8, None)

            psi_total += np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi_total / X_reference.shape[1])

    # ─── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        payload = {
            "hmm": self.hmm,
            "scaler": self.scaler,
            "state_map": self.state_map,
            "version": self.version,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.hmm = payload["hmm"]
        self.scaler = payload["scaler"]
        self.state_map = payload["state_map"]
        self.version = payload["version"]
