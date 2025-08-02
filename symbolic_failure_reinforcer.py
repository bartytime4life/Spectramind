"""
SpectraMind V50 – Symbolic Failure Reinforcer (Ultimate Version)
-----------------------------------------------------------------
Amplifies symbolic loss penalties in spectral bins with historically frequent violations.
Integrates with:
- symbolic_loss.py (applies weighting to symbolic vector)
- generate_diagnostic_summary.py (violation log generator)
- constraint_violation_log.json (input mask source)
- anomaly_feedback_trainer.py (to emphasize retraining on unstable bins)
- train_v50.py (used in symbolic loss path)
- symbolic_rule_scorer.py (rule strength modulator)
"""

import numpy as np
import torch
import os
import json

class SymbolicFailureReinforcer:
    def __init__(self,
                 violation_log_path: str = "constraint_violation_log.json",
                 boost: float = 2.0,
                 threshold: float = 0.3,
                 fallback_weight: float = 1.0):
        """
        Args:
            violation_log_path: JSON file with per-bin violation statistics
            boost: multiplicative factor for failure-prone bins
            threshold: mean violation rate above which boost applies
            fallback_weight: default value for all bins if file is missing
        """
        self.weights = torch.ones(283) * fallback_weight
        self.threshold = threshold
        self.boost = boost
        self._load_violations(violation_log_path)

    def _load_violations(self, path: str):
        if not os.path.exists(path):
            print("⚠️ No violation log found. Using uniform symbolic weights.")
            return
        with open(path, 'r') as f:
            data = json.load(f)
        mean_violation = np.array(data.get("mean_violation", [0.0] * 283))
        boosted = mean_violation > self.threshold
        self.weights[boosted] = self.boost
        print(f"✅ Symbolic weight boost applied to {boosted.sum()} bins.")

    def apply(self, symbolic_loss_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            symbolic_loss_vector: (B, 283) tensor
        Returns:
            Weighted scalar symbolic loss
        """
        device = symbolic_loss_vector.device
        weight_vec = self.weights.to(device)
        weighted = symbolic_loss_vector * weight_vec  # broadcasted
        return weighted.mean()


if __name__ == "__main__":
    dummy_loss = torch.rand(8, 283)
    reinforcer = SymbolicFailureReinforcer(boost=3.0, threshold=0.25)
    loss = reinforcer.apply(dummy_loss)
    print(f"Weighted symbolic loss: {loss.item():.4f}")