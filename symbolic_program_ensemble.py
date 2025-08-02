"""
SpectraMind V50 – Symbolic Program Ensemble
-------------------------------------------
Combines multiple symbolic rule sets into a logic ensemble for voting, averaging,
priority override, or weighted blending during constraint evaluation.
"""

import json
from typing import Dict, Callable, List, Union, Optional


class SymbolicProgramEnsemble:
    def __init__(
        self,
        rule_modules: List[Callable],
        mode: str = "vote",
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            rule_modules: List of callables, each (mu, metadata) → Dict[str, float]
            mode: 'vote', 'avg', 'priority', 'weighted'
            weights: Only used for 'weighted' mode
        """
        self.rule_modules = rule_modules
        self.mode = mode
        self.weights = weights

        if self.mode == "weighted":
            if not weights:
                raise ValueError("Weights required for weighted mode.")
            if len(weights) != len(rule_modules):
                raise ValueError("Weights length must match number of rule modules.")

        if self.mode not in {"vote", "avg", "priority", "weighted"}:
            raise ValueError(f"Invalid mode: {self.mode}")

    def evaluate(
        self,
        mu,
        metadata: Dict,
        return_full: bool = False
    ) -> Union[Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
        """
        Evaluate all rule modules and return fused symbolic violations.

        Args:
            mu: np.ndarray or Tensor (283,) – predicted μ spectrum
            metadata: dict – auxiliary metadata (e.g., star type, temperature)
            return_full: if True, return both fused result and raw module outputs

        Returns:
            Dict of blended rule violations, or full dict if return_full=True
        """
        raw_outputs = []
        for mod in self.rule_modules:
            try:
                out = mod(mu, metadata)
                if not isinstance(out, dict):
                    raise TypeError("Symbolic rule module must return a dict[str -> float]")
                raw_outputs.append(out)
            except Exception as e:
                print(f"❌ Rule module exception: {e}")
                raw_outputs.append({})  # Fallback

        if self.mode == "vote":
            fused = self._vote(raw_outputs)
        elif self.mode == "avg":
            fused = self._average(raw_outputs)
        elif self.mode == "priority":
            fused = self._priority(raw_outputs)
        elif self.mode == "weighted":
            fused = self._weighted(raw_outputs)
        else:
            raise RuntimeError("Invalid mode")

        if return_full:
            return {
                "blended": fused,
                "raw": {f"module_{i}": ro for i, ro in enumerate(raw_outputs)}
            }
        return fused

    def _vote(self, outputs: List[Dict[str, float]]) -> Dict[str, float]:
        vote_counts = {}
        rule_total = {}

        for out in outputs:
            for rule, val in out.items():
                if val > 0.5:
                    vote_counts[rule] = vote_counts.get(rule, 0) + 1
                rule_total[rule] = rule_total.get(rule, 0) + 1

        return {
            rule: vote_counts.get(rule, 0) / rule_total[rule]
            for rule in rule_total
        }

    def _average(self, outputs: List[Dict[str, float]]) -> Dict[str, float]:
        merged = {}
        counts = {}
        for out in outputs:
            for rule, val in out.items():
                merged[rule] = merged.get(rule, 0.0) + val
                counts[rule] = counts.get(rule, 0) + 1
        return {
            rule: merged[rule] / counts[rule]
            for rule in merged
        }

    def _priority(self, outputs: List[Dict[str, float]]) -> Dict[str, float]:
        for out in outputs:
            if out:
                return out
        return {}

    def _weighted(self, outputs: List[Dict[str, float]]) -> Dict[str, float]:
        weighted_sum = {}
        weight_sum = {}

        for i, out in enumerate(outputs):
            w = self.weights[i]
            for rule, val in out.items():
                weighted_sum[rule] = weighted_sum.get(rule, 0.0) + w * val
                weight_sum[rule] = weight_sum.get(rule, 0.0) + w

        return {
            rule: weighted_sum[rule] / weight_sum[rule]
            for rule in weighted_sum
        }


if __name__ == "__main__":
    # Dummy rule modules for testing
    def rule_photonic(mu, meta):
        return {"photonic_band": 0.7, "smoothness": 0.3}

    def rule_thermo(mu, meta):
        return {"photonic_band": 0.4, "asymmetry": 0.9}

    def rule_shape(mu, meta):
        return {"asymmetry": 0.8, "monotonicity": 0.2}

    dummy_mu = [0.0] * 283
    dummy_meta = {"star_type": "G", "planet_radius": 1.2}

    print("🔍 Symbolic Ensemble Fusion Demo")
    for mode in ["vote", "avg", "priority", "weighted"]:
        print(f"\n🔧 Mode = {mode.upper()}")
        try:
            ensemble = SymbolicProgramEnsemble(
                rule_modules=[rule_photonic, rule_thermo, rule_shape],
                mode=mode,
                weights=[0.5, 0.3, 0.2] if mode == "weighted" else None
            )
            result = ensemble.evaluate(dummy_mu, dummy_meta, return_full=True)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"❌ {mode} failed: {e}")