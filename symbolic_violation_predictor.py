"""
SpectraMind V50 – Symbolic Violation Predictor (Ultimate Version)
-----------------------------------------------------------------
Evaluates predicted μ spectra against symbolic rule programs.
Supports per-planet scoring, program ranking, per-rule overlays, and CI-safe output.
"""

import torch
from typing import List, Dict, Optional, Union
from symbolic_program_hypotheses import SymbolicProgramHypotheses


class SymbolicViolationPredictor:
    def __init__(self, engines: List, names: Optional[List[str]] = None):
        """
        Args:
            engines: List of symbolic logic engine instances (.apply(mu, metadata))
            names: Optional list of engine names
        """
        self.hypotheses = SymbolicProgramHypotheses(engines, names)

    def predict(
        self,
        mu_batch: torch.Tensor,
        metadata_batch: List[Dict],
        *,
        best_only: bool = True,
        top_k: Optional[int] = None,
        sort_key: str = "total_loss",
        return_matrix: bool = False,
        return_trace: bool = False,
        round_digits: int = 6
    ) -> Union[List[Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Evaluates symbolic rule programs over a batch of μ predictions.

        Args:
            mu_batch: Tensor (N, 283)
            metadata_batch: list of length N with planet/star metadata
            best_only: if True, use only top-1 ranked symbolic program
            top_k: use top-k ensemble programs (if best_only=False)
            sort_key: ranking key (e.g., 'total_loss', 'max_rule_score')
            return_matrix: if True, returns rule → planet matrix instead of per-planet list
            return_trace: if True, return raw ranking info per planet
            round_digits: decimal precision

        Returns:
            List of violation dicts per planet OR rule matrix OR full trace
        """
        assert mu_batch.shape[0] == len(metadata_batch), "Mismatch between μ and metadata"

        outputs = []
        rule_matrix = {}

        for i, (mu, meta) in enumerate(zip(mu_batch, metadata_batch)):
            if return_matrix:
                rule_matrix[f"planet_{i}"] = self.hypotheses.evaluate_all(
                    mu,
                    meta,
                    return_matrix=True,
                    sort_key=sort_key,
                    top_k=top_k,
                    round_digits=round_digits
                )
            elif return_trace:
                trace = self.hypotheses.evaluate_all(
                    mu,
                    meta,
                    return_matrix=False,
                    sort_key=sort_key,
                    top_k=top_k,
                    round_digits=round_digits
                )
                outputs.append(trace)
            else:
                best = self.hypotheses.evaluate_all(
                    mu,
                    meta,
                    return_matrix=False,
                    sort_key=sort_key,
                    top_k=1 if best_only else top_k,
                    round_digits=round_digits
                )
                top = next(iter(best.values()))
                outputs.append(top["violations"])

        return rule_matrix if return_matrix else outputs