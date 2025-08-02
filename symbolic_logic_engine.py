"""
SpectraMind V50 – Symbolic Logic Engine (Final Diagnostics Edition)
-------------------------------------------------------------------
Routes μ through symbolic rule programs (single or multi-program).
Supports scalar and vector constraint violations, rule weighting,
and returns diagnostics (entropy, dominant rule) if requested.
"""

import torch
from typing import Optional, Dict, Union
from symbolic_loss import compute_symbolic_losses
from symbolic_program_hypotheses import SymbolicProgramHypotheses


class SymbolicLogicEngine:
    def __init__(
        self,
        config: Union[dict, list],
        allow_multi: bool = False,
        return_vector: bool = False,
        rule_weights: Optional[Dict[str, float]] = None,
        return_diagnostics: bool = False
    ):
        """
        Args:
            config: symbolic config dict or list of them
            allow_multi: enable symbolic_program_hypotheses engine
            return_vector: if True, return binwise vectors instead of scalars
            rule_weights: optional dict to weight rule importance
            return_diagnostics: if True, return entropy + top_rule + stats
        """
        self.configs = config if isinstance(config, list) else [config]
        self.multi = allow_multi and len(self.configs) > 1
        self.return_vector = return_vector
        self.rule_weights = rule_weights or {}
        self.return_diagnostics = return_diagnostics

    def apply(self, mu: torch.Tensor, metadata: Optional[dict] = None) -> Union[dict, Dict[str, Union[dict, float]]]:
        """
        Evaluates symbolic rules on μ.

        Args:
            mu: (B, 283) or (283,)
            metadata: optional dict with planet info

        Returns:
            dict of rule_name → value (scalar or vector)
            OR dict with diagnostics if return_diagnostics=True
        """
        if mu.ndim == 2:
            mu = mu.mean(dim=0)

        if self.multi:
            engines = [
                SymbolicLogicEngine(cfg, allow_multi=False, return_vector=self.return_vector)
                for cfg in self.configs
            ]
            names = [f"program_{i}" for i in range(len(engines))]
            runner = SymbolicProgramHypotheses(engines, names)
            return runner.evaluate_all(mu, metadata, return_matrix=False, sort_key="total_loss")

        # Single program evaluation
        losses = compute_symbolic_losses(mu, self.configs[0], meta=metadata, log_raw=self.return_vector)

        # Weight if applicable
        if self.rule_weights:
            losses = {
                rule: float(losses[rule]) * self.rule_weights.get(rule, 1.0)
                for rule in losses
            }

        total_loss = sum(float(v.mean() if isinstance(v, torch.Tensor) else v) for v in losses.values())
        losses["total_loss"] = total_loss

        if not self.return_diagnostics:
            return losses

        # Diagnostics
        scores = {k: v.mean().item() if isinstance(v, torch.Tensor) else v for k, v in losses.items() if k != "total_loss"}
        rule_entropy = -sum(v * torch.log(torch.tensor(v + 1e-8)) for v in scores.values())
        top_rule = max(scores.items(), key=lambda x: x[1])[0]

        return {
            "losses": losses,
            "summary": {
                "entropy": float(rule_entropy),
                "top_rule": top_rule,
                "num_rules": len(scores),
                "total_loss": total_loss
            }
        }