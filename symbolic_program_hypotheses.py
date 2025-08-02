"""
SpectraMind V50 – Symbolic Program Hypotheses (Ultimate Version)
-----------------------------------------------------------------
Evaluates multiple symbolic rule programs on predicted μ spectra.
Supports sorting, matrix mode, top-k filtering, and full violation metadata.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union


class SymbolicProgramHypotheses:
    def __init__(self, logic_engines: List, names: Optional[List[str]] = None):
        """
        Args:
            logic_engines: list of symbolic logic engine instances (.apply(mu, metadata) → Dict[str, float])
            names: optional list of program names (must match length of logic_engines)
        """
        self.engines = logic_engines
        self.names = names or [f"program_{i}" for i in range(len(logic_engines))]
        assert len(self.names) == len(self.engines), "Length of names must match engines"

    def evaluate_all(
        self,
        mu: torch.Tensor,
        metadata: Dict,
        return_matrix: bool = False,
        sort_key: str = "total_loss",
        top_k: Optional[int] = None,
        include_metadata: bool = True,
        round_digits: int = 6
    ) -> Union[Dict[str, Dict], Dict[str, Dict[str, float]]]:
        """
        Evaluates all symbolic programs on input μ and returns full diagnostics.

        Args:
            mu (Tensor): (283,) – predicted spectrum
            metadata (Dict): planet/star context
            return_matrix (bool): if True, returns rule → program matrix instead of program list
            sort_key (str): metric to sort by (e.g., 'total_loss', 'max_rule_score')
            top_k (int): optionally limit to top-k programs
            include_metadata (bool): whether to include count and dominant rule info
            round_digits (int): decimal precision for all outputs

        Returns:
            Dict[str, Dict]: ranked program results or rule matrix
        """
        program_scores = {}
        rule_matrix = {}

        for engine, name in zip(self.engines, self.names):
            try:
                loss_dict = engine.apply(mu, metadata)
                if not isinstance(loss_dict, dict):
                    raise ValueError(f"Engine '{name}' returned non-dict")
            except Exception as e:
                print(f"❌ Error in symbolic engine '{name}': {e}")
                loss_dict = {}

            clean_losses = {
                k: float(v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in loss_dict.items()
                if isinstance(v, (int, float, np.float32, np.float64, torch.Tensor)) and not np.isnan(v)
            }

            total_loss = round(sum(clean_losses.values()), round_digits)
            entry = {
                "total_loss": total_loss,
                "violations": {k: round(v, round_digits) for k, v in clean_losses.items()}
            }

            if include_metadata:
                entry["rule_count"] = len(clean_losses)
                if clean_losses:
                    dom_rule = max(clean_losses.items(), key=lambda x: x[1])
                    entry["max_rule"] = dom_rule[0]
                    entry["max_rule_score"] = round(dom_rule[1], round_digits)
                else:
                    entry.update({"max_rule": None, "max_rule_score": None})

            program_scores[name] = entry

            for rule, score in clean_losses.items():
                rule_matrix.setdefault(rule, {})[name] = round(score, round_digits)

        if return_matrix:
            return rule_matrix

        if sort_key in {"total_loss", "rule_count", "max_rule_score"}:
            program_scores = dict(sorted(
                program_scores.items(),
                key=lambda x: x[1].get(sort_key, 0.0),
                reverse=True
            ))

        if top_k:
            program_scores = dict(list(program_scores.items())[:top_k])

        return program_scores