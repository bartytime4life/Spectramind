"""
SpectraMind V50 – Symbolic Influence Plot (Beyond Ultimate)
------------------------------------------------------------
Visualizes symbolic rule contributions across spectral bins with entropy overlay and dominant rule shading.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, Optional


def plot_symbolic_influence(
    influence_map: Dict[str, Union[np.ndarray, list]],
    show_entropy: bool = True,
    highlight_dominant: bool = True,
    top_k_rules: int = 5,
    figsize=(14, 5),
    title="Symbolic Rule Influence Across Spectrum"
):
    """
    Args:
        influence_map: Output of compute_influence_map(...) with keys:
            - per-rule (e.g., "smoothness", "photonic")
            - "composite_weighted", "entropy", "top_rule_name"
        show_entropy: Whether to overlay entropy as secondary signal
        highlight_dominant: Draw vertical lines for dominant rules per bin
        top_k_rules: Only show top-k rule curves by total magnitude
        figsize: plot size
        title: plot title

    Returns:
        matplotlib.figure.Figure
    """
    spectrum_len = len(influence_map["composite_weighted"])
    x = np.arange(spectrum_len)

    # Filter rules
    rule_keys = [k for k in influence_map.keys()
                 if k not in {"composite_max", "composite_sum", "composite_weighted",
                              "entropy", "top_rule", "top_rule_name", "entropy_mean",
                              "dominant_rules"}]

    rule_scores = {
        k: float(np.linalg.norm(np.array(influence_map[k]))) for k in rule_keys
    }
    top_rules = sorted(rule_scores.items(), key=lambda x: -x[1])[:top_k_rules]

    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot rule influence curves
    for rule, _ in top_rules:
        curve = np.array(influence_map[rule])
        ax1.plot(x, curve, label=rule, lw=2, alpha=0.85)

    ax1.set_xlabel("Spectral Bin")
    ax1.set_ylabel("Symbolic Influence")
    ax1.set_title(title)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right", fontsize="small")

    # Entropy overlay (secondary axis)
    if show_entropy and "entropy" in influence_map:
        ax2 = ax1.twinx()
        entropy = np.array(influence_map["entropy"])
        ax2.plot(x, entropy, color="gray", lw=1.5, linestyle="--", alpha=0.6, label="Entropy")
        ax2.set_ylabel("Entropy")
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left", fontsize="small")

    # Dominant rule lines
    if highlight_dominant and "top_rule_name" in influence_map:
        prev_rule = None
        rule_spans = []
        start = 0

        for i, r in enumerate(influence_map["top_rule_name"]):
            if r != prev_rule and prev_rule is not None:
                rule_spans.append((start, i - 1, prev_rule))
                start = i
            prev_rule = r
        rule_spans.append((start, spectrum_len - 1, prev_rule))

        for s, e, rule in rule_spans:
            ax1.axvspan(s, e, color="lightcoral", alpha=0.05, label=None)

    return fig