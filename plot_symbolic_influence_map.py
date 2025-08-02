"""
SpectraMind V50 – Symbolic Influence Plot (Final Hardened Version)
-------------------------------------------------------------------
Visualizes symbolic rule contributions across spectral bins with entropy and dominant rule overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, Optional


def plot_symbolic_influence(
    influence_map: Dict[str, Union[np.ndarray, list]],
    show_entropy: bool = True,
    highlight_dominant: bool = True,
    show_composite: bool = False,
    annotate_rules: bool = False,
    top_k_rules: int = 5,
    figsize=(14, 5),
    title="Symbolic Rule Influence Across Spectrum",
    print_summary: bool = True
):
    """
    Args:
        influence_map: Output from compute_influence_map(...) with keys:
            - per-rule keys (e.g., 'smoothness', 'photonic')
            - required keys: 'composite_weighted', 'entropy', 'top_rule_name'
        show_entropy: overlays entropy (right Y-axis)
        highlight_dominant: shows shaded spans by dominant rule
        show_composite: overlays composite influence line
        annotate_rules: text labels for rule spans
        top_k_rules: number of top rules to display
        figsize: matplotlib figure size
        title: plot title
        print_summary: whether to print entropy stats and rule coverage

    Returns:
        matplotlib.figure.Figure
    """
    spectrum_len = len(influence_map["composite_weighted"])
    x = np.arange(spectrum_len)

    rule_keys = [k for k in influence_map.keys()
                 if k not in {
                     "composite_max", "composite_sum", "composite_weighted",
                     "entropy", "top_rule", "top_rule_name", "entropy_mean",
                     "dominant_rules"
                 }]

    rule_scores = {
        k: float(np.linalg.norm(np.array(influence_map[k]))) for k in rule_keys
    }
    top_rules = sorted(rule_scores.items(), key=lambda x: -x[1])[:top_k_rules]

    fig, ax1 = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")

    # Plot per-rule curves
    for i, (rule, _) in enumerate(top_rules):
        curve = np.array(influence_map[rule])
        ax1.plot(x, curve, label=rule, lw=2, alpha=0.9, color=cmap(i % 10))

    # Composite line
    if show_composite:
        ax1.plot(
            x,
            np.array(influence_map["composite_weighted"]),
            color="black",
            linestyle=":",
            lw=2,
            label="Composite Weighted"
        )

    ax1.set_xlabel("Spectral Bin")
    ax1.set_ylabel("Symbolic Influence")
    ax1.set_title(title)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Secondary Y-axis for entropy
    if show_entropy and "entropy" in influence_map:
        entropy = np.array(influence_map["entropy"])
        ax2 = ax1.twinx()
        ax2.plot(x, entropy, color="gray", lw=1.5, linestyle="--", label="Entropy")
        ax2.set_ylabel("Entropy")
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left", fontsize="small")

    # Highlight dominant rule spans
    if highlight_dominant and "top_rule_name" in influence_map:
        spans = []
        labels = set()
        prev = influence_map["top_rule_name"][0]
        start = 0

        for i, r in enumerate(influence_map["top_rule_name"]):
            if r != prev:
                spans.append((start, i - 1, prev))
                start = i
            prev = r
        spans.append((start, spectrum_len - 1, prev))

        for s, e, rule in spans:
            color = cmap(rule_keys.index(rule) % 10) if rule in rule_keys else "lightcoral"
            label = rule if rule not in labels else None
            ax1.axvspan(s, e, color=color, alpha=0.05, label=label)
            if annotate_rules and label:
                ax1.text((s + e) // 2, ax1.get_ylim()[1] * 0.95, rule,
                         fontsize=8, ha='center', color=color)
            labels.add(rule)

    ax1.legend(loc="upper right", fontsize="small")

    if print_summary:
        print("🧠 Dominant Rule Summary:")
        print(influence_map.get("dominant_rules", {}))
        print(f"🧪 Mean entropy: {influence_map.get('entropy_mean', 'N/A'):.4f}")

    return fig