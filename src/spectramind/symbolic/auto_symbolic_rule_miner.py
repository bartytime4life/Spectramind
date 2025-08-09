# auto_symbolic_rule_miner.py
import numpy as np

def find_high_entropy_bins(mu, threshold=0.15):
    std = mu.std(axis=0)
    return np.where(std > threshold)[0].tolist()
