import numpy as np, pandas as pd, json
from pathlib import Path

TOTAL_BINS = 567
FGS_BINS = 1
AIRS_BINS = TOTAL_BINS - FGS_BINS
FGS_WEIGHT = 0.4
AIRS_WEIGHT = 0.6 / AIRS_BINS
SPECTRAL_WEIGHTS = np.array([FGS_WEIGHT] + [AIRS_WEIGHT] * AIRS_BINS, dtype=np.float32)

def compute_gll_matrix(y_true, mu_pred, sigma_pred, weights=SPECTRAL_WEIGHTS):
    sigma_pred = np.clip(sigma_pred, 1e-8, 1e6)
    squared_term = ((y_true - mu_pred) / sigma_pred) ** 2
    log_term = 2.0 * np.log(sigma_pred) + np.log(2 * np.pi)
    gll = squared_term + log_term
    return gll * weights

def evaluate_gll(labels_df: pd.DataFrame, preds_df: pd.DataFrame) -> float:
    labels_df = labels_df.set_index("planet_id")
    preds_df = preds_df.set_index("planet_id")
    mu_cols = [f"mu_{i}" for i in range(TOTAL_BINS)]
    sigma_cols = [f"sigma_{i}" for i in range(TOTAL_BINS)]
    y_true = labels_df[mu_cols].values.astype(np.float32)
    mu_pred = preds_df[mu_cols].values.astype(np.float32)
    sigma_pred = preds_df[sigma_cols].values.astype(np.float32)
    gll_matrix = compute_gll_matrix(y_true, mu_pred, sigma_pred)
    return float(np.mean(np.sum(gll_matrix, axis=1)))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--tag", default="submission")
    args = p.parse_args()
    labels_df = pd.read_csv(args.labels); preds_df = pd.read_csv(args.preds)
    score = evaluate_gll(labels_df, preds_df)
    print(f"GLL Score: {score:.6f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"score_name":"GLL","score_value":score,"tag":args.tag}, f, indent=2)
