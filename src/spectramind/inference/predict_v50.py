import argparse, pandas as pd, numpy as np, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config_v50.yaml")
args = parser.parse_args()
pathlib.Path("outputs").mkdir(exist_ok=True, parents=True)
planets = ["PL_0001","PL_0002"]
mu = np.random.randn(len(planets), 283)
sigma = np.abs(np.random.randn(len(planets), 283)) + 0.1
cols = ["planet_id"] + [f"mu_{i}" for i in range(283)] + [f"sigma_{i}" for i in range(283)]
import pandas as pd
df = pd.DataFrame(columns=cols)
df["planet_id"] = planets
for i in range(283):
    df[f"mu_{i}"] = mu[:, i]
for i in range(283):
    df[f"sigma_{i}"] = sigma[:, i]
df.to_csv("outputs/submission.csv", index=False)
print("🔮 Dummy prediction written to outputs/submission.csv")
