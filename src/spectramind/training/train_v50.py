import argparse, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config_v50.yaml")
args = parser.parse_args()
pathlib.Path("outputs").mkdir(exist_ok=True, parents=True)
print("🧪 Training placeholder finished.")
