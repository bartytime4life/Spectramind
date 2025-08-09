import json, random, hashlib, yaml
from pathlib import Path
from datetime import datetime

OUTDIR = Path("outputs/ablations"); OUTDIR.mkdir(parents=True, exist_ok=True)
def hash_config(cfg): return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def mutate_config(base):
    cfg = json.loads(json.dumps(base))
    seed = random.randint(0, 99999)
    cfg['seed'] = seed; cfg['experiment_id'] = f"ablation_{seed}"
    cfg['training']['lr'] = 10 ** random.uniform(-5.5, -3.0)
    return cfg

if __name__ == "__main__":
    base_cfg_path = Path("configs/config_v50.yaml")
    base = yaml.safe_load(open(base_cfg_path))
    trial_cfg = mutate_config(base)
    out = OUTDIR / f"{trial_cfg['experiment_id']}.yaml"
    yaml.safe_dump(trial_cfg, open(out, "w"))
    print(f"Saved trial config: {out}")
