import yaml, zipfile
from pathlib import Path

def main(config="configs/config_v50.yaml"):
    cfg = yaml.safe_load(Path(config).read_text())
    outdir = Path(cfg["paths"]["outputs_dir"])
    bundle = outdir / cfg["submit"]["bundle_name"]
    with zipfile.ZipFile(bundle, "w") as z:
        for fname in ["submission.csv", "model.pt"]:
            fpath = outdir/fname
            if fpath.exists():
                z.write(fpath, arcname=fname)
    print(f"✅ Bundle created: {bundle}")
if __name__ == "__main__":
    main()
