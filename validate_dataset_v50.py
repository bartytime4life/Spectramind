"""
SpectraMind V50 – Dataset Validator
-----------------------------------
Checks that each planet has correctly shaped FGS, AIRS, and label .npy or .npz files.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rich.console import Console
from rich.table import Table
import hashlib

console = Console()

REQUIRED_SUFFIXES = {
    "fgs": (1, 32, 32),
    "airs": (1, 32, 356),
    "label": (567,)
}

def hash_file(path: str) -> str:
    """Compute SHA-256 hash of a file"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_file(root, pid, key, expected_shape, allow_npz=False, hash_check=False):
    path_npy = os.path.join(root, f"{pid}_{key}.npy")
    path_npz = os.path.join(root, f"{pid}_{key}.npz")

    path = path_npy if os.path.exists(path_npy) else (path_npz if allow_npz and os.path.exists(path_npz) else None)
    if path is None:
        return f"❌ Missing: {path_npy if not allow_npz else f'{path_npy} or {path_npz}'}"

    try:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr['arr_0'] if 'arr_0' in arr else next(iter(arr.values()))
    except Exception as e:
        return f"⚠️ Failed to load {path}: {e}"

    if arr.shape != expected_shape:
        return f"❌ Shape mismatch: {path} → {arr.shape}, expected {expected_shape}"

    if hash_check:
        sha = hash_file(path)
        return f"ℹ️ SHA256 {os.path.basename(path)} = {sha}"

    return None

def validate_dataset(data_dir, csv_file, skip_label=False, verbose=False, allow_npz=False, log_errors=False, hash_check=False):
    df = pd.read_csv(csv_file)
    planet_ids = df["planet_id"].tolist()
    console.print(f"🛰️ Validating [cyan]{len(planet_ids)}[/] planet IDs in [bold]{data_dir}[/]")

    errors = []
    checked = 0

    for pid in tqdm(planet_ids, desc="🔎 Validating", ncols=80):
        for suffix, shape in REQUIRED_SUFFIXES.items():
            if skip_label and suffix == "label":
                continue
            err = check_file(data_dir, pid, suffix, shape, allow_npz=allow_npz, hash_check=hash_check)
            if err:
                errors.append({"planet_id": pid, "file": f"{suffix}", "error": err})
                if verbose:
                    console.print(f"[red]{err}[/]")
            elif verbose:
                console.print(f"[green]✅ {pid}_{suffix}.npy passed[/]")
            checked += 1

    if errors:
        console.print(f"\n[bold red]❌ {len(errors)} issues found out of {checked} checks.[/]")
        if log_errors:
            err_df = pd.DataFrame(errors)
            out_path = os.path.join(data_dir, "validation_errors.csv")
            err_df.to_csv(out_path, index=False)
            console.print(f"[yellow]📄 Logged to: {out_path}[/]")

        with open("v50_debug_log.md", "a") as f:
            f.write(f"\n[DATA VALIDATOR] {len(errors)} errors logged at {Path(csv_file).name}\n")

        table = Table(title="Validation Failures", show_lines=True)
        table.add_column("Planet", style="bold")
        table.add_column("File")
        table.add_column("Issue")
        for row in errors[:15]:  # Truncate
            table.add_row(row["planet_id"], row["file"], row["error"])
        console.print(table)
    else:
        console.print(f"\n[bold green]✅ All {checked} files validated successfully for {len(planet_ids)} planets.[/]")
    return len(errors) == 0

# CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="🧪 Validate SpectraMind V50 AIRS/FGS/label .npy/.npz files")
    parser.add_argument("--data_dir", default="./data/train", help="Directory with .npy or .npz files")
    parser.add_argument("--csv", default="./data/train.csv", help="CSV with 'planet_id' column")
    parser.add_argument("--skip_label", action="store_true", help="Skip label validation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output per file")
    parser.add_argument("--allow_npz", action="store_true", help="Allow .npz fallback")
    parser.add_argument("--log_errors", action="store_true", help="Write validation_errors.csv")
    parser.add_argument("--hash_check", action="store_true", help="Print SHA256 for each file")
    args = parser.parse_args()

    success = validate_dataset(
        data_dir=args.data_dir,
        csv_file=args.csv,
        skip_label=args.skip_label,
        verbose=args.verbose,
        allow_npz=args.allow_npz,
        log_errors=args.log_errors,
        hash_check=args.hash_check
    )
    exit(0 if success else 1)
