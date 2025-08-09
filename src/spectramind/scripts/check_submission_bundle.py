from pathlib import Path
import zipfile, sys

def main():
    zpath = Path("outputs/submission_bundle.zip")
    if not zpath.exists():
        print("Bundle missing.")
        sys.exit(1)
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
    need = {"submission.csv"}
    missing = need - names
    if missing:
        print(f"Missing in bundle: {missing}")
        sys.exit(2)
    print("✅ Bundle looks good.")
if __name__ == "__main__":
    main()
