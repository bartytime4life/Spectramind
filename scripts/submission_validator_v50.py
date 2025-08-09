import sys, pandas as pd
TOTAL=283
def main(path):
    df = pd.read_csv(path)
    mu = [c for c in df.columns if c.startswith("mu_")]
    sg = [c for c in df.columns if c.startswith("sigma_")]
    assert len(mu)==TOTAL and len(sg)==TOTAL, f"Expected {TOTAL} mu and {TOTAL} sigma columns"
    print("✅ submission.csv column count OK")
if __name__ == "__main__":
    main(sys.argv[1])
