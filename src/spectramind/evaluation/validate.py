import pandas as pd
from rich import print

def main(sub_csv: str = "outputs/submission.csv"):
    df = pd.read_csv(sub_csv)
    assert df.shape[1] == 1 + 283 + 283, "submission must have 567 columns"
    print("✅ submission shape valid")
    print(df.head(1).to_string(index=False))

if __name__ == "__main__":
    main()
