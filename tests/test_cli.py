from typer.testing import CliRunner
from src.spectramind.cli.cli_v50 import app
import os, pandas as pd

runner = CliRunner()

def test_train_and_predict():
    r = runner.invoke(app, ["train"])
    assert r.exit_code == 0
    r = runner.invoke(app, ["predict"])
    assert r.exit_code == 0
    assert os.path.exists("outputs/submission.csv")
    df = pd.read_csv("outputs/submission.csv")
    assert df.shape[1] == 567
