import typer
app = typer.Typer(help="SpectraMind V50 CLI")
@app.command()
def version():
    print("SpectraMind V50 CLI v0.1.0")
if __name__ == "__main__":
    app()
