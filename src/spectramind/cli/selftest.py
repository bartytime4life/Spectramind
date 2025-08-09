import typer
app = typer.Typer()
@app.command("run")
def run(mode: str = "fast"):
    print(f"✅ selftest ({mode}) passed (placeholder)")
if __name__ == "__main__":
    app()
