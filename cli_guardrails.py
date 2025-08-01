"""
SpectraMind V50 – CLI Guardrails and UX Safety Module
------------------------------------------------------
Provides --confirm, --dry-run, and debug logging wrappers for critical commands.
"""

import typer
import datetime
import inspect
import traceback
from functools import wraps
from pathlib import Path

LOG_PATH = Path("v50_debug_log.md")


def guardrails_wrapper(command_name: str, args: dict, dry_run: bool = False, confirm: bool = True, fail_on_missing_args: bool = False):
    if dry_run:
        for k, v in args.items():
            typer.echo(f"[DRY RUN] 💡 Would perform: {command_name}({k}={v})")
        _log_cli_call(command_name, args, dry_run=True)
        raise typer.Exit()

    if confirm:
        if not typer.confirm(f"⚠️  This will execute: {command_name} with args {args}. Proceed?"):
            typer.echo("❌ Aborted by user.")
            raise typer.Abort()

    if fail_on_missing_args:
        missing = [k for k, v in args.items() if v in (None, '', [])]
        if missing:
            typer.echo(f"[WARN] Missing arguments: {missing}")
            raise typer.Exit(code=2)

    _log_cli_call(command_name, args)
    typer.echo(f"✅ Starting {command_name}...")


def guarded_command(confirm=True, dry_run_arg="dry_run", confirm_arg="confirm"):
    """
    Decorator for Typer commands to auto-wrap with guardrails logic.

    Usage:
        @app.command()
        @guarded_command()
        def run_task(arg1: str, dry_run: bool = False, confirm: bool = True): ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            frame = inspect.currentframe().f_back
            caller_file = Path(frame.f_globals.get("__file__", "")).name
            command_name = f"{caller_file}::{func.__name__}"

            dry_run_val = kwargs.get(dry_run_arg, False)
            confirm_val = kwargs.get(confirm_arg, True)
            guardrails_wrapper(command_name, kwargs, dry_run=dry_run_val, confirm=confirm_val)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _log_cli_error(command_name, kwargs, e)
                raise e
        return wrapper
    return decorator


def _log_cli_call(command: str, args: dict, logfile: Path = LOG_PATH, dry_run: bool = False):
    timestamp = datetime.datetime.now().isoformat()
    level = "[DRY RUN]" if dry_run else "[INFO]"
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"\n{level} [{timestamp}] CLI Command: {command}\n")
        for k, v in args.items():
            f.write(f"    {k}: {v}\n")


def _log_cli_error(command: str, args: dict, exception: Exception, logfile: Path = LOG_PATH):
    timestamp = datetime.datetime.now().isoformat()
    tb = traceback.format_exc()
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"\n[ERROR] [{timestamp}] CLI Command Failed: {command}\n")
        for k, v in args.items():
            f.write(f"    {k}: {v}\n")
        f.write(f"\nException:\n{str(exception)}\n")
        f.write(f"Traceback:\n{tb}\n")
