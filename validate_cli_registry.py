"""
SpectraMind V50 – CLI Registry Validator (Extended)
----------------------------------------------------
Validates that all CLI registry modules exist AND contain expected Typer @app.command decorators.
"""

import os
import json
import re
import sys
from pathlib import Path

REGISTRY_PATH = "cli_command_registry.json"
SEARCH_DIR = "cli"
DEBUG_LOG = "v50_debug_log.md"

# Matches: @app.command(...) followed by def <func>(
COMMAND_REGEX = re.compile(r"@app\.command(?:\([^)]*\))?\s*\ndef\s+(\w+)\s*\(")


def flatten_modules(mod_entry):
    return mod_entry if isinstance(mod_entry, list) else [mod_entry]


def extract_commands_from_file(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"⚠️ Failed to read {filepath}: {e}")
        return set()
    return set(COMMAND_REGEX.findall(code))


def validate_registry():
    if not os.path.exists(REGISTRY_PATH):
        print(f"❌ Registry file not found: {REGISTRY_PATH}")
        sys.exit(1)

    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    missing_files = {}
    missing_commands = {}

    for cmd_name, mod_list in registry.items():
        for mod in flatten_modules(mod_list):
            found = False
            for root, _, files in os.walk(SEARCH_DIR):
                if mod in files:
                    path = os.path.join(root, mod)
                    found = True
                    declared = extract_commands_from_file(path)
                    if cmd_name not in declared and mod != "cli_v50.py":
                        missing_commands.setdefault(mod, []).append(cmd_name)
                    break
            if not found:
                missing_files.setdefault(cmd_name, []).append(mod)

    # Output
    passed = True

    if not missing_files:
        print("✅ All CLI registry files are present.")
    else:
        passed = False
        print("❌ Missing CLI module files:")
        for cmd, mods in missing_files.items():
            print(f"  ⛔ {cmd}: {mods}")

    if not missing_commands:
        print("✅ All CLI commands are declared via @app.command.")
    else:
        passed = False
        print("❌ Undeclared CLI commands in modules:")
        for mod, cmds in missing_commands.items():
            print(f"  ⛔ {mod}: {cmds}")

    # Optional log
    with open(DEBUG_LOG, "a") as f:
        f.write("\n[CLI REGISTRY VALIDATION]\n")
        f.write("Missing files:\n")
        for cmd, mods in missing_files.items():
            f.write(f"  {cmd}: {mods}\n")
        f.write("Missing commands:\n")
        for mod, cmds in missing_commands.items():
            f.write(f"  {mod}: {cmds}\n")

    if not passed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    validate_registry()