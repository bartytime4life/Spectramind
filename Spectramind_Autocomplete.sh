#!/bin/bash
# SpectraMind V50 – Autocomplete Installer (Bash + Zsh)
# ------------------------------------------------------
# Adds tab-completion for the `spectramind` CLI on bash/zsh shells
# Usage: bash install_autocomplete.sh

set -euo pipefail

BASH_SCRIPT="$HOME/.spectramind_autocomplete.sh"
ZSH_SCRIPT="$HOME/.spectramind_autocomplete_zsh"

# --- Bash Completion Script ---
cat <<EOF > "$BASH_SCRIPT"
#!/bin/bash
_spectramind_completions() {
  COMPREPLY=()
  local cur
  cur="