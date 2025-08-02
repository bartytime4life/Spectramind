#!/bin/bash
# SpectraMind V50 – Unified CLI Smoke Test Suite
# Run with: bash test_cli_v50.sh

echo "🚀 Starting SpectraMind V50 CLI Self-Test"

commands=(
    "spectramind test run"
    "spectramind --version"
    "spectramind generate-dummy-data"
    "spectramind core train --config configs/config_v50.yaml"
    "spectramind core predict"
    "spectramind core score-gll"
    "spectramind core package"
    "spectramind submit make-submission"
    "spectramind diagnose dashboard --no-tsne"
    "spectramind analyze-log --limit 3"
)

for cmd in "${commands[@]}"; do
    echo -e "\n▶️ Testing: $cmd"
    eval "$cmd"
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed: $cmd"
    else
        echo "✅ Passed: $cmd"
    fi
done

echo -e "\n🧪 Test Suite Complete. Review log output and results above."