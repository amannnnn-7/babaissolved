#!/usr/bin/env bash
# Clone vendor/baba-is-auto and apply our pybind11 patches.
# Idempotent: safe to run multiple times.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/baba-is-auto"
PATCH="$ROOT/scripts/patches/baba-is-auto-bindings.patch"

if [[ -d "$VENDOR/.git" ]]; then
    echo "✔ vendor/baba-is-auto already cloned"
else
    echo "→ cloning vendor/baba-is-auto"
    git clone --recursive https://github.com/utilForever/baba-is-auto "$VENDOR"
fi

cd "$VENDOR"

# Apply patch only if it hasn't been applied already.
if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "✔ pybind11 patch already applied"
else
    echo "→ applying pybind11 patch"
    git apply "$PATCH"
fi

echo "✔ vendor setup complete"
