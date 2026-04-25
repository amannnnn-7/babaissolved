#!/usr/bin/env bash
# Build pyBaba (vendor/baba-is-auto) into the active uv venv.
#
# We use a conda-forge toolchain because the system may not have cmake/g++
# installed and many shared boxes (Codespaces, Colab, locked-down WSL) don't
# permit `sudo apt install`. Adjust CONDA_TOOLS if you have a different setup.

set -euo pipefail

CONDA_TOOLS="${CONDA_TOOLS:-$HOME/miniconda3/envs/babaenv/bin}"

# Make sure vendor/ is cloned + patched before we try to build it.
"$(dirname "$0")/setup_vendor.sh"

if [[ ! -x "$CONDA_TOOLS/cmake" ]]; then
    echo "ERROR: cmake not found at $CONDA_TOOLS/cmake" >&2
    echo "Create a conda env with the toolchain:" >&2
    echo "  conda create -y -n babaenv -c conda-forge --override-channels \\" >&2
    echo "      cmake make gxx_linux-64 gcc_linux-64 python=3.11 pybind11" >&2
    exit 1
fi

export PATH="$CONDA_TOOLS:$PATH"
export CC="$CONDA_TOOLS/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_TOOLS/x86_64-conda-linux-gnu-g++"

cd "$(dirname "$0")/.."

# setuptools must already be in the venv since we disable build isolation so
# the env-vars above propagate to CMake.
uv pip install -q setuptools wheel
uv pip install --reinstall --no-deps --no-build-isolation ./vendor/baba-is-auto

echo
echo "✔ pyBaba installed. Smoke test:"
uv run python -c "import pyBaba; g = pyBaba.Game('vendor/baba-is-auto/Resources/Maps/baba_is_you.txt'); print('numRules:', g.GetRuleManager().GetNumRules())"
