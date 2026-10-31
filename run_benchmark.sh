#!/usr/bin/env bash
# CST435 reproducible benchmark runner for macOS.
# Creates .venv, installs benchmark dependencies, pre-caches MNIST,
# runs the exact benchmark suite, and regenerates both figures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

START_TIME="$(date +%s)"

echo "CST435 MNIST scalability benchmark"
echo "Repository: $SCRIPT_DIR"
echo ""

# 1. Locate Python 3.12.
PYTHON_BIN=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.12 was not found." >&2
    echo "Install it from https://www.python.org/downloads/ and rerun this script." >&2
    exit 1
fi

echo "[1/5] Python 3.12 found: $("$PYTHON_BIN" --version)"

# 2. Create or validate the virtual environment.
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if [ -x "$VENV_PYTHON" ]; then
    if ! "$VENV_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
        echo "ERROR: The existing .venv is not Python 3.12. Delete .venv and rerun this script." >&2
        exit 1
    fi
    echo "[2/5] Using existing .venv"
else
    echo "[2/5] Creating .venv"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# 3. Install benchmark dependencies.
echo "[3/5] Installing benchmark dependencies"
"$VENV_PYTHON" -m pip install --disable-pip-version-check --requirement "$SCRIPT_DIR/requirements-benchmark.txt"

# 4. Pre-cache MNIST so download time is outside the timed runs.
echo "[4/5] Pre-caching MNIST"
"$VENV_PYTHON" -c "from src.dataset import get_mnist_datasets; train, test = get_mnist_datasets(); print(f'MNIST ready: {len(train)} train / {len(test)} test')"

# 5. Run the exact benchmark and regenerate the figures.
echo "[5/5] Running benchmark (CNN, 1 epoch, batch 128, p=1 2 4 8)"
"$VENV_PYTHON" "$SCRIPT_DIR/src/benchmark.py" --model cnn --epochs 1 --batch-size 128 --process-list 1 2 4 8

echo "Regenerating figures"
"$VENV_PYTHON" "$SCRIPT_DIR/src/plot_scalability.py"

ARTIFACTS=(
    "$SCRIPT_DIR/docs/benchmark_results.md"
    "$SCRIPT_DIR/figures/speedup_curve.png"
    "$SCRIPT_DIR/figures/scalability_visualization.png"
)

for artifact in "${ARTIFACTS[@]}"; do
    if [ ! -s "$artifact" ]; then
        echo "ERROR: missing or empty artifact: $artifact" >&2
        exit 1
    fi
done

END_TIME="$(date +%s)"
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Benchmark complete."
echo "System: $(uname -srm)"
if command -v sysctl >/dev/null 2>&1; then
    CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    PHYSICAL="$(sysctl -n hw.physicalcpu 2>/dev/null || true)"
    LOGICAL="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
    if [ -n "$CPU" ]; then
        echo "CPU: $CPU"
    fi
    if [ -n "$PHYSICAL" ]; then
        echo "Cores: $PHYSICAL physical / ${LOGICAL:-unknown} logical"
    fi
fi
echo "Python: $("$VENV_PYTHON" --version)"
echo "Elapsed: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "Results: ${ARTIFACTS[0]}"
echo "Figures: ${ARTIFACTS[1]}"
echo "         ${ARTIFACTS[2]}"
