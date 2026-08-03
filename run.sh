#!/usr/bin/env bash
# Top-level orchestration for the Lupine mlip elastic benchmark reproducible artifact.
#
# One command: ./run.sh
# Produces results/mlip_elastic_benchmark_results.json byte-matching the canonical run
# within model nondeterminism.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${BENCHMARK_PYTHON:-$(command -v python3.11 || command -v python3)}"
# Prefer the canonical venv used to generate the existing data, if present.
CANONICAL_VENV="/home/alex/Dev/lupine/lupine/data/.venv"
if [[ -z "${BENCHMARK_VENV:-}" ]]; then
    if [[ -x "${CANONICAL_VENV}/bin/python" ]]; then
        export BENCHMARK_VENV="${CANONICAL_VENV}"
    else
        export BENCHMARK_VENV="${SCRIPT_DIR}/.venv"
    fi
fi

# Use the venv interpreter for all Python calls if it is executable.
if [[ -x "${BENCHMARK_VENV}/bin/python" ]]; then
    PYTHON="${BENCHMARK_VENV}/bin/python"
fi

mkdir -p results/raw

echo "== Lupine MLIP Elastic Benchmark Reproducible Artifact =="
echo "Python: ${PYTHON}"
echo "Venv:   ${BENCHMARK_VENV}"
echo "Driver: ${BENCHMARK_DRIVER:-$(python3 -c 'import yaml; print(yaml.safe_load(open("config/mlip_elastic_benchmark.yaml"))["data"]["driver"])' 2>/dev/null || echo '<config default>')}"
echo

# 1. Warm the model cache so all headline timings are cache-warm.
echo "[1/4] Warming model cache ..."
"${PYTHON}" scripts/runner.py --warm-cache

# 2. Run the full 192-case matrix (16 elements x 3 models x 2 functionals x 2 supercells).
echo "[2/4] Running full benchmark matrix ..."
"${PYTHON}" scripts/runner.py

# 3. Aggregate per-case outputs into mlip_elastic_benchmark_results.json.
echo "[3/4] Aggregating results ..."
"${PYTHON}" scripts/aggregate.py

# 4. Validate schema, coverage, and plausible cost ratios.
echo "[4/4] Verifying results ..."
"${PYTHON}" scripts/verify.py

echo
echo "Artifact complete: results/mlip_elastic_benchmark_results.json"
ls -lh results/mlip_elastic_benchmark_results.json
