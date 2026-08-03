#!/usr/bin/env python3
"""Smoke-test validator for the mlip elastic benchmark artifact.

Checks schema conformance, required element coverage, and plausible cost ratios.
In --smoke mode it first runs the 2-element subset through runner + aggregate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "mlip_elastic_benchmark.yaml"
RESULTS_PATH = ROOT / "results" / "mlip_elastic_benchmark_results.json"
SMOKE_RESULTS_PATH = ROOT / "results" / "mlip_elastic_benchmark_results_smoke.json"

REQUIRED_TOP_KEYS = [
    "schema_version",
    "provenance",
    "arms",
    "cost_ratios",
    "per_element",
    "caveats",
]
REQUIRED_PROVENANCE_KEYS = ["git_sha", "matpes_release", "host", "run_at"]
REQUIRED_ARMS = ["raw-1x1x1", "corrected-1x1x1", "scalar-bulk-1x1x1", "ref-3x3x3", "ensemble-1x1x1"]
REQUIRED_ARM_KEYS = ["mae_cij", "core_hours", "n_cases"]
REQUIRED_RATIO_KEYS = ["raw_vs_ref", "ensemble_vs_raw", "corrected_vs_raw"]
REQUIRED_RECORD_KEYS = ["element", "arm", "c11", "c12", "c44", "mae", "runtime_s"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def smoke_python() -> str:
    """Return the Python interpreter to use for the smoke pipeline."""
    env = os.environ.get("BENCHMARK_VENV", "")
    candidates = [
        Path(env) / "bin" / "python" if env else None,
        Path("/home/alex/Dev/lupine/lupine/data/.venv/bin/python"),
        ROOT / ".venv" / "bin" / "python",
        Path("/usr/bin/python3.11"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return sys.executable


def run_smoke_pipeline(config: dict) -> Path:
    """Run runner.py --smoke then aggregate.py --smoke."""
    python = smoke_python()
    print(f"Running smoke runner (Ca, Cu) with {python} ...")
    subprocess.run(
        [python, str(ROOT / "scripts" / "runner.py"), "--smoke"],
        check=True,
        cwd=str(ROOT),
    )
    print("Aggregating smoke results ...")
    subprocess.run(
        [
            python,
            str(ROOT / "scripts" / "aggregate.py"),
            "--smoke",
            "--output",
            str(SMOKE_RESULTS_PATH),
        ],
        check=True,
        cwd=str(ROOT),
    )
    return SMOKE_RESULTS_PATH


def validate(results: dict, config: dict, smoke: bool) -> list[str]:
    errors: list[str] = []

    # Top-level keys.
    for key in REQUIRED_TOP_KEYS:
        if key not in results:
            errors.append(f"missing top-level key: {key}")

    if results.get("schema_version") != config["schema"]:
        errors.append(
            f"schema_version mismatch: {results.get('schema_version')} != {config['schema']}"
        )

    # Provenance.
    provenance = results.get("provenance", {})
    for key in REQUIRED_PROVENANCE_KEYS:
        if key not in provenance:
            errors.append(f"missing provenance key: {key}")

    # Arms.
    arms = results.get("arms", {})
    for arm_name in REQUIRED_ARMS:
        if arm_name not in arms:
            errors.append(f"missing arm: {arm_name}")
            continue
        arm = arms[arm_name]
        for key in REQUIRED_ARM_KEYS:
            if key not in arm:
                errors.append(f"{arm_name}: missing key {key}")

    expected_elements = set(config["smoke"]["elements"] if smoke else config["cases"]["elements"])
    for arm_name in REQUIRED_ARMS:
        arm = arms.get(arm_name, {})
        n = arm.get("n_cases", 0)
        expected_n = len(expected_elements)
        if n != expected_n:
            errors.append(f"{arm_name}: n_cases={n}, expected {expected_n}")

    # Every expected element should appear in corrected-1x1x1 records.
    corrected_elements = {
        r["element"]
        for r in results.get("per_element", [])
        if r.get("arm") == "corrected-1x1x1"
    }
    missing = expected_elements - corrected_elements
    if missing:
        errors.append(f"corrected-1x1x1 missing elements: {sorted(missing)}")

    # Per-element record shape.
    for i, record in enumerate(results.get("per_element", [])):
        for key in REQUIRED_RECORD_KEYS:
            if key not in record:
                errors.append(f"per_element[{i}]: missing key {key}")

    # Cost-ratio plausibility.
    ratios = results.get("cost_ratios", {})
    for key in REQUIRED_RATIO_KEYS:
        if key not in ratios:
            errors.append(f"missing cost ratio: {key}")
    rvr = ratios.get("raw_vs_ref")
    evr = ratios.get("ensemble_vs_raw")
    cvr = ratios.get("corrected_vs_raw")
    if rvr is not None and not (1.0 <= rvr <= 100.0):
        errors.append(f"raw_vs_ref cost ratio implausible: {rvr}")
    if evr is not None and not (1.0 <= evr <= 50.0):
        errors.append(f"ensemble_vs_raw cost ratio implausible: {evr}")
    if cvr is not None and not (0.1 <= cvr <= 10.0):
        errors.append(f"corrected_vs_raw cost ratio implausible: {cvr}")

    # Note: we do NOT enforce that the operator improves. The mlip elastic benchmark design
    # pre-registers a pivot to the supercell-independence story if it does not.

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify mlip elastic benchmark results.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke subset end-to-end and validate the smoke output.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to mlip_elastic_benchmark.yaml.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="Path to mlip_elastic_benchmark_results.json to validate.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.smoke:
        results_path = run_smoke_pipeline(config)
    else:
        results_path = args.results

    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        return 1

    results = load_json(results_path)
    errors = validate(results, config, smoke=args.smoke)

    if errors:
        print(f"FAILED: {len(errors)} issue(s) found:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"OK: {results_path} validates against lupine.mlip_elastic_benchmark.v1 schema.")
    print(
        f"    Arms: raw={results['arms']['raw-1x1x1']['mae_cij']:.2f} GPa, "
        f"corrected={results['arms']['corrected-1x1x1']['mae_cij']:.2f} GPa, "
        f"ref={results['arms']['ref-3x3x3']['mae_cij']:.2f} GPa, "
        f"ensemble={results['arms']['ensemble-1x1x1']['mae_cij']:.2f} GPa"
    )
    print(
        f"    Cost ratios: raw vs ref = {results['cost_ratios']['raw_vs_ref']:.1f}x, "
        f"ensemble vs raw = {results['cost_ratios']['ensemble_vs_raw']:.1f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
