#!/usr/bin/env python3
"""MLIP elastic benchmark runner.

Reads config/mlip_elastic_benchmark.yaml and executes the per-case Layer-2 driver for every
model x functional x supercell x element case, writing raw per-case JSON to
results/raw/. Supports a 2-element smoke subset and a warm-cache pass.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "mlip_elastic_benchmark.yaml"
RAW_DIR = ROOT / "results" / "raw"
SMOKE_ELEMENTS = {"Ca", "Cu"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cases_from_config(config: dict, smoke: bool = False) -> list[dict]:
    elements = config["smoke"]["elements"] if smoke else config["cases"]["elements"]
    models = config["cases"]["models"]
    functionals = config["cases"]["functionals"]
    supercells = config["cases"]["supercells"]
    cases = []
    for element in elements:
        for model in models:
            for functional in functionals:
                for supercell in supercells:
                    cases.append(
                        {
                            "element": element,
                            "model": model,
                            "functional": functional,
                            "supercell": int(supercell),
                        }
                    )
    return cases


def venv_python(config: dict) -> str:
    """Return the Python interpreter to use (venv or system fallback)."""
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


def driver_path(config: dict) -> str:
    return os.environ.get("BENCHMARK_DRIVER", config["data"]["driver"])


def output_name(case: dict) -> str:
    return f"{case['element']}_{case['model']}_{case['functional']}_{case['supercell']}x{case['supercell']}x{case['supercell']}.json"


def run_case(case: dict, python: str, driver: str, output_dir: Path, dry_run: bool = False) -> dict:
    out_file = output_dir / output_name(case)
    if out_file.exists():
        return {
            "case": case,
            "output": str(out_file),
            "status": "skipped",
            "reason": "output already exists",
        }

    cmd = [
        python,
        driver,
        "--element", case["element"],
        "--model", case["model"],
        "--functional", case["functional"],
        "--supercell", str(case["supercell"]),
        "--output", str(out_file),
    ]
    log = {
        "case": case,
        "command": " ".join(cmd),
        "output": str(out_file),
        "status": "ok",
    }
    if dry_run:
        log["status"] = "dry-run"
        return log

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=7200,
        )
        elapsed = time.time() - start
        log["runtime_seconds"] = round(elapsed, 2)
        if proc.returncode != 0:
            log["status"] = "failed"
            log["stderr"] = proc.stderr[-2000:]
            # Write a failure stub so aggregation can see it.
            failure = {
                "schema_version": "lupine.layer2.raw.v1",
                **case,
                "status": "failed",
                "error": proc.stderr[-2000:],
                "runtime_seconds": round(elapsed, 2),
            }
            out_file.write_text(json.dumps(failure, indent=2) + "\n")
        else:
            # Validate that the driver actually wrote the file.
            if not out_file.exists():
                log["status"] = "failed"
                log["stderr"] = "driver succeeded but output file is missing"
    except subprocess.TimeoutExpired:
        log["status"] = "timeout"
        log["runtime_seconds"] = round(time.time() - start, 2)
    except Exception as exc:  # noqa: BLE001
        log["status"] = "error"
        log["error"] = str(exc)
    return log


def warm_cache(config: dict, python: str, driver: str, output_dir: Path) -> list[dict]:
    """Run one small case per unique model/functional to trigger model downloads.

    This makes all subsequent timings cache-warm, which is required by the
    headline cost model.
    """
    cases = cases_from_config(config, smoke=True)
    # Use the smallest, fastest element from the smoke set (Ca is lighter than Cu).
    warm_element = config["smoke"]["elements"][0]
    seen = set()
    logs = []
    for case in cases:
        key = (case["model"], case["functional"])
        if key in seen:
            continue
        seen.add(key)
        warm_case = {**case, "element": warm_element, "supercell": 1}
        logs.append(run_case(warm_case, python, driver, output_dir))
    return logs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MLIP elastic benchmark matrix.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the 2-element smoke subset (Ca, Cu).",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Pre-download model checkpoints using a minimal warm-up pass.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the case matrix without executing the driver.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to mlip_elastic_benchmark.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory for per-case raw JSON outputs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    python = venv_python(config)
    driver = driver_path(config)

    if not Path(driver).exists():
        print(f"ERROR: driver not found: {driver}", file=sys.stderr)
        print("Set BENCHMARK_DRIVER or mount the lupine repo at the expected path.", file=sys.stderr)
        return 1

    logs: list[dict] = []

    if args.warm_cache:
        print("Warming model cache ...")
        warm_logs = warm_cache(config, python, driver, args.output_dir)
        logs.extend(warm_logs)
        n_ok = sum(1 for log in warm_logs if log["status"] in {"ok", "skipped"})
        print(f"Warm-cache: {n_ok}/{len(warm_logs)} models ready.")

    cases = cases_from_config(config, smoke=args.smoke)
    print(f"Running {len(cases)} cases (smoke={args.smoke}) ...")
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['element']} {case['model']}/{case['functional']} supercell={case['supercell']}")
        log = run_case(case, python, driver, args.output_dir, dry_run=args.dry_run)
        logs.append(log)
        if log["status"] not in {"ok", "skipped", "dry-run"}:
            print(f"  -> {log['status']}: {log.get('stderr', log.get('error', ''))[:200]}", file=sys.stderr)

    summary_path = args.output_dir / "_runner_summary.json"
    summary = {
        "schema_version": "lupine.mlip_elastic_benchmark.runner.v1",
        "cases_total": len(cases),
        "smoke": args.smoke,
        "warm_cache": args.warm_cache,
        "dry_run": args.dry_run,
        "python": python,
        "driver": driver,
        "status_counts": {},
        "cases": logs,
    }
    for log in logs:
        status = log["status"]
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    n_ok = summary["status_counts"].get("ok", 0) + summary["status_counts"].get("skipped", 0)
    print(f"Done: {n_ok}/{len(logs)} cases completed successfully.")
    print(f"Summary written to {summary_path}")
    return 0 if n_ok == len(logs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
