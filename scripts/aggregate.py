#!/usr/bin/env python3
"""Aggregate raw mlip elastic benchmark outputs into mlip_elastic_benchmark_results.json.

Reads per-case JSON from results/raw/, falls back to existing 3x3x3 libraries,
computes the four arms, LOO-PCA bias, conformal radius, and cost ratios.

Headline numbers are reported against TPBE_0K; Tr2SCAN_0K is kept as a
sensitivity column because the r2SCAN targets are synthesized from a scalar
bulk-modulus shift.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "mlip_elastic_benchmark.yaml"
RAW_DIR = ROOT / "results" / "raw"
RESULTS_PATH = ROOT / "results" / "mlip_elastic_benchmark_results.json"


def _add_operator_path(config: dict) -> None:
    op_root = Path(os.environ.get("BENCHMARK_OPERATOR_ROOT", config["data"]["operator_module_root"]))
    if str(op_root) not in sys.path:
        sys.path.insert(0, str(op_root))


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_raw_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def parse_output_filename(name: str) -> dict[str, Any] | None:
    if not name.endswith(".json"):
        return None
    stem = name[:-5]
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    element, model, functional, supercell_str = parts
    supercell = int(supercell_str.split("x")[0])
    return {
        "element": element,
        "model": model,
        "functional": functional,
        "supercell": supercell,
    }


def parse_legacy_filename(name: str, supercell: int) -> dict[str, Any] | None:
    if not name.endswith(".json"):
        return None
    stem = name[:-5]
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    element, model, functional = parts
    return {
        "element": element,
        "model": model,
        "functional": functional,
        "supercell": supercell,
    }


def index_raw_results(raw_dir: Path) -> dict[tuple[str, str, str, int], dict]:
    index: dict[tuple[str, str, str, int], dict] = {}
    for path in find_raw_files(raw_dir):
        keys = parse_output_filename(path.name)
        if keys is None:
            continue
        try:
            data = load_json(path)
        except json.JSONDecodeError:
            continue
        if data.get("status") != "ok":
            continue
        index[(keys["element"], keys["model"], keys["functional"], keys["supercell"])] = data
    return index


def index_existing_3x3x3_16elem(directory: Path) -> dict[tuple[str, str, str, int], dict]:
    index: dict[tuple[str, str, str, int], dict] = {}
    if not directory.exists():
        return index
    for elem_dir in directory.iterdir():
        if not elem_dir.is_dir():
            continue
        for path in elem_dir.glob("*.json"):
            keys = parse_legacy_filename(path.name, supercell=3)
            if keys is None:
                continue
            try:
                data = load_json(path)
            except json.JSONDecodeError:
                continue
            if data.get("status") != "ok":
                continue
            index[(keys["element"], keys["model"], keys["functional"], keys["supercell"])] = data
    return index


def index_existing_1x1x1(directory: Path) -> dict[tuple[str, str, str, int], dict]:
    index: dict[tuple[str, str, str, int], dict] = {}
    if not directory.exists():
        return index
    for path in directory.glob("*.json"):
        keys = parse_legacy_filename(path.name, supercell=1)
        if keys is None:
            continue
        try:
            data = load_json(path)
        except json.JSONDecodeError:
            continue
        if data.get("status") != "ok":
            continue
        index[(keys["element"], keys["model"], keys["functional"], keys["supercell"])] = data
    return index


def index_existing_3x3x3_cuni(directory: Path) -> dict[tuple[str, str, str, int], dict]:
    index: dict[tuple[str, str, str, int], dict] = {}
    if not directory.exists():
        return index
    for path in directory.glob("*.json"):
        keys = parse_legacy_filename(path.name, supercell=3)
        if keys is None:
            continue
        try:
            data = load_json(path)
        except json.JSONDecodeError:
            continue
        if data.get("status") != "ok":
            continue
        index[(keys["element"], keys["model"], keys["functional"], keys["supercell"])] = data
    return index


def get_case(
    index: dict[tuple[str, str, str, int], dict],
    element: str,
    model: str,
    functional: str,
    supercell: int,
) -> dict | None:
    return index.get((element, model, functional, supercell))


def cij_vector(data: dict) -> np.ndarray:
    return np.array([data["c11"], data["c12"], data["c44"]], dtype=float)


def mae_vs_target(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def loo_pca_bias(
    raw_preds: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """LOO first-principal-component bias for each element against TPBE targets."""
    elements = sorted(raw_preds.keys())
    errors = {e: raw_preds[e] - targets[e] for e in elements}
    biases = {}
    for e in elements:
        others = [k for k in elements if k != e]
        error_matrix = np.stack([errors[o] for o in others])
        centered = error_matrix - error_matrix.mean(axis=0)
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        pc = vh[0] * s[0]
        mean_error = error_matrix.mean(axis=0)
        if np.dot(pc, mean_error) < 0:
            pc = -pc
        biases[e] = pc
    return biases


def conformal_radius(rows: list[dict], coverage: float = 0.90) -> float:
    from lupine.operator import leave_one_out_calibration

    return leave_one_out_calibration(rows, coverage=coverage).radius


def bootstrap_ci(maes: list[float], n: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.array(maes, dtype=float)
    if len(samples) < 2:
        return (float(samples[0]) if len(samples) else 0.0, float(samples[0]) if len(samples) else 0.0)
    means = [np.mean(rng.choice(samples, size=len(samples), replace=True)) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(ROOT),
            )
            .stdout.strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def build_arm_records(
    config: dict,
    raw_index: dict,
    existing_1x1x1_index: dict,
    existing_3x3x3_index: dict,
    targets: dict[str, dict],
    smoke: bool,
) -> tuple[dict[str, dict], list[dict], dict[str, float]]:
    elements = config["smoke"]["elements"] if smoke else config["cases"]["elements"]

    def lookup(element: str, model: str, functional: str, supercell: int) -> dict | None:
        case = get_case(raw_index, element, model, functional, supercell)
        if case is not None:
            return case
        if supercell == 1:
            case = get_case(existing_1x1x1_index, element, model, functional, supercell)
            if case is not None:
                return case
        if supercell == 3:
            case = get_case(existing_3x3x3_index, element, model, functional, supercell)
            if case is not None:
                return case
        return None

    tpbe = {e: np.array(targets[e]["TPBE_0K"], dtype=float) for e in elements}
    tr2scan = {e: np.array(targets[e]["Tr2SCAN_0K"], dtype=float) for e in elements}
    shift = {e: tr2scan[e] - tpbe[e] for e in elements}

    best_model = config["arms"]["raw-1x1x1"]["model"]
    best_functional = config["arms"]["raw-1x1x1"]["functional"]

    raw_1x1x1: dict[str, dict] = {}
    raw_3x3x3: dict[str, dict] = {}
    missing = []
    for e in elements:
        case = lookup(e, best_model, best_functional, 1)
        if case is None:
            missing.append((e, best_model, best_functional, 1))
            continue
        raw_1x1x1[e] = case
        case3 = lookup(e, best_model, best_functional, 3)
        if case3 is None:
            missing.append((e, best_model, best_functional, 3))
            continue
        raw_3x3x3[e] = case3

    if missing:
        for item in missing:
            print(f"WARNING: missing case {item}", file=sys.stderr)
        if not smoke:
            raise RuntimeError(f"Missing {len(missing)} required cases; rerun runner.py first.")

    raw_preds_1x1x1 = {e: cij_vector(raw_1x1x1[e]) for e in raw_1x1x1}
    raw_preds_3x3x3 = {e: cij_vector(raw_3x3x3[e]) for e in raw_3x3x3}

    biases = loo_pca_bias(raw_preds_1x1x1, {e: tpbe[e] for e in raw_preds_1x1x1})

    ensemble_models = config["arms"]["ensemble-1x1x1"]["models"]
    ensemble_preds: dict[str, list[np.ndarray]] = {e: [] for e in elements}
    ensemble_cases: dict[str, list[dict]] = {e: [] for e in elements}
    for e in elements:
        for model in ensemble_models:
            case = lookup(e, model, best_functional, 1)
            if case is None:
                missing.append((e, model, best_functional, 1))
                continue
            ensemble_preds[e].append(cij_vector(case))
            ensemble_cases[e].append(case)
    if missing and not smoke:
        raise RuntimeError(f"Missing {len(missing)} ensemble cases; rerun runner.py first.")

    # Scalar-bulk v0.2 operator: LOO re-scaling of the bulk-modulus functional shift.
    from lupine.operator import correct_scalar_bulk, leave_one_out_scalar_bulk

    scalar_bulk_rows = [
        {
            "element": e,
            "raw_prediction": raw_preds_1x1x1[e].tolist(),
            "functional_shift": shift[e].tolist(),
            "target_r2scan": tr2scan[e].tolist(),
        }
        for e in elements
        if e in raw_preds_1x1x1
    ]
    scalar_alphas = leave_one_out_scalar_bulk(scalar_bulk_rows)

    arm_a_records = []
    arm_b_records = []
    arm_c_records = []
    arm_d_records = []
    arm_e_records = []
    calibration_rows = []

    for e in elements:
        if e not in raw_preds_1x1x1:
            continue
        target_pbe = tpbe[e]
        target_r2s = tr2scan[e]
        raw = raw_preds_1x1x1[e]
        bias = biases[e]
        corrected = raw - bias + shift[e]
        corrected_sb = correct_scalar_bulk(raw, shift[e], scalar_alphas[e])
        ref3 = raw_preds_3x3x3.get(e)
        ens = np.mean(ensemble_preds[e], axis=0) if ensemble_preds[e] else None

        runtime_a = raw_1x1x1[e].get("runtime_seconds", 0.0)
        runtime_b = runtime_a
        runtime_c = raw_3x3x3[e].get("runtime_seconds", 0.0) if ref3 is not None else 0.0
        runtime_d = sum(c.get("runtime_seconds", 0.0) for c in ensemble_cases.get(e, []))

        def rec(arm: str, pred: np.ndarray, runtime: float) -> dict:
            return {
                "element": e,
                "arm": arm,
                "c11": float(pred[0]),
                "c12": float(pred[1]),
                "c44": float(pred[2]),
                "mae": mae_vs_target(pred, target_pbe),
                "mae_tr2scan": mae_vs_target(pred, target_r2s),
                "runtime_s": runtime,
            }

        arm_a_records.append(rec("raw-1x1x1", raw, runtime_a))
        arm_b_records.append(rec("corrected-1x1x1", corrected, runtime_b))
        arm_e_records.append(rec("scalar-bulk-1x1x1", corrected_sb, runtime_a))
        calibration_rows.append(
            {
                "raw_prediction": raw.tolist(),
                "bias_vector": bias.tolist(),
                "functional_shift": shift[e].tolist(),
                "target": target_r2s.tolist(),
            }
        )
        if ref3 is not None:
            arm_c_records.append(rec("ref-3x3x3", ref3, runtime_c))
        if ens is not None:
            arm_d_records.append(rec("ensemble-1x1x1", ens, runtime_d))

    def summarize(records: list[dict]) -> dict:
        if not records:
            return {
                "mae_cij": None,
                "mae_cij_tr2scan": None,
                "median_mae_cij": None,
                "mae_ci_low": None,
                "mae_ci_high": None,
                "core_hours": 0.0,
                "total_runtime_seconds": 0.0,
                "n_cases": 0,
            }
        maes = [r["mae"] for r in records]
        maes_r2 = [r["mae_tr2scan"] for r in records]
        total_s = sum(r["runtime_s"] for r in records)
        core_hours = total_s * config["cost"]["n_cores"] / config["cost"]["seconds_to_hours"]
        low, high = bootstrap_ci(maes)
        return {
            "mae_cij": round(float(np.mean(maes)), 4),
            "mae_cij_tr2scan": round(float(np.mean(maes_r2)), 4),
            "median_mae_cij": round(float(np.median(maes)), 4),
            "mae_ci_low": round(low, 4),
            "mae_ci_high": round(high, 4),
            "core_hours": round(core_hours, 6),
            "total_runtime_seconds": round(total_s, 2),
            "n_cases": len(records),
        }

    arms = {
        "raw-1x1x1": summarize(arm_a_records),
        "corrected-1x1x1": {
            **summarize(arm_b_records),
            "bias_method": config["bias"]["method"],
        },
        "ref-3x3x3": summarize(arm_c_records),
        "ensemble-1x1x1": {
            **summarize(arm_d_records),
            "n_models": len(ensemble_models),
        },
    }

    radius = conformal_radius(calibration_rows, coverage=config["bias"]["coverage"])
    arms["corrected-1x1x1"]["conformal_radius_gpa"] = round(radius, 4)

    arms["scalar-bulk-1x1x1"] = {
        **summarize(arm_e_records),
        "bias_method": "LOO-scalar-bulk",
    }

    cost_ratios = {}
    raw_hours = arms["raw-1x1x1"]["core_hours"]
    if raw_hours and raw_hours > 0:
        cost_ratios["raw_vs_ref"] = round(arms["ref-3x3x3"]["core_hours"] / raw_hours, 2)
        cost_ratios["ensemble_vs_raw"] = round(arms["ensemble-1x1x1"]["core_hours"] / raw_hours, 2)
        cost_ratios["scalar_bulk_vs_raw"] = 1.0
        cost_ratios["corrected_vs_raw"] = 1.0

    per_element = arm_a_records + arm_b_records + arm_c_records + arm_d_records + arm_e_records

    return arms, per_element, cost_ratios


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate mlip elastic benchmark results.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    _add_operator_path(config)

    targets = load_json(Path(config["data"]["targets"]))["elements"]
    raw_index = index_raw_results(args.raw_dir)
    existing_1x1x1 = index_existing_1x1x1(Path(config["data"]["existing_1x1x1"]))
    existing_3x3x3: dict[tuple[str, str, str, int], dict] = {}
    existing_3x3x3.update(index_existing_3x3x3_16elem(Path(config["data"]["existing_3x3x3_16elem"])))
    existing_3x3x3.update(index_existing_3x3x3_cuni(Path(config["data"]["existing_3x3x3_cuni"])))

    arms, per_element, cost_ratios = build_arm_records(
        config,
        raw_index,
        existing_1x1x1,
        existing_3x3x3,
        targets,
        smoke=args.smoke,
    )

    result = {
        "schema_version": config["schema"],
        "provenance": {
            "git_sha": git_sha(),
            "matpes_release": config["matpes_release"],
            "host": os.environ.get("HOSTNAME", "unknown"),
            "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "smoke": args.smoke,
            "target_reference": "TPBE_0K",
            "shift_reference": "Tr2SCAN_0K",
        },
        "arms": arms,
        "cost_ratios": cost_ratios,
        "per_element": per_element,
        "caveats": [
            "r2SCAN targets approximated by scalar bulk-modulus shift (Liu 2024); headline MAE is against TPBE_0K",
            "QET=TensorNet alias deduplicated; ensemble uses three distinct architectures",
            "Costs are cache-warm, single-seed, single-relax, CPU-equivalent core-hours (n_cores=1)",
            "Au uses a PW91-GGA fallback, not PBE",
            "LOO-PCA bias is fitted on the other 15 elements against TPBE_0K; in-sample bias invalidates the claim",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
