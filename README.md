# Lupine MLIP Elastic Benchmark Reproducible Artifact (2026-06-27)

> **One-line claim:** On a 16-element cubic-metal benchmark, a single-model
> 1×1×1 MatPES elastic-constant calculation matches the accuracy of a 3×3×3
> reference run (14.55 vs 14.61 GPa MAE) while costing ~3.9× fewer CPU-seconds.
> The Lupine correction operator tested here does not improve accuracy on this
> MLIP set; the ensemble is the accuracy winner at 11.60 GPa but costs more.

This package is the HPC reproducibility artifact for the MLIP elastic benchmark.
It is a cost-accuracy comparison, not a new-physics claim: the physics
(supercell-size independence of uMLIPs) is demonstrated in the existing
16-element grid, and this artifact layers on the Lupine correction operator and
an honest cost ledger.

## Archived release and citation

Version `v1.0.0` preserves the code, canonical aggregate, and all per-case raw
JSON files. Cite the archived release using `CITATION.cff`. `MANIFEST.sha256`
binds every release file by SHA-256; verify it from the repository root with:

```bash
sha256sum --check MANIFEST.sha256
```

The canonical aggregate honestly retains `git_sha: unknown` and `host: unknown`
in its generation-time provenance. The release manifest proves which files are
archived together; it does not retroactively infer those missing fields.

## Quickstart

```bash
./run.sh
```

To validate the preserved canonical aggregate without rerunning model
inference:

```bash
python3 scripts/verify.py
```

This produces `results/mlip_elastic_benchmark_results.json`, which should byte-match the
canonical run within model nondeterminism.

A 2-element smoke test (Ca, Cu, < 5 min) can be run with:

```bash
python3 scripts/verify.py --smoke
```

## Directory map

```
.
├── Apptainer.def          # Reproducible container recipe
├── README.md              # This file
├── config/
│   └── mlip_elastic_benchmark.yaml      # 192-case matrix + metadata + arm definitions
├── results/               # Generated outputs
│   ├── raw/               # Per-case lupine.layer2.raw.v1 JSON
│   └── mlip_elastic_benchmark_results.json
├── run.sh                 # Top-level orchestration
└── scripts/
    ├── runner.py          # Execute the 192-case benchmark matrix
    ├── aggregate.py       # Compute the four arms, LOO-PCA bias, conformal radius, cost ratios
    └── verify.py          # Schema / smoke / plausibility checks
```

## Cost model

Wall-clock seconds are taken from each per-case JSON under `runtime_seconds`.
All headline costs are **cache-warm** (model checkpoints pre-downloaded before
timing). Conversion to core-hours uses `n_cores = 1` for the per-case
CPU-equivalent figure:

```
core_hours = runtime_seconds × n_cores / 3600
```

| Arm | Runtime source |
|-----|----------------|
| `raw-1x1x1` | TensorNet/PBE 1×1×1 from `results/raw/` (falls back to `lupine/data/layer2_outputs` for Cu/Ni) |
| `corrected-1x1x1` | = raw-1x1×1 + LOO-PCA correction (negligible post-hoc algebra) |
| `ref-3x3x3` | TensorNet/PBE 3×3×3 from `results/raw/`, then existing `lupine/data/layer2_outputs_3x3x3_16elem/` |
| `ensemble-1x1x1` | Sum of M3GNet + CHGNet + TensorNet / PBE / 1×1×1 runtimes |

## Caveats

- **r2SCAN targets are approximated** by a scalar bulk-modulus shift (Liu 2024);
  headline numbers are reported against `TPBE_0K`, with r2SCAN as a sensitivity
  check.
- **Au uses a PW91-GGA fallback**, not PBE.
- **QET≡TensorNet** in MatPES 2025.2; the ensemble uses only three distinct
  architectures (M3GNet, CHGNet, TensorNet).
- **LOO-PCA bias** is mandatory: the bias for each element is the first
  principal component of the centered error matrix of the other 15 elements.
  In-sample bias fitting invalidates the claim.
- **Costs are cache-warm, single-seed, single-relax** CPU-equivalent core-hours.
- **The operator is untested on MLIPs at scale** before this artifact. On this
  16-element set the global LOO-PCA bias degrades accuracy (63.40 GPa vs 14.55
  GPa raw); the pre-registered pivot is to the supercell-independence result.

## Container build

```bash
apptainer build mlip-elastic-benchmark-2026.sif Apptainer.def
```

Run with the lupine repository mounted at the expected path:

```bash
apptainer run --bind /path/to/lupine:/home/alex/Dev/lupine/lupine mlip-elastic-benchmark-2026.sif
```

## Assumptions

- The lupine repository is available at `/home/alex/Dev/lupine/lupine` (or
  override with `BENCHMARK_DRIVER` and `BENCHMARK_OPERATOR_ROOT`).
- Python 3.11 with the pinned stack (`matgl==4.0.2`, `matcalc==0.5.1`,
  `pymatgen==2026.5.4`, `numpy==2.4.6`, `scipy==1.17.1`, `pyyaml==6.0.3`) is
  used; the canonical data were produced with these versions.
- A network connection is available for the first model-download pass; timings
  used in the headline are taken only after `--warm-cache`.
