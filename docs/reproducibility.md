# Reproduction guide

## Release boundary

This repository is extracted from
`route_a_core_checks_evidence_2026-09-16_v2.zip`, SHA-256
`7c3e033e5e913706250fdebcdfe1579c15a079ee4a0b775072c00fc3867d999e`.
Its execution scope is the final K1 clinical stabilization, K2 calibration, and
Q-interface/component sensitivities present in that package. Earlier discovery,
cross-modal experiments, and raw-assay preprocessing are not silently claimed
to be included.

No experiments were rerun to prepare this release. Packaging validation checks
source provenance, syntax/imports, entrypoint help, and absence of data from the
Git repository. It does not establish fresh numerical equivalence of every
execution path. Final numerical results and their prior independent audit
receipts remain in the separate review materials.

## Environment and determinism

The archive records Python 3.13.5 on Linux with the exact package versions in
`requirements.txt`; these are recorded versions, not newly tested installation
claims. Reuse the original environment or resolve these exact builds where
available. Do not silently substitute versions when claiming exact replay.
LightGBM, BLAS/compiler differences, and dependency changes can affect fitted
results. The release vendors the actual graph backend used in the archive.

The scripts set numerical worker thread counts to one. Also set these before
starting a fresh interpreter:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Execute commands from the repository root. Use new output directories outside
the repository; immutable outputs intentionally reject accidental overwrites.
Only load trusted local NumPy caches: a few archived array readers require
`allow_pickle=True` for the original metadata representation.

## Existing evidence, without model refitting

The sibling `../review_materials/` folder holds final aggregate tables, figures,
full synthetic fitted intermediates, independent audit receipts, and an
extraction manifest. It is not pushed to GitHub. After receiving that folder,
one may reproduce figures from the delivered source tables:

```bash
python analysis/plot_main.py \
  --source-tables ../review_materials/figures/source_data \
  --output ../reproduced_figures/main
python analysis/plot_sensitivity.py \
  --source-tables ../review_materials/figures/source_data \
  --output ../reproduced_figures/q_sensitivity
```

The optional numerical auditor reads stored fits, recreates recorded synthetic
draws from their seeds, and checks moments, Fieller sets, fold separation,
family calculations, and file hashes without refitting models:

```bash
python analysis/verify_simulations.py \
  --input ../review_materials/synthetic/calibration \
  --summary ../review_materials/tables \
  --output ../calibration_audit.json
python analysis/verify_simulations.py --repair \
  --input ../review_materials/synthetic/q_sensitivity \
  --summary ../review_materials/tables/plan_b/Q_repair_summary_v1 \
  --output ../q_sensitivity_audit.json
```

These are instructions for future reproduction. Neither numerical replay nor
plot regeneration was executed during curation. Historical experiments excluded
from this release are not rechecked by the bundle-only auditor.

## Full synthetic execution (optional; generates new output files)

```bash
python analysis/run_calibration.py --output ../runs/calibration --reps 200 --workers 6
python analysis/summarize_calibration.py \
  --input ../runs/calibration --output ../runs/calibration_summary
python analysis/run_q_sensitivity.py --output ../runs/q_sensitivity --reps 100 --workers 6
python analysis/summarize_q_sensitivity.py \
  --input ../runs/q_sensitivity --output ../runs/q_sensitivity_summary
python analysis/nuisance_sensitivity.py \
  --input ../runs/calibration --output ../runs/component_sensitivity \
  --reason "One-component nuisance sensitivity of the final calibration" --workers 6
```

The primary seed root is `202609160601`: four cells (`n=85/241`, signal `0/1`),
200 datasets per cell, two balanced topologies, for 800 distinct datasets.
The Q-sensitivity root is `202609160602`: 100 datasets per cell, 400 additional
datasets. Component replacements reuse the primary 800 datasets and add no
independent evidence. The F2 generator, population truths, seed mapping,
candidate-null controls, paired comparisons, failure denominators, and
unbounded interval outcomes must all be retained. Small smoke-test batches are
not the reported experiment, and balance checks require even replicate counts.

## Optional software tests

The archived targeted tests are included for future maintenance. To run them
in an appropriate environment, install `pytest==9.1.1` and use `python -m pytest`.
They were not executed during packaging: some tests generate small numerical
fixtures, and this release was requested without experimental reruns. The source
archive's prior test receipt remains in the separate review folder.

## Clinical execution

Clinical execution requires the original authorized, frozen analysis caches;
the ZIP intentionally excludes patient-level arrays and patient fold records.
This code-only release cannot reconstruct them from aggregate tables.
See [data_contract.md](data_contract.md) for the exact interface and missing
inputs. Do not guess patient matches or replace frozen folds.

With those caches available locally:

```bash
export BREAST_ICI_CLINICAL_CACHE=/absolute/path/to/authorized_cache_root
python analysis/clinical_stabilization.py --output ../runs/clinical_stabilization --workers 2
export BREAST_ICI_STABILIZATION_RESULTS=/absolute/path/to/runs/clinical_stabilization
python analysis/clinical_q_sensitivity.py --output ../runs/clinical_q_sensitivity --workers 2
```

The first command holds patients, graph realizations, folds, Q and m fixed. It
reconstructs required inner nuisance predictions and compares the four inner/
outer stabilization combinations. The second changes Q and matched g only.
Both primary cohorts and all three repartitions are retained. The source cache
names are stable artifact identifiers, not invitations to rerun earlier work.

## Interpretation and audit discipline

Retain failed fits in their planned denominators. Preserve positive and negative
controls, full Fieller sets, Monte Carlo uncertainty, and paired comparisons.
The Q sensitivity is not a replacement primary analysis selected for a better
p-value. Repartitioning does not create new patients; oracle substitutions do
not create new datasets. Publication claims should respect the documented
[method boundaries](method.md).
