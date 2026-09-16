# BreastCancer_ICI

Manuscript code for candidate-role-aware analysis of breast cancer immunotherapy
benefit biomarkers, with CXCL9 conditional on IFNG as the clinical application.

The implementation combines target-local causal discovery, adjustment-interface
checks, nested orthogonal scores, and ratio-aware inference. It includes the
final clinical stabilization, simulation calibration, and Q-interface sensitivity
analyses. It is research software, not a clinical prediction service.

## Setup

Use Python 3.13 and the recorded dependency versions:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from this repository. Analysis entrypoints load `src/` directly;
an editable installation is optional: `python -m pip install -e . --no-deps`.

## Contents

| Directory | Purpose |
| --- | --- |
| `src/tacb_bci/route_ab/` | Role closure, nuisance estimation, benefit scores, and inference |
| `src/tacb_bci/_vendor/` | Required native TLCD and graph-role dependencies |
| `analysis/` | Final experiment, summary, figure, and audit entrypoints |
| `tests/` | Targeted score, ratio-inference, and role-interface tests |
| `docs/` | Reproduction instructions, input contracts, and interpretation |

No datasets, fitted outputs, or figures are stored in this repository. The
separate review-materials folder contains the archived final evidence.

Start with [reproducibility](docs/reproducibility.md), the
[clinical input contract](docs/data_contract.md), and the
[method description](docs/method.md).

## Scope

The estimand concerns information about incremental treatment benefit, not the
effect of intervening on gene expression. Identification and finite-sample
limitations remain applicable; a graph certificate alone does not establish
clinical causality.

Code was curated from the sealed final core-check evidence package without
rerunning experiments. `SOURCE_PROVENANCE.json` records source hashes and changes.
See [third-party notices](THIRD_PARTY_NOTICES.md) before redistribution.
