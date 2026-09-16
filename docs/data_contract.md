# Clinical input contract and data separation

No clinical or synthetic datasets, model outputs, figures, or experimental
tables are committed to this repository. Keep authorized clinical caches
outside Git and set `BREAST_ICI_CLINICAL_CACHE` to their root. The expected
layout preserves the archived source artifact identifiers:

```text
authorized_cache_root/
  results/
    route_ab/
      cxcl9_ifng_benefit_increment_v1/{ISPY2,NeoTRIP}/
      common_nuisance_controls_v1/{ISPY2,NeoTRIP}/
      ifng_partial_family_audit_v1/
      role_calibrated_final_family_v1/
      ispy2_extended_role_context_v2/
      cxcl9_nonlinear_IFNG_context_v1/{ISPY2,NeoTRIP}/
      two_legal_nuisance_interfaces_v1/ISPY2/
      route_a_first_cycle_v1/NeoTRIP/
    route_a_enhancement/
      E3_full_spline_stability_v1/{trial}_split{0,1,2}_all/
```

## Required cache interfaces

| Artifact | Required content |
| --- | --- |
| `cxcl9_ifng_benefit_increment_v1/{trial}` | `extended_W.csv` with the original ordered baseline measurements, including CXCL9, IFNG, and the 18-marker family; `patient_scores.csv` with `patient_hash`, binary `arm`, binary `y`, `q0_new`, `q1_new` |
| `common_nuisance_controls_v1/{trial}` | `patient_scores.csv` with patient hashes and original five-fold membership |
| `ifng_partial_family_audit_v1` | `{trial}_family_arrays.npz`, including `{linear,spline}__{gene}__rB` for every original candidate |
| `role_calibrated_final_family_v1` | `qualification_arrays.npz`, `role_calibrated_all18_partial_benefit.csv`, `two_study_average_and_conjunction.csv` |
| `ispy2_extended_role_context_v2` | `all_arrays.npz` containing `e_closed`; five `fold{f}_extended_roles.json` records with graph, roles, patient axes, inner nuisance fits, and g selections |
| `cxcl9_nonlinear_IFNG_context_v1/{trial}` | `nested_models.json` with exact training/test identities, nested partitions, and model selection records |
| `E3_full_spline_stability_v1/{trial}_split{k}_all` | `job_before_results.json`, `LOCAL_patient_axis.csv`, `clinical_results.csv`; `fit/arrays.npz`, `fit/model_records.json`, `fit/roles.json`, and original graph/inner-fold files |
| `two_legal_nuisance_interfaces_v1/ISPY2` | Q sensitivity: `estimation_arrays.npz` and five `fold{f}.json` outcome-interface records |
| `route_a_first_cycle_v1/NeoTRIP` | Q sensitivity: five `roles_fold{f}.json` records; unresolved added measurements remain explicitly unresolved |

All row orders, names, treatment/outcome coding, precision, saved seeds, folds,
and graph identities must match the original cache. Read CSVs with round-trip
float precision. Do not impute absent identifiers, substitute a different
normalization, or reuse a cache across patients. Clinical measurements represent
pretreatment assays; the code does not turn on-treatment measurements into
baseline causal variables.

For the Q sensitivity, additionally set `BREAST_ICI_STABILIZATION_RESULTS` to
the complete clinical stabilization output, which contains `LOCAL_arrays.npz`,
`LOCAL_model_records.json`, and `LOCAL_fold{f}_{method}_inner.npz` for each run.
These are private intermediates, not public code assets.

## What the separate review folder contains

`review_materials/` contains the final aggregate clinical tables, sanitized
model interfaces, complete confidence sets, final synthetic inputs and fitted
intermediates, summary tables, figures and their source data, and audit receipts.
`EXTRACTION_MANIFEST.json` maps every extracted evidence file to its source ZIP
member and SHA-256 digest. Component diagnostics reuse existing synthetic
datasets and are labeled accordingly.

The source ZIP's `LOCAL_ONLY_INPUT_MANIFEST.json` lists withheld local patient
artifacts by path/hash; it is retained only in the separate review folder.
It is not a complete raw-data acquisition or preprocessing manifest. The ZIP
does not contain the patient-level caches above, so this curation cannot make
clinical re-fitting self-contained. No nonpublic data were fetched, copied from
other workspace locations, or uploaded. A journal data-access statement must
distinguish the aggregate/synthetic evidence from these patient-level inputs.
