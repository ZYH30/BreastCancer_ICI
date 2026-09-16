# Third-party provenance

The required graph backend is a source subset from the archived
`Causal-Uplift-Model_993ed47` tree, identified in the evidence package by commit
`993ed47266539545b7eb14249f3e3ccfd75d1f25`.

The following files are vendored byte-for-byte under `src/tacb_bci/_vendor/`:

- `graph_roles.py`
- `tlcd/__init__.py`
- `tlcd/tlcd_runner.py`
- `tlcd/direction_methods.py`

`SOURCE_PROVENANCE.json` records their archive paths and SHA-256 digests. Only the
LightGBM graph backend used by the released analyses is supported. Optional KAN
and language-model backends from the wider upstream project are not bundled.

The source evidence package does not supply an accompanying upstream license
file. This release does not invent or extend a license grant for those files.
The repository owner should confirm applicable permissions before public
redistribution and add the appropriate license notices. Python dependencies
retain their respective licenses.
