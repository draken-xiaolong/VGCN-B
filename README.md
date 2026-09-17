# VGCN-B reproducibility package

This repository preserves the implementation associated with VGCN-B and adds
explicit reproducibility checks. It is **not yet a fully reproduced or validated
ownership-verification system**. Read `REPRODUCTION_STATUS.md` before use.

## Environment

Use Python 3.11 or newer. Install the PyTorch build appropriate for your machine,
then `python -m pip install -r requirements.txt`. The audit ran with Python 3.13.3,
PyTorch 2.8.0+cu129 and an RTX 5070 Ti. The original scaler was serialized by
scikit-learn 1.6.1; use that version to avoid a cross-version pickle warning.
Dependencies other than the installed audit environment have not been tested as
a fresh environment. Local model/scaler files are Python-serialized artifacts;
only load artifacts from a trusted source.

## Data layout

Raw data are not redistributed here pending source/license confirmation. Place
the author's original GeoJSON files in:

```
convertToGeoJson/GeoJson/TrainingSet/   # 46 original maps
convertToGeoJson/GeoJson/TestSet/       # 43 original maps
zNC-Test/vector-data-geojson/           # six representative original maps
```

The six map stems are H51-BRGA, H51-RESA, H51-RESP,
tianjin-latest-free.shp-gis_osm_pois_free_1,
tianjin-latest-free.shp-gis_osm_railways_free_1, and
tianjin-latest-free.shp-gis_osm_waterways_free_1.
Input manifests in `ReproductionAudit` identify the audited test files by hash.
The absent raw data remain a prerequisite for full reproduction.

## Reproduce checkpoint inference

Run from this repository root, using a UTF-8 console:

```
python -X utf8 reproduction_audit.py
python -X utf8 reproduce_compound.py
python -X utf8 key_scrambling_audit.py
```

These commands write fresh outputs under `ReproductionAudit`. The second command
regenerates attacked data locally; those large derivatives are intentionally
excluded from Git. The supplied inference checkpoint retains only model weights,
without optimizer state. It has no embedded configuration; the loader uses the
original defaults (13 input features, hidden width 128, output width 1024, dual pooling).

To use an external original-data directory without copying it, set `VGCN_TEST_DATA`
to the 43-map test GeoJSON directory and `VGCN_COMPOUND_DATA` to the six-map directory.
The key-scrambling diagnostic demonstrates why different-key registration similarity
must not be substituted for impostor testing against the claimed record's key.

The compound wrapper fixes registration time to `1789600000` (override with
`VGCN_REGISTRATION_TIMESTAMP`) and sorts input filenames. The original script used
wall-clock time to scramble the logo, which can slightly change recovered-logo NC
between runs. A public fixed timestamp is an evaluation control, not a secret key.

## Rebuild augmentation and train

Run `python -X utf8 smoke_train.py` for a small synthetic test of early and late
loss schedules, finite gradients/weights, and parameter updates. This test requires
no map data and does not establish full training reproduction.

The original scripts use working-directory-relative paths. Execute each stage
from the indicated directory, in this order:

1. `convertToGeoJson-Attacked`: `python -X utf8 convertToGeoJson-Attacked-TrainingSet.py`
2. `convertToGraph`: `python -X utf8 convertToGraph-TrainingSet.py`
3. `VGCN`: `python -X utf8 VGCN.py`

Before fresh training set `VGAT_RESUME_CHECKPOINT=0`,
`VGAT_MODEL_BASENAME=retrained_gcn`, `VGAT_RANDOM_SEED=42`,
`VGAT_NUM_EPOCHS=12`, and `VGAT_BATCH_SIZE=4` in the environment.
The `VGAT_` prefix is retained for compatibility; the backbone is a GCN.
Set `VGAT_MODEL_PATH` to the newly trained checkpoint for evaluation.
Graph conversion can replace generated graphs, so run it in a dedicated checkout.
This sequence documents the original entry points; end-to-end fresh training has
not yet been verified. It must not be reported as successful based on inference.

`zNC-Test/Fig1.py` through `Fig12.py` contain the original robustness experiments.
They generate intermediate files and use fixed repository-relative data paths.
Historical `VGAT` column labels refer to the selected VGCN checkpoint in these
experiments and are retained to make original result comparisons traceable.

## Registration code

`BlockChain` contains the original Solidity registry and OSS/IPFS scripts.
No credentials, cloud config, dependency directory or deployed addresses are
included. Configure your own service credentials. **The original contract permits
unauthorized replacement of existing path registrations.** It is preserved for
audit and must not be treated as a secure ownership registry.
No live blockchain or cloud service operation has been reproduced.

An additional `AppendOnlyVectorMapRegistry` prototype separates claimant namespaces
and prohibits replacing an existing record. It has a new interface and is not yet
integrated with the cloud scripts. For local security tests run `npm ci` followed
by `npm test` in `BlockChain`. Compilation uses local solc 0.8.26. The two tests
demonstrate the original vulnerability and check the prototype's record isolation.

## Regenerate manuscript Fig. 5

Run `python figures/fig5/generate_fig5.py`. The self-contained source-data bundle
replots the ten attack panels in a consistent style and exports PDF, SVG and
600 dpi PNG/TIFF. It preserves the archived numerical values and missing entries;
this plot regeneration does not rerun the experiments.

## Scientific limitations

- Original-map descriptor collisions were observed among different point maps.
- Public logo, zero-watermark and permutation metadata reveal the binary template.
- High recovered-logo NC does not establish a low false-positive rate.
- Six-map robustness averages do not establish source-held-out generalization.
- Training, independent threshold calibration, all baselines and all ablations
  remain unverified in this audit.

Do not describe these limitations as resolved by packaging or code publication.
