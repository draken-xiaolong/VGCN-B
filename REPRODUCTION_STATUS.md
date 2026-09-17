# Reproduction status, 2026-09-17

Verified: existing-checkpoint inference on 43/43 original test maps; 903 different-map pairs, including 11 identical binary-descriptor pairs. Six-map compound-attack mean NC: 0.913014, versus archived 0.912462 (both round to 0.91). This is not a 43-map compound result or fresh-training reproduction.

All 169 packaged Python sources passed syntax parsing. Two Solidity security regression tests passed using local solc 0.8.26. The original registry allows cross-account record replacement. The additional append-only prototype prevents overwriting an existing record and isolates claimant namespaces. It does not prove copyright ownership or prevent competing claims; cloud scripts still use the original interface.

Unverified: fresh training, all baseline runs, ablations, calibrated FPR/FNR, source-held-out generalization, and live cloud/ledger operation. Original data source and redistribution documentation remain prerequisites.

The inference checkpoint omits optimizer state. Its file hash differs from the original; parameter tensors are checked for equality. `checkpoint_provenance.json` records both hashes. The audit JSON records the checkpoint used for that run. The scaler was serialized with scikit-learn 1.6.1 and loaded under 1.7.1 with a version warning.

Point geometries have constant initial features under the original 13-feature design; graph topology did not prevent all descriptor collisions. Public logo, zero-watermark and permutation metadata reveal the binary template. Scrambling is not cryptographic template protection.

An expanded audit uses each claimed registration's key for every other original
test map, with fixed registration timestamp 1789600000. At the original nominal
threshold NC >= 0.90, 369 of 1806 directed impostor trials are accepted (20.43%).
These dependent trials describe this finite collection; this is not an independent
deployment estimate or a calibrated ROC analysis. Selected scores were independently
checked using the original explicit scrambling and inverse-scrambling routines.

The feature-permutation counterexample produces different-key zero-watermark NC
0.530848 but impostor recovery NC 1.000 under the claimed record's key. Fixed
Arnold iteration counts provide only 24 different permutations for 32x32 inputs
with the matrix used by this implementation.

Migration verification found that the original graph converter reused existing
graph caches solely by file count, even after attack inputs changed. Therefore
the earlier 0.914331 mean and repeated identical CSVs did not demonstrate a clean
reproduction. The wrapper now rebuilds its generated graphs on every invocation,
fixes registration time, sorts inputs, and checks all six results. Clean CPU and
GPU runs both give mean NC 0.902804 with identical six-decimal result CSVs.
This differs from the historical 0.912462 and does not reproduce the original
rounded 0.91. Earlier wall-clock/cached runs are retained as audit history only.

The synthetic training smoke test passed for early and late loss schedules with
finite reported losses and model weights and a nonzero parameter update. No
original-data training claim follows from this smoke test.

## Fresh-environment migration check (2026-09-17)

A fresh GitHub clone on another drive and a newly installed isolated Python 3.13
environment (no system site packages) passed the portable workflow on CPU.
All 98 original inputs/resources matched SHA-256 checks. Inference completed all
43 maps with the same 11 descriptor collisions and 369/1806 accepted impostors.
The six rebuilt compound results matched a separately rebuilt GPU run byte for
byte (CSV precision: six decimal places). Training smoke, key diagnostic and
Fig. 5 generation passed. Fresh `npm ci` and the two local registry tests passed.
The full dependency lock and machine-readable `migration_validation.json` are
included. This emulates migration on the same physical host; it is not a test on
a second computer and does not establish full training or baseline reproduction.
