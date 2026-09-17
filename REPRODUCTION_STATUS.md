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

The released wrapper fixes registration time and sorts input files. Its six-map
compound NC mean is 0.914331; two consecutive runs produced byte-identical result
CSVs. This is an explicitly controlled reproduction variant, not an exact replay
of the historical run, whose registration time was not fixed. The earlier audit
with wall-clock registration gave 0.913014 and the unmodified release rerun gave
0.913640. All means round to 0.91.

The synthetic training smoke test passed for early and late loss schedules with
finite reported losses and model weights and a nonzero parameter update. No
original-data training claim follows from this smoke test.
