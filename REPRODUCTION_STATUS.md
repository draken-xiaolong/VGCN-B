# Reproduction status, 2026-09-17

Verified: existing-checkpoint inference on 43/43 original test maps; 903 different-map pairs, including 11 identical binary-descriptor pairs. Six-map compound-attack mean NC: 0.913014, versus archived 0.912462 (both round to 0.91). This is not a 43-map compound result or fresh-training reproduction.

All 169 packaged Python sources passed syntax parsing. Two Solidity security regression tests passed using local solc 0.8.26. The original registry allows cross-account record replacement. The additional append-only prototype prevents overwriting an existing record and isolates claimant namespaces. It does not prove copyright ownership or prevent competing claims; cloud scripts still use the original interface.

Unverified: fresh training, all baseline runs, ablations, calibrated FPR/FNR, source-held-out generalization, and live cloud/ledger operation. Original data source and redistribution documentation remain prerequisites.

The inference checkpoint omits optimizer state. Its file hash differs from the original; parameter tensors are checked for equality. The audit JSON records the original checkpoint hash. The scaler was serialized with scikit-learn 1.6.1 and loaded under 1.7.1 with a version warning.

Point geometries have constant initial features under the original 13-feature design; graph topology did not prevent all descriptor collisions. Public logo, zero-watermark and permutation metadata reveal the binary template. Scrambling is not cryptographic template protection.
