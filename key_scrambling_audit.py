"""Test keyed feature permutations against a confirmed real-map collision.

Diagnostic simulation, not a production cryptographic construction. Public test
UUIDs/timestamps are used deliberately. No claim of key secrecy is made.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'zNC-Test'))
import fig_common as f


def permutation(registration_id, timestamp, size):
    payload = json.dumps([registration_id, timestamp], separators=(',', ':')).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest(), 'big')
    return np.random.default_rng(seed).permutation(size)


def main():
    model, device = f.load_improved_gat_model('cuda' if torch.cuda.is_available() else 'cpu')
    names = ['H51-AANP', 'H51-BRGP']
    bits = []
    for name in names:
        data_dir = Path(os.environ.get('VGCN_TEST_DATA', str(ROOT / 'convertToGeoJson/GeoJson/TestSet')))
        frame = gpd.read_file(data_dir / (name+'.geojson'))
        bits.append(f.extract_features_from_graph(f.gdf_to_graph(frame), model, device).ravel())
    a, b = bits
    assert np.array_equal(a, b), 'Expected audited collision did not reproduce'
    logo = f.load_cat32().ravel()
    pa = permutation('00000000-0000-4000-8000-000000000001', 1789600000, len(a))
    pb = permutation('00000000-0000-4000-8000-000000000002', 1789600001, len(a))
    za, zb = a[pa] ^ logo, b[pb] ^ logo
    recovered_correct_protocol = za ^ b[pa]
    recovered_wrong_protocol = za ^ b[pb]
    # Compute the order of the exact integer Arnold matrix modulo 32.
    matrix = np.array([[1,1],[1,2]], dtype=np.int64)
    power = np.eye(2, dtype=np.int64)
    period = None
    for i in range(1, 1025):
        power = (power @ matrix) % 32
        if np.array_equal(power, np.eye(2,dtype=np.int64)):
            period = i
            break
    report = {
        'maps':names,
        'raw_bit_agreement':float(np.mean(a == b)),
        'different_key_zero_watermark_nc':f.calc_nc(za,zb),
        'impostor_recovery_nc_using_claimed_record_key':f.calc_nc(logo,recovered_correct_protocol),
        'impostor_recovery_nc_using_query_own_key_INCORRECT_PROTOCOL':f.calc_nc(logo,recovered_wrong_protocol),
        'same_key_permutation_preserves_hamming':bool(np.count_nonzero(a != b)==np.count_nonzero(a[pa] != b[pa])),
        'arnold_matrix_period_mod_32':period,
        'interpretation':'Different-key registration appearance can differ while same-record impostor verification still accepts a collided descriptor. Arnold iteration count alone gives only 24 distinct permutations at N=32.',
        'reference':'Author-provided 2026-0166__PDFA.pdf, Section 2.3, PDF page 6. The reference permutes the logo; this diagnostic tests the proposed feature-permutation variation.'
    }
    output = ROOT / 'ReproductionAudit/key_scrambling_audit.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
