"""Read-only audit of existing VGCN weights on original test maps.

Outputs go to ReproductionAudit; no historical experiment files are changed.
This is checkpoint inference, not a claim of training reproduction.
"""
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'zNC-Test'))
import fig_common as f


def main():
    out = ROOT / 'ReproductionAudit'
    out.mkdir(exist_ok=True)
    model, device = f.load_improved_gat_model('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(os.environ.get('VGCN_TEST_DATA', str(ROOT / 'convertToGeoJson/GeoJson/TestSet')))
    files = sorted(data_dir.glob('*.geojson'))
    if not files:
        raise FileNotFoundError(f'No original test GeoJSON maps found: {data_dir}')
    descriptors, rows, failures = [], [], []
    for path in files:
        print('AUDIT', path.name, flush=True)
        try:
            frame = gpd.read_file(path)
            graph = f.gdf_to_graph(frame)
            bits = f.extract_features_from_graph(graph, model, device).reshape(-1)
            descriptors.append(bits)
            rows.append({'name': path.stem, 'nodes': len(frame), 'ones': int(bits.sum()),
                         'geometry_types': ','.join(sorted(set(frame.geom_type))),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        except Exception as exc:
            failures.append({'name': path.name, 'error': repr(exc)})
    with (out / 'test_maps.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name','nodes','ones','geometry_types','sha256'])
        writer.writeheader()
        writer.writerows(rows)
    logo = f.load_cat32().reshape(-1)
    if descriptors:
        np.savez_compressed(out / 'descriptor_cache.npz', bits=np.stack(descriptors), names=np.array([r['name'] for r in rows]))
    pairs = []
    for i, a in enumerate(descriptors):
        for j in range(i + 1, len(descriptors)):
            b = descriptors[j]
            # Same fixed permutation on the registered and recovered logo is
            # immaterial to descriptor bit agreement. Report descriptor scores
            # directly; logo NC below uses an explicit identity permutation.
            pairs.append({'map_a': rows[i]['name'], 'map_b': rows[j]['name'],
                          'bit_agreement': float(np.mean(a == b)),
                          'descriptor_nc': f.calc_nc(a,b),
                          'identity_permutation_logo_nc': f.calc_nc(logo, logo ^ a ^ b)})
    if pairs:
        with (out / 'impostor_pairs.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    # Verification uses the key of the claimed registration, not the query's own key.
    # Evaluate the original Arnold scheme under a fixed public registration time.
    directed = []
    registration_time = 1789600000
    positions = np.arange(1024, dtype=np.int64).reshape(32,32)
    for i, a in enumerate(descriptors):
        key, _ = f.generate_scramble_key(rows[i]['name'], timestamp=registration_time)
        permuted_positions = f.scramble_image(positions, key).ravel()
        inverse = np.argsort(permuted_positions)
        for j, b in enumerate(descriptors):
            if i == j:
                continue
            recovered = logo ^ (a ^ b)[inverse]
            score = f.calc_nc(logo, recovered)
            directed.append({'claimed_map':rows[i]['name'], 'query_map':rows[j]['name'],
                             'recovered_logo_nc':score, 'accept_at_nominal_0_90':int(score>=0.90)})
    if directed:
        with (out / 'same_record_impostors.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(directed[0]))
            writer.writeheader()
            writer.writerows(directed)
    report = {'mode':'existing_checkpoint_original_map_inference', 'python':platform.python_version(),
              'torch':torch.__version__, 'device':device, 'expected_maps':len(files),
              'completed_maps':len(rows), 'failures':failures,
              'checkpoint_sha256':hashlib.sha256(f.MODEL_PATH.read_bytes()).hexdigest(),
              'scaler_sha256':hashlib.sha256(f.GLOBAL_SCALER_PATH.read_bytes()).hexdigest(),
              'impostor_pairs':len(pairs),
              'identical_descriptor_pairs':sum(p['bit_agreement']==1 for p in pairs),
              'same_record_directed_impostor_trials':len(directed),
              'nominal_threshold':0.90,
              'accepted_impostors':sum(p['accept_at_nominal_0_90'] for p in directed),
              'empirical_false_acceptance_fraction':sum(p['accept_at_nominal_0_90'] for p in directed)/len(directed) if directed else None,
              'registration_time':registration_time,
              'note':'No calibrated threshold, independent validation, attack ROC or retraining yet.'}
    (out / 'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
