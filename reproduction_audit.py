"""Read-only audit of existing VGCN weights on original test maps.

Outputs go to ReproductionAudit; no historical experiment files are changed.
This is checkpoint inference, not a claim of training reproduction.
"""
import csv
import hashlib
import json
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
    files = sorted((ROOT / 'convertToGeoJson/GeoJson/TestSet').glob('*.geojson'))
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
    report = {'mode':'existing_checkpoint_original_map_inference', 'python':platform.python_version(),
              'torch':torch.__version__, 'device':device, 'expected_maps':len(files),
              'completed_maps':len(rows), 'failures':failures,
              'checkpoint_sha256':hashlib.sha256(f.MODEL_PATH.read_bytes()).hexdigest(),
              'scaler_sha256':hashlib.sha256(f.GLOBAL_SCALER_PATH.read_bytes()).hexdigest(),
              'impostor_pairs':len(pairs),
              'identical_descriptor_pairs':sum(p['bit_agreement']==1 for p in pairs),
              'note':'No calibrated threshold, independent validation, attack ROC or retraining yet.'}
    (out / 'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
