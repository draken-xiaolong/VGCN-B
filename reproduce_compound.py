"""Run the original six-map compound experiment into isolated output paths."""
import importlib.util
import os
import sys
import json
import csv
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, str(ROOT / 'zNC-Test'))
spec = importlib.util.spec_from_file_location('compound', ROOT / 'zNC-Test/Fig12.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.DIR_VECTOR_GEOJSON = Path(os.environ.get('VGCN_COMPOUND_DATA', str(module.DIR_VECTOR_GEOJSON)))
if not list(module.DIR_VECTOR_GEOJSON.glob('*.geojson')):
    raise FileNotFoundError(f'No compound-test input maps: {module.DIR_VECTOR_GEOJSON}')
OUT = ROOT / 'ReproductionAudit/compound'
module.SCRIPT_DIR = OUT
module.DIR_VECTOR_GEOJSON_ATTACKED = OUT / 'attacked_geojson'
module.DIR_GRAPH = OUT / 'graphs'
module.DIR_GRAPH_ORIGINAL = OUT / 'graphs/Original'
module.DIR_GRAPH_ATTACKED = OUT / 'graphs/Attacked/compound_seq'
module.DIR_ZEROWM = OUT / 'zero_watermarks'
module.DIR_RESULTS = OUT / 'results'
registration_timestamp = int(os.environ.get('VGCN_REGISTRATION_TIMESTAMP', '1789600000'))
original_key_generator = module.generate_scramble_key
module.generate_scramble_key = lambda name: original_key_generator(name, timestamp=registration_timestamp)
if __name__ == '__main__':
    originals = sorted(p for p in module.DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._'))
    if len(originals) != 6:
        raise ValueError(f'Expected six compound-test maps, found {len(originals)}')
    # The historical converter skips graph folders that already contain the
    # expected file count, even when the regenerated attack data have changed.
    # Remove only this wrapper's generated graphs, after resolving and checking
    # the complete path; never reuse graphs from a different input order/run.
    graph_dir = module.DIR_GRAPH.resolve()
    expected_parent = (ROOT / 'ReproductionAudit/compound').resolve()
    if not expected_parent.is_relative_to(ROOT.resolve()) or graph_dir.parent != expected_parent or graph_dir.name != 'graphs':
        raise RuntimeError(f'Unsafe generated-graph directory: {graph_dir}')
    if graph_dir.exists():
        shutil.rmtree(graph_dir)
    attacked = module.step3_generate_compound_seq_attacks(originals)
    if len(attacked) != 6:
        raise RuntimeError('Not all six attacks were generated')
    module.step4_convert_to_graph(originals, attacked)
    if len(list(module.DIR_GRAPH_ORIGINAL.glob('*_graph.pkl'))) != 6 or len(list(module.DIR_GRAPH_ATTACKED.rglob('*_graph.pkl'))) != 6:
        raise RuntimeError('Graph conversion did not produce six originals and six attacks')
    module.step5_generate_zero_watermark()
    module.step6_evaluate_nc()
    with (module.DIR_RESULTS / 'fig12_compound_seq_nc.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    values = [float(row['VGAT']) for row in rows if row['类型'] == 'data']
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise RuntimeError('Compound evaluation did not produce six finite scores')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'protocol.json').write_text(json.dumps({
        'registration_timestamp': registration_timestamp,
        'attack_seed': 42,
        'input_order': [p.name for p in originals],
        'graph_cache_policy': 'rebuild on every invocation',
        'completed_maps': len(values),
        'mean_nc_from_csv': sum(values) / len(values),
        'note': 'Fixed public timestamp is for reproducibility, not secret-key security.'
    }, indent=2), encoding='utf-8')
