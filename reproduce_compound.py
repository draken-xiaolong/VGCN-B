"""Run the original six-map compound experiment into isolated output paths."""
import importlib.util
import os
import sys
import json
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
    attacked = module.step3_generate_compound_seq_attacks(originals)
    module.step4_convert_to_graph(originals, attacked)
    module.step5_generate_zero_watermark()
    module.step6_evaluate_nc()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'protocol.json').write_text(json.dumps({
        'registration_timestamp': registration_timestamp,
        'attack_seed': 42,
        'input_order': [p.name for p in originals],
        'note': 'Fixed public timestamp is for reproducibility, not secret-key security.'
    }, indent=2), encoding='utf-8')
