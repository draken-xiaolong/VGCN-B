"""Run the original six-map compound experiment into isolated output paths."""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, str(ROOT / 'zNC-Test'))
spec = importlib.util.spec_from_file_location('compound', ROOT / 'zNC-Test/Fig12.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OUT = ROOT / 'ReproductionAudit/compound'
module.SCRIPT_DIR = OUT
module.DIR_VECTOR_GEOJSON_ATTACKED = OUT / 'attacked_geojson'
module.DIR_GRAPH = OUT / 'graphs'
module.DIR_GRAPH_ORIGINAL = OUT / 'graphs/Original'
module.DIR_GRAPH_ATTACKED = OUT / 'graphs/Attacked/compound_seq'
module.DIR_ZEROWM = OUT / 'zero_watermarks'
module.DIR_RESULTS = OUT / 'results'
if __name__ == '__main__':
    module.main()
