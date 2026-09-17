"""Run portable checkpoint checks; full retraining is a separate, unverified workflow."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Only synthetic training and Fig. 5; no map inputs required')
    args = parser.parse_args()
    scripts = ['smoke_train.py', 'figures/fig5/generate_fig5.py']
    if not args.quick:
        scripts = ['verify_inputs.py'] + scripts + ['reproduction_audit.py', 'reproduce_compound.py', 'key_scrambling_audit.py']
    out = ROOT / 'ReproductionAudit/portable_run'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'run_report.json'
    report = {'status': 'running', 'quick': args.quick, 'python_executable': sys.executable, 'full_retraining': False, 'steps': []}
    env = os.environ.copy()
    env.update({'MPLBACKEND': 'Agg', 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    for script in scripts:
        print(f'Running {script}', flush=True)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        log = out / (Path(script).stem + '.log')
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / script)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report['steps'].append({'script': script, 'exit_code': result.returncode, 'log': log.relative_to(ROOT).as_posix()})
        if result.returncode:
            report['status'] = 'failed'
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            raise SystemExit(f'Failed: {script}. Read {log}')
    if not args.quick:
        audit = json.loads((ROOT / 'ReproductionAudit/audit.json').read_text(encoding='utf-8'))
        report['inference'] = {key: audit[key] for key in ['completed_maps', 'failures', 'identical_descriptor_pairs', 'accepted_impostors', 'same_record_directed_impostor_trials']}
        if audit['completed_maps'] != 43 or audit['failures']:
            report['status'] = 'failed'
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            raise SystemExit('Inference did not finish all 43 test maps')
        with (ROOT / 'ReproductionAudit/compound_clean_reference.csv').open(encoding='utf-8-sig', newline='') as stream:
            reference = {row['复合攻击(顺序)'].strip(): float(row['VGAT']) for row in csv.DictReader(stream) if row['类型'] == 'data'}
        with (ROOT / 'ReproductionAudit/compound/results/fig12_compound_seq_nc.csv').open(encoding='utf-8-sig', newline='') as stream:
            actual = {row['复合攻击(顺序)'].strip(): float(row['VGAT']) for row in csv.DictReader(stream) if row['类型'] == 'data'}
        match = actual.keys() == reference.keys() and all(abs(actual[key] - reference[key]) <= 1e-6 for key in reference)
        report['compound'] = {'scores': actual, 'matches_clean_reference_at_1e_minus_6': match}
        if not match:
            report['status'] = 'numerical_mismatch'
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            raise SystemExit('Compound scores differ from the clean reference. Inspect environment and logs.')
    report['status'] = 'passed'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
