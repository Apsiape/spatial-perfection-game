"""Verify recorded checksums and run all finite checks sequentially."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='Directory for fresh logs and results')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for line in (ROOT/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    print('CHECKSUMS: all recorded ancillary files verified', flush=True)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
        env[name] = '1'
    for name in ('BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[name] = '1'
    env['TEMP'] = env['TMP'] = str(args.output.resolve())
    env['GAME_CHECK_OUTPUT'] = str(args.output)
    import psutil
    cases = [('game', ['--output', str(args.output/'game.json')]),
             ('checks', ['--output', str(args.output/'results.json')])]
    cases += [(name, []) for name in ('fms_core', 'check_bridge', 'check_rsl',
                                     'gamma_chain', 'exact_checks',
                                     'check_extraction', 'check_descent')]
    cases += [(name, []) for name in ('energy_checks', 'field_checks',
              'faithfulness_checks', 'semifinite_checks',
              'semifinite_model_checks', 'fingerprint_checks', 'ladder_checks')]
    records = []
    for name, extra in cases:
        command = [sys.executable, '-B', '-u', str(ROOT/(name+'.py')), *extra]
        start = time.monotonic()
        peak = 0
        with (args.output/(name+'.log')).open('w', encoding='utf-8', newline='\n') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, cwd=args.output)
            process = psutil.Process(proc.pid)
            exceeded = False
            while proc.poll() is None:
                try:
                    rss = psutil.Process().memory_info().rss + process.memory_info().rss
                    rss += sum(p.memory_info().rss for p in process.children(recursive=True))
                    peak = max(peak, rss)
                    if rss >= 850_000_000:
                        proc.kill()
                        exceeded = True
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.05)
            code = proc.wait()
        row = dict(script=name+'.py', exit_code=code,
                   sampled_peak_combined_rss_MB=round(peak/1e6, 3),
                   seconds=round(time.monotonic()-start, 3),
                   memory_ceiling_triggered=exceeded)
        records.append(row)
        (args.output/'resources.json').write_text(json.dumps(records, indent=2)+'\n',
                                                 encoding='utf-8', newline='\n')
        print(json.dumps(row), flush=True)
        if code or exceeded:
            print((args.output/(name+'.log')).read_text(encoding='utf-8'), flush=True)
            raise SystemExit(code or 1)
    assert (args.output/'game.json').read_bytes() == (ROOT/'game.json').read_bytes()
    print('ALL FINITE CHECKS PASSED', flush=True)

if __name__ == '__main__':
    main()
