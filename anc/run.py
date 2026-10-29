"""Verify checksums and run all finite checks sequentially."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='Directory for fresh logs and results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for line in (ROOT/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    print('CHECKSUMS: all recorded ancillary files verified', flush=True)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
        env[name] = '1'
    cases = [
        ('game', [sys.executable, '-B', str(ROOT/'game.py'), '--output', str(args.output/'game.json')]),
        ('checks', [sys.executable, '-B', str(ROOT/'checks.py'), '--output', str(args.output/'results.json')])
    ]
    for name, command in cases:
        result = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
        (args.output/(name+'.log')).write_text(result.stdout + result.stderr, encoding='utf-8', newline='\n')
        print(result.stdout, end='', flush=True)
    assert (args.output/'game.json').read_bytes() == (ROOT/'game.json').read_bytes()
    print('ALL FINITE CHECKS PASSED', flush=True)

if __name__ == '__main__':
    main()
