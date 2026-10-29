"""Independent scalar/operator checks; imports no supplied executable code."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
import math
from fractions import Fraction as Q
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get('GAME_CHECK_OUTPUT', '.')).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261028)
out = {}
a = math.log(2)

def record(name, value):
    out[name] = value
    print(name, json.dumps(value), flush=True)

def geom_moments(r):
    return r/(1-r), r/(1-r)**2

# J = 2G+B, G geometric on {0,1,...}, B Bernoulli(1/9), independent.
mg, vg = geom_moments(Q(1, 64))
meanj = 2*mg + Q(1, 9)
varj = 4*vg + Q(8, 81)
assert meanj == Q(1, 7) and varj == Q(8, 49)
mg2, vg2 = geom_moments(Q(1, 4))
var_release = 4*vg2 + Q(8, 81)
mean_b = Q(5, 7)*2*Q(1, 4)/(1-Q(1, 4))**2
second_b = Q(5, 7)*4*Q(1, 4)*(1+Q(1, 4))/(1-Q(1, 4))**3
var_b = second_b-mean_b**2
assert var_release == Q(152, 81)
assert var_b == Q(6800, 3969) and var_release-varj == var_b
# Independently count the released copy-two boundary: 16 A at k=0, 15 thereafter.
for k in range(1, 30):
    assert (15*Q(1, 24)/4**k)/Q(7, 8) == Q(5, 7)/4**k
assert (16*Q(1, 24))/Q(7, 8) == Q(16, 21)
# Exact power-series product, using counted boundary rather than division.
def pj(n):
    return (Q(7, 8) if n%2 == 0 else Q(7, 64))/64**(n//2)
def pb(n):
    return Q(16, 21) if n == 0 else (Q(0) if n%2 else Q(5, 7)/4**(n//2))
for n in range(100):
    pm = (Q(2, 3) if n%2 == 0 else Q(1, 12))/4**(n//2)
    assert sum(pj(j)*pb(n-j) for j in range(n+1)) == pm
record('FP_exact', dict(mean_J=str(meanj), variance_J=str(varj),
    variance_boundary=str(var_b), variance_release=str(var_release),
    C_Gamma=float(varj)*a*a, Delta=math.log(7/6)-a/7,
    gap_12=a/7+math.log(14/15)))

# Arbitrary boundary laws: direct mixtures of ladders and Renyi/variance tests.
viol = 0
for trial in range(400):
    nb = int(rng.integers(1, 10))
    b = rng.normal(size=nb)*rng.uniform(0, 3)
    w = rng.dirichlet(np.ones(nb))
    js = np.arange(100)
    ps = np.array([float(pj(int(j))) for j in js])
    ks = b[:, None]+a*js
    ws = w[:, None]*ps
    mean = float(np.sum(ws*ks))
    var = float(np.sum(ws*(ks-mean)**2))
    vb = float(np.sum(w*(b-np.sum(w*b))**2))
    gap01 = math.log(float(np.sum(ws*np.exp(ks))))-mean
    gap12 = mean+math.log(float(np.sum(ws*np.exp(-ks))))
    assert abs(var-(vb+float(varj)*a*a)) < 2e-12
    if gap01 < math.log(7/6)-a/7-1e-12 or gap12 < a/7+math.log(14/15)-1e-12:
        viol += 1
assert viol == 0
record('FP_boundary_mixtures', dict(trials=400, violations=viol))

out['status']='PASS'
(ROOT/'fingerprint-checks.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print('ALL FINGERPRINT CHECKS PASSED',flush=True)
