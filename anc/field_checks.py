"""Finite field-resource checks; no imports of the game's Python implementations.

Reads only the rational game table. All writes stay in this script's directory.
Finite calculations supplement the analytic arguments in the manuscript.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
from fractions import Fraction
from itertools import product
from math import comb
import hashlib
import json
import sys
import numpy as np
import sympy as s

OUT = Path(os.environ.get('GAME_CHECK_OUTPUT', '.')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path(__file__).resolve().parent.parent
table = json.loads((SRC/'anc/game.json').read_text(encoding='utf-8'))
I, X, Z = s.eye(2), s.Matrix([[0, 1], [1, 0]]), s.diag(1, -1)
Y = s.Matrix([[0, -s.I], [s.I, 0]])
tensor = s.kronecker_product
square = [[tensor(X,I), tensor(I,X), tensor(X,X)],
          [tensor(I,Z), tensor(Z,I), tensor(Z,Z)],
          [tensor(X,Z), tensor(Z,X), tensor(Y,Y)]]
flag = s.diag(1,0,0,0,0)
cs = [s.Matrix([1,0,1,0])/s.sqrt(2), s.Matrix([0,1,0,1])/s.sqrt(2)]
rs = [s.Matrix([1,1,1,1])/2, s.Matrix([1,-1,1,-1])/2]

def key(answer):
    return tuple(answer) if isinstance(answer, list) else answer

def base_projector(tag, answer, alice):
    """Construct directly in the answer ordering of the supplied JSON."""
    if answer == 'f':
        return flag
    typ, idx = tag[0], int(tag[1:])
    if isinstance(answer, int):
        v = s.Matrix([1, *list(answer*(cs[idx] if alice else rs[idx]))])
        return s.simplify(v*v.T/2)
    if typ == 'R':
        matrices = square[idx-1]
    elif typ == 'C':
        matrices = [square[i][idx-1] for i in range(3)]
    elif typ == 'T':
        matrices = [square[i][0] for i in range(3)]
    else:
        matrices = square[0]
    out = s.eye(4)
    for val, op in zip(answer, matrices):
        out = out*(s.eye(4)+val*op)/2
    return s.diag(s.zeros(1), out)

base = {}
for side, alice in (('alice',True), ('bob',False)):
    for name, answers in table[side].items():
        if name == 'label':
            continue
        _, tag = name.split(':')
        for answer in answers:
            a = key(answer)
            if a != 'bot':
                base[alice, tag, a] = base_projector(tag,a,alice)

# Build genuinely infinite matrix rows from their disjoint blocks; no finite
# compression or artificial top boundary enters any row used below.
def oscillator_block(h, n):
    k, r = divmod(n,20)
    if h == 1:
        j = r//4 if r < 16 else r-16
        return [20*k+16+j] + [20*k+4*j+i for i in range(4)]
    if r >= 16:
        return [20*(k+1)] + [20*k+16+j for j in range(4)]
    if r == 0 and k > 0:
        return [20*k] + [20*(k-1)+16+j for j in range(4)]
    return None

def row(name, answer, alice, n):
    if name == 'label':
        match = ('A' if n%20 < 16 else 'B') == answer
        return {n:s.Integer(1)} if match else {}
    hs, tag = name.split(':')
    block = oscillator_block(int(hs), n)
    if block is None:
        return {n:s.Integer(1)} if answer == 'bot' else {}
    if answer == 'bot':
        return {}
    matrix = base[alice,tag,answer]
    rr = block.index(n)
    return {j:matrix[rr,cc] for cc,j in enumerate(block) if matrix[rr,cc] != 0}

def coeff(n):
    k,r = divmod(n,20)
    return s.Rational(1,2**k)/(s.sqrt(24) if r < 16 else s.sqrt(48))

row_cache = {}
def getrow(name, a, alice, n):
    q = (name,a,alice,n)
    if q not in row_cache:
        row_cache[q] = row(name,a,alice,n)
    return row_cache[q]

# Full PVM identities checked on rows of the infinite matrices, including the
# exceptional k=0 boundary and several translated periods.
for side,alice in (('alice',True),('bob',False)):
    for name,answers0 in table[side].items():
        answers = list(map(key,answers0))
        for n in range(60):
            summed = {}
            for a in answers:
                e = getrow(name,a,alice,n)
                for j,v in e.items():
                    summed[j] = s.simplify(summed.get(j,0)+v)
                    assert getrow(name,a,alice,j).get(n,0) == v
                for b in answers:
                    ep = {}
                    for ell,v in e.items():
                        for j,w in getrow(name,b,alice,ell).items():
                            ep[j] = ep.get(j,0)+v*w
                    target = e if a == b else {}
                    for j in set(ep)|set(target):
                        assert s.simplify(ep.get(j,0)-target.get(j,0)) == 0
            assert {j:v for j,v in summed.items() if v != 0} == {n:1}
print('Independent infinite-row PVM tests: 21 questions, rows 0..59 passed',flush=True)

per_pair = []
events = losing_events = zero_amplitudes = 0
for pair in table['supported_pairs']:
    q,r = pair['alice'], pair['bob']
    forbidden = {tuple(a) for a in pair['forbidden']}
    assert pair['probability'] == [1,52]
    pair_zeros = 0
    for ai,a0 in enumerate(table['alice'][q]):
        for bi,b0 in enumerate(table['bob'][r]):
            events += 1
            if (ai,bi) not in forbidden:
                continue
            losing_events += 1
            a,b = key(a0),key(b0)
            for n in range(60):
                amplitudes = {}
                for ell,v in getrow(q,a,True,n).items():
                    for j,w in getrow(r,b,False,ell).items():
                        amplitudes[j] = amplitudes.get(j,0)+v*coeff(ell)*w
                for amplitude in amplitudes.values():
                    assert s.simplify(amplitude) == 0, (q,r,a,b,n,amplitude)
                    pair_zeros += 1
    zero_amplitudes += pair_zeros
    per_pair.append(dict(alice=q,bob=r,forbidden=len(forbidden),zero_matrix_entries=pair_zeros))
assert len(per_pair) == 52 and events == 1547
print(f'Independent table-to-PVM check: {events} events; {losing_events} losing events; '
      f'{zero_amplitudes} nontrivial zero-amplitude checks on all 52 pairs',flush=True)

# Mutant tests: these must detect the two easiest gluing mistakes.
def bridge_loss(flag_coefficient, magic_coefficient):
    D = s.diag(flag_coefficient,*([magic_coefficient]*4))
    total = 0
    for x,y in product(range(2),repeat=2):
        for a,b in product((1,-1),repeat=2):
            if a*b != (-1)**(x*y):
                total += s.trace(base[True,f'T{x}',a]*D*base[False,f'U{y}',b]*D)
    return s.simplify(total)
assert bridge_loss(1,s.sqrt(2)) == 0
assert bridge_loss(1,1) == 3-2*s.sqrt(2)
assert bridge_loss(0,1) == 1
print('Mutation checks: wrong Schmidt ratio and missing offset flag both detected',flush=True)

# An independent rank-level contradiction follows already from equal ranks and
# the 16-fold scaling law; numerical ranks are not a no-go proof.
assert s.Rational(1,72)*16*4 == s.Rational(8,9)
assert s.Rational(8,9)-s.Rational(1,72) == s.Rational(7,8)

def enc(a,b,c):
    t = a+b+c
    return comb(t+2,3)+a*(t+1)-a*(a-1)//2+b
def dec(j):
    t = 0
    while comb(t+3,3) <= j:
        t += 1
    offset = j-comb(t+2,3)
    for a in range(t+1):
        count = t-a+1
        if offset < count:
            return a,offset,t-a-offset
        offset -= count
    raise AssertionError(j)
count = 0
for t in range(81):
    for a in range(t+1):
        for b in range(t-a+1):
            c = t-a-b
            j = enc(a,b,c)
            assert dec(j) == (a,b,c)
            for u in (0,1,7,100):
                assert j+u+1 <= (t+u+1)**3
            count += 1
print('Storage inverse and spectator bound:',count,'triples passed',flush=True)

# Coherent input with a nontrivial spectator: test the equality, rather than
# merely a diagonal probability example. Input amplitudes are rational.
amplitudes = {(0,0,0,0):Fraction(1,2), (0,0,0,1):Fraction(-1,3),
              (1,0,2,0):Fraction(2,5), (1,0,2,2):Fraction(1,7),
              (0,3,1,1):Fraction(-3,11), (2,0,0,2):Fraction(2,13)}
storage_moments = {}
norm = sum(abs(v)**2 for v in amplitudes.values())
for p in range(1,7):
    lhs = sum(abs(v)**2*(enc(a,b,c)+u+1)**p for (a,b,c,u),v in amplitudes.items())/norm
    rhs = sum(abs(v)**2*(a+b+c+u+1)**(3*p) for (a,b,c,u),v in amplitudes.items())/norm
    assert lhs <= rhs
    storage_moments[p] = str(lhs)

# Energy estimate stress test: a finite smooth span that is NOT h-invariant.
h = np.array([[2.,1.,0.],[1.,3.,1.],[0.,1.,4.]])
f = np.eye(3)[:,:2]
assert np.linalg.norm((np.eye(3)-f@f.T)@h@f) > 0
energy_ratios = []
for k in range(1,5):
    embed = f.copy()
    for _ in range(k-1): embed = np.kron(embed,f)
    H = np.zeros((3**k,3**k))
    for pos in range(k):
        op = np.array([[1.]])
        for i in range(k): op = np.kron(op,h if i == pos else np.eye(3))
        H += op
    for p in range(1,5):
        Ap = max([1.] + [np.linalg.norm(np.linalg.matrix_power(h,r)@f,2)**(1/r) for r in range(1,p+1)])
        observed = np.linalg.norm(np.linalg.matrix_power(H,p)@embed,2)
        bound = Ap**p*k**p
        assert observed <= bound*(1+1e-12)
        energy_ratios.append(observed/bound)
print('Non-invariant-span energy stress test: 16 sector/power inequalities passed',flush=True)

# Finite canonical pairs and explicit symplectic completion in a 6-mode model.
rng = np.random.default_rng(20261028)
mat = rng.normal(size=(6,6))
h6 = mat.T@mat+2*np.eye(6)
vals,vec = np.linalg.eigh(h6)
hp = vec@np.diag(vals**.5)@vec.T
hm = vec@np.diag(vals**-.5)@vec.T
J = np.block([[np.zeros((6,6)),np.eye(6)],[-np.eye(6),np.zeros((6,6))]])
omega = lambda a,b: a@J@b
pairs = [(np.r_[hm[:,i],np.zeros(6)],np.r_[np.zeros(6),hp[:,i]]) for i in range(3)]
def orthogonalize(w):
    w = w.copy()
    for q,p in pairs:
        w -= omega(w,p)*q-omega(w,q)*p
    return w
for _ in range(3):
    candidates = [orthogonalize(w) for w in np.eye(12)]
    q = max(candidates,key=np.linalg.norm)
    q = q/np.linalg.norm(q)
    p = max(candidates,key=lambda w:abs(omega(q,w)))
    p = p/omega(q,p)
    pairs.append((q,p))
S = np.column_stack([v for pair in pairs for v in pair])
Jpairs = np.kron(np.eye(6),[[0.,1.],[-1.,0.]])
symplectic_error = np.max(np.abs(S.T@J@S-Jpairs))
assert symplectic_error < 1e-11
print('Finite canonical-completion stress test error:',symplectic_error,flush=True)

import psutil
from types import SimpleNamespace
usage = psutil.Process().memory_info()
mem = SimpleNamespace(PeakWorkingSetSize=getattr(usage,'peak_wset',usage.rss),
                      PeakPagefileUsage=getattr(usage,'peak_pagefile',usage.rss))
assert max(mem.PeakWorkingSetSize, mem.PeakPagefileUsage) < 500_000_000
paths = [SRC/'main.tex',SRC/'anc/game.json',SRC/'anc/checks.py',
         Path(__file__),Path(__file__),Path(__file__)]
result = dict(status='passed',python=sys.version,events=events,losing_events=losing_events,
    zero_amplitude_entries=zero_amplitudes,per_pair=per_pair,storage_triples=count,
    storage_moments=storage_moments,energy_test_max_ratio=max(energy_ratios),
    symplectic_error=float(symplectic_error),peak_working_set_bytes=mem.PeakWorkingSetSize,
    peak_commit_bytes=mem.PeakPagefileUsage,compute_threads=1,
    sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    limitation='Finite tests supplement analytic proofs; no continuum discretization is a QFT proof.')
(OUT/'independent-checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('Peak working set / commit:',mem.PeakWorkingSetSize,mem.PeakPagefileUsage,flush=True)
print('ALL INDEPENDENT CHECKS PASSED',flush=True)
