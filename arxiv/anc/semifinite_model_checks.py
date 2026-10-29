"""Single-threaded scalar and sparse checks of the new type-II construction.

Run with python -u -B. All writes are confined to this script's directory.
Numerics check finite truncations; they do not prove the infinite theorem.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[key] = "1"
import ctypes
import itertools
import json
import math
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from scipy import sparse as ss
import sympy as sp

OUT = Path(os.environ.get('GAME_CHECK_OUTPUT', '.')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = Path(__file__).resolve().parent/'game.json'

# Independent exact scalar verification, including the sharp entropy constant.
k = sp.symbols("k", integer=True, nonnegative=True)
tu = sp.Rational(3, 4) * 16**(-k)
tv = sp.Rational(3, 16) * 16**(-k)
hu = sp.Rational(7, 6) * 4**(-k)
hv = sp.Rational(7, 12) * 4**(-k)
assert sp.summation(tu + tv, (k, 0, sp.oo)) == 1
assert sp.summation(tu * hu + tv * hv, (k, 0, sp.oo)) == 1
assert sp.simplify(tu - 4*tv) == 0
assert sp.simplify(tv - 4*tu.subs(k, k+1)) == 0
assert sp.simplify(hu - 2*hv) == 0
assert sp.simplify(hv - 2*hu.subs(k, k+1)) == 0
delta = sp.log(sp.Rational(7, 6)) - sp.log(2)/7
series_delta = (sp.Rational(8, 9)*sp.log(sp.Rational(7, 6))
                + sp.Rational(1, 9)*sp.log(sp.Rational(7, 12))
                - sp.log(4)/63)
assert sp.simplify(series_delta-delta) == 0
A, B, D = sp.symbols("A B D")
expr_D = 72*A + sp.Rational(17, 9)*sp.log(2)
expr_B = 63*A + sp.Rational(16, 9)*sp.log(2)
assert sp.simplify(expr_D - sp.Rational(8, 7)*expr_B + sp.log(2)/7) == 0

# Reconstruct the 5-dimensional ideal gadget without importing executable code.
I = np.eye(2)
X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])
Y = np.array([[0., -1j], [1j, 0.]])
kr = np.kron
square = [[kr(X, I), kr(I, X), kr(X, X)],
          [kr(I, Z), kr(Z, I), kr(Z, Z)],
          [kr(X, Z), kr(Z, X), kr(Y, Y)]]
square = [[m.real for m in row] for row in square]
signs = list(itertools.product((1, -1), repeat=3))
rows = [s for s in signs if math.prod(s) == 1]
cols = [[s for s in signs if math.prod(s) == eta] for eta in (1, 1, -1)]
c = [np.array([1, 0, 1, 0])/math.sqrt(2),
     np.array([0, 1, 0, 1])/math.sqrt(2)]
r = [np.array([1, 1, 1, 1])/2, np.array([1, -1, 1, -1])/2]
dist = [(1, 1, 1), (1, -1, -1)]

def proj(mats, s):
    p = np.eye(4)
    for m, t in zip(mats, s):
        p = p @ (np.eye(4)+t*m)/2
    return p

def embed(p):
    e = np.zeros((5, 5)); e[1:, 1:] = p
    return e

flag = np.diag([1., 0., 0., 0., 0.])
ga, gb = {}, {}
for i in range(3):
    ga[f"R{i+1}"] = {s: embed(proj(square[i], s)) for s in rows}
    ga[f"R{i+1}"]["f"] = flag
    gb[f"C{i+1}"] = {s: embed(proj([square[j][i] for j in range(3)], s))
                      for s in cols[i]}
    gb[f"C{i+1}"]["f"] = flag
for j in range(2):
    ga[f"T{j}"] = {s: embed(proj([square[i][0] for i in range(3)], s))
                     for s in cols[0] if s != dist[j]}
    gb[f"U{j}"] = {s: embed(proj(square[0], s)) for s in rows if s != dist[j]}
    for t in (1, -1):
        va = np.r_[1., t*c[j]] / math.sqrt(2)
        vb = np.r_[1., t*r[j]] / math.sqrt(2)
        ga[f"T{j}"][t] = np.outer(va, va)
        gb[f"U{j}"][t] = np.outer(vb, vb)
for ms in [*ga.values(), *gb.values()]:
    assert np.max(np.abs(sum(ms.values())-np.eye(5))) < 1e-14
    assert all(np.linalg.eigvalsh(e).min() > -1e-14 for e in ms.values())

table = json.loads(SOURCE.read_text(encoding="utf-8"))
unpack = lambda a: tuple(a) if isinstance(a, list) else a

def finite(K):
    ambient = 4*16**K
    Us, Vs = [], []
    densities = []
    pos = 0
    for j in range(K):
        nu = 3*16**(K-j)
        nv = 12*16**(K-j-1)
        Us.append(list(range(pos, pos+nu))); pos += nu
        Vs.append(list(range(pos, pos+nv))); pos += nv
        densities += [7/6*4.**(-j)]*nu + [7/12*4.**(-j)]*nv
    n = pos
    assert n == 4*(16**K-1)
    b1, b2 = [], []
    for j in range(K):
        b1 += [[f, *Us[j][4*l:4*l+4]] for l, f in enumerate(Vs[j])]
        flags = Us[j+1] if j+1 < K else [None]*(len(Vs[j])//4)
        b2 += [[f, *Vs[j][4*l:4*l+4]] for l, f in enumerate(flags)]
    boundary = ss.csr_matrix((np.ones(len(Us[0])), (Us[0], Us[0])), shape=(n, n))

    def lift(e, blocks):
        rr, cc, vv = [], [], []
        nz = list(zip(*np.nonzero(e)))
        for block in blocks:
            for a, b in nz:
                x, y = block[a], block[b]
                if x is not None and y is not None:
                    rr.append(x); cc.append(y); vv.append(e[a, b])
        return ss.csr_matrix((vv, (rr, cc)), shape=(n, n))

    measurements = {"alice": {}, "bob": {}}
    for side, gadget in (("alice", ga), ("bob", gb)):
        for name, answers in table[side].items():
            if name == "label":
                ui = [i for group in Us for i in group]
                vi = [i for group in Vs for i in group]
                ms = {"A": ss.csr_matrix((np.ones(len(ui)), (ui, ui)), shape=(n, n)),
                      "B": ss.csr_matrix((np.ones(len(vi)), (vi, vi)), shape=(n, n))}
            else:
                h, q = name.split(":")
                ms = {a: lift(e, b1 if h == "1" else b2) for a, e in gadget[q].items()}
                if h == "2": ms["bot"] = boundary
            measurements[side][name] = [ms[unpack(a)] for a in answers]
    comp = 0.
    for family in measurements.values():
        for ms in family.values():
            err = sum(ms)-ss.eye(n, format="csr")
            if err.nnz: comp = max(comp, float(np.max(np.abs(err.data))))
    assert comp < 1e-13
    lam = np.array(densities)
    retained = 1-64.**(-K)
    assert abs(lam.sum()/ambient-retained) < 1e-14
    s = np.sqrt(lam/(ambient*retained))
    losses, minimum_probability = {}, 0.
    for pair in table["supported_pairs"]:
        am = measurements["alice"][pair["alice"]]
        bm = measurements["bob"][pair["bob"]]
        loss = 0.
        for ai, bi in pair["forbidden"]:
            m = am[ai].multiply(bm[bi]).tocoo()
            pr = float(np.dot(m.data, s[m.row]*s[m.col]))
            minimum_probability = min(minimum_probability, pr)
            loss += pr
        losses[f"{pair['alice']},{pair['bob']}"] = loss
    actual = sum(losses.values())/52
    expected = float(Q(7, 208*(64**K-1)))
    assert abs(actual-expected) < 1e-14
    assert minimum_probability > -1e-14
    entropy = -float(np.dot(lam, np.log(lam/retained)))/(ambient*retained)
    return dict(K=K, matrix_dimension=n, retained_trace=1-16.**(-K),
                retained_probability=retained, observed_loss=actual,
                exact_loss=str(Q(7, 208*(64**K-1))),
                entropy_relative_to_original_trace=entropy,
                max_completeness_error=comp, per_pair_losses=losses)

cases = [finite(K) for K in (1, 2)]
spatial_entropy_bits = math.log2(24)+7/9
result = dict(
    delta_nats=float(delta.evalf()), delta_bits=float((delta/sp.log(2)).evalf()),
    delta_expression="ln(7/6)-ln(2)/7",
    spatial_entropy_bits=spatial_entropy_bits,
    spatial_entropy_nats=spatial_entropy_bits*math.log(2),
    trace_strategy_loss_upper_bound=(3-2*math.sqrt(2))/208,
    exact_scalar_checks="normalization, traces, density ratios, entropy elimination passed",
    finite_cases=cases,
)
if os.name == "nt":
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    *[(name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
    pm = PMC(); pm.cb = ctypes.sizeof(pm)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = ctypes.c_void_p
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = (ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong)
    assert get_info(process(), ctypes.byref(pm), ctypes.sizeof(pm))
    result["peak_working_set_bytes"] = pm.PeakWorkingSetSize
    result["peak_commit_bytes"] = pm.PeakPagefileUsage
    assert max(pm.PeakWorkingSetSize, pm.PeakPagefileUsage) < 500_000_000
(OUT/"type-ii-checks.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
for key, val in result.items():
    if key == "finite_cases":
        for case in val:
            print({k: v for k, v in case.items() if k != "per_pair_losses"}, flush=True)
    else:
        print(key, val, flush=True)
print("ALL TYPE-II CONSTRUCTION CHECKS PASSED", flush=True)
