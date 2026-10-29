"""Independent audit checks. Writes only beside this script; one numerical thread.

No executable release or supplied-check code is imported. Exact local checks,
semantic-predicate checks, and a separate sparse probability calculation are used.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[name] = "1"
import ctypes
import hashlib
import itertools
import json
import math
import random
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import scipy.sparse as sparse
import sympy as s

OUT = Path(os.environ.get('GAME_CHECK_OUTPUT', '.')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path(__file__).resolve().parent.parent
TABLE = json.loads((SRC / "anc/game.json").read_text(encoding="utf-8"))
R = s.Rational
rt2 = s.sqrt(2)
I2, I4, I5 = s.eye(2), s.eye(4), s.eye(5)
pauli_x = s.Matrix([[0, 1], [1, 0]])
pauli_z = s.diag(1, -1)
pauli_y = s.Matrix([[0, -s.I], [s.I, 0]])
tensor = s.kronecker_product
square = [[tensor(pauli_x, I2), tensor(I2, pauli_x), tensor(pauli_x, pauli_x)],
          [tensor(I2, pauli_z), tensor(pauli_z, I2), tensor(pauli_z, pauli_z)],
          [tensor(pauli_x, pauli_z), tensor(pauli_z, pauli_x), tensor(pauli_y, pauli_y)]]
triples = list(itertools.product((1, -1), repeat=3))
chosen = [(1, 1, 1), (1, -1, -1)]
cv = [s.Matrix([1, 0, 1, 0])/rt2, s.Matrix([0, 1, 0, 1])/rt2]
rv = [s.Matrix([1, 1, 1, 1])/2, s.Matrix([1, -1, 1, -1])/2]
assert s.Matrix(2, 2, lambda x,y: (cv[x].T*rv[y])[0]) == s.Matrix([[1, 1], [1, -1]])/rt2
hadamard = s.Matrix([[1,1],[1,-1]])/rt2
conjugacy = tensor(hadamard,hadamard)
primary_square = [[tensor(pauli_z,I2),tensor(I2,pauli_z),tensor(pauli_z,pauli_z)],
                  [tensor(I2,pauli_x),tensor(pauli_x,I2),tensor(pauli_x,pauli_x)],
                  [tensor(pauli_z,pauli_x),tensor(pauli_x,pauli_z),tensor(pauli_y,pauli_y)]]
for i,j in itertools.product(range(3),repeat=2):
    assert s.simplify(conjugacy*square[i][j]*conjugacy.T-primary_square[i][j]) == s.zeros(4)
assert [s.simplify(conjugacy*c) for c in cv] == [s.Matrix([1,1,0,0])/rt2,s.Matrix([1,-1,0,0])/rt2]
assert [s.simplify(conjugacy*r) for r in rv] == [s.eye(4)[:,0],s.eye(4)[:,1]]

def encode(a):
    return tuple(a) if isinstance(a, list) else a

def base_win(x, y, a, b):
    if x[0] == "R" and y[0] == "C":
        if a == "f" or b == "f":
            return a == b
        return a[int(y[1])-1] == b[int(x[1])-1]
    if x[0] == "T" and y == "C1":
        if isinstance(a, int):
            return b == "f" or b == chosen[int(x[1])]
        return a == b
    if x == "R1" and y[0] == "U":
        if isinstance(b, int):
            return a == "f" or a == chosen[int(y[1])]
        return a == b
    if x[0] == "T" and y[0] == "U" and isinstance(a, int) and isinstance(b, int):
        return a*b == (-1)**(int(x[1])*int(y[1]))
    return True

def gamma_win(x, y, a, b):
    copy, x0 = x.split(":")
    if y == "label":
        cls = a != "f" if copy == "1" else a in ("f", "bot")
        return cls == (b == "A")
    copy_b, y0 = y.split(":")
    assert copy == copy_b
    if a == "bot" or b == "bot":
        assert copy == "2"
        return a == b
    return base_win(x0, y0, a, b)

event_count = 0
for pair in TABLE["supported_pairs"]:
    assert pair["probability"] == [1, 52]
    bad = {tuple(z) for z in pair["forbidden"]}
    for i, a in enumerate(TABLE["alice"][pair["alice"]]):
        for j, b in enumerate(TABLE["bob"][pair["bob"]]):
            assert ((i, j) not in bad) == gamma_win(pair["alice"], pair["bob"], encode(a), encode(b))
            event_count += 1
assert event_count == 1547

def joint(observables, answer):
    product = I4
    for observable, bit in zip(observables, answer):
        product = product * (I4 + bit*observable)/2
    return s.simplify(product)

def embed(p):
    return s.diag(s.zeros(1), p)

flag = s.diag(1, 0, 0, 0, 0)
measurements = {"alice": {}, "bob": {}}
for side in measurements:
    for name, answers in TABLE[side].items():
        if not name.startswith("1:"):
            continue
        q = name.split(":")[1]
        effects = {}
        for raw in answers:
            a = encode(raw)
            if a == "f":
                effect = flag
            elif q[0] == "R":
                effect = embed(joint(square[int(q[1])-1], a))
            elif q[0] == "C":
                effect = embed(joint([square[i][int(q[1])-1] for i in range(3)], a))
            elif isinstance(a, int):
                direction = cv[int(q[1])] if side == "alice" else rv[int(q[1])]
                vector = s.Matrix([1, *list(a*direction)]) / rt2
                effect = vector * vector.T
            else:
                observables = [square[i][0] for i in range(3)] if side == "alice" else square[0]
                effect = embed(joint(observables, a))
            effects[a] = s.simplify(effect)
        assert sum(effects.values(), s.zeros(5)) == I5
        for e in effects.values():
            assert e == e.T and s.simplify(e*e-e) == s.zeros(5)
        measurements[side][q] = effects

Kideal = s.diag(1, rt2, rt2, rt2, rt2)
Klast = s.diag(0, 1, 1, 1, 1)
ideal_zeros, last_sum, flat_sum = 0, s.Integer(0), s.Integer(0)
last_cells = {}
for x, ae in measurements["alice"].items():
    for y, be in measurements["bob"].items():
        partial = s.Integer(0)
        for a, e in ae.items():
            for b, f in be.items():
                if base_win(x, y, a, b):
                    continue
                assert s.simplify(e*Kideal*f) == s.zeros(5)
                ideal_zeros += 1
                lp = s.simplify(s.trace(Klast*e*Klast*f))
                tp = s.simplify(s.trace(e*f))
                assert lp >= 0 and tp >= 0
                partial += lp
                flat_sum += tp
        if partial:
            last_cells[x+","+y] = str(s.simplify(partial))
        last_sum += partial
assert ideal_zeros == 224 and s.simplify(last_sum) == 1
assert s.simplify(flat_sum - (3-2*rt2)) == 0
trace_loss = s.simplify(flat_sum*(R(1, 5)+R(1, 20))/52)
assert trace_loss == (3-2*rt2)/208

# Exact geometric sums and moments are derived from a single boundary chain.
k = s.symbols("k", integer=True, nonnegative=True)
trace_u, trace_v = R(3, 4)/16**k, R(3, 16)/16**k
den_u, den_v = R(7, 6)/4**k, R(7, 12)/4**k
assert s.summation(trace_u+trace_v, (k, 0, s.oo)) == 1
assert s.summation(trace_u*den_u+trace_v*den_v, (k, 0, s.oo)) == 1
assert s.simplify(trace_u/4-trace_v) == 0
assert s.simplify(trace_v/4-trace_u.subs(k,k+1)) == 0
assert s.simplify(den_u/2-den_v) == 0
assert s.simplify(den_v/2-den_u.subs(k,k+1)) == 0
delta = s.log(R(7,6))-s.log(2)/7
chain_moment = R(8,9)*s.log(R(7,6))+R(1,9)*s.log(R(7,12))-s.log(4)/63
assert s.simplify(chain_moment-delta) == 0

# Solve moment laws instead of inserting the eliminated coefficients.
p_mass, f_mass, m_mass, b_mass = s.symbols("p_mass f_mass m_mass b_mass")
masses = s.solve([f_mass-8*p_mass, m_mass-8*f_mass,
                  b_mass-(m_mass-p_mass), f_mass+m_mass-1],
                 [p_mass, f_mass, m_mass, b_mass])
assert masses[b_mass] == R(7,8)
Ap, Af, Am, J, D = s.symbols("Ap Af Am J D")
moments = s.solve([Af-8*Ap-8*s.log(2)*masses[p_mass],
                   Am-8*Af-8*s.log(2)*masses[f_mass],
                   J-(Am-Ap), D-(Af+Am)], [Af, Am, J, D])
assert s.simplify(moments[D]-R(8,7)*moments[J]+s.log(2)/7) == 0

# Nonuniform boundary spectra give a constructive family of attempted violations.
# It is a direct sum of copies of the infinite chain, with root densities g_i.
rng = random.Random(20261028)
excesses = []
for trial in range(1000):
    n = rng.randrange(2, 10)
    alpha = np.array([rng.random()+0.01 for _ in range(n)])
    alpha /= alpha.sum()
    g = np.exp(np.array([rng.uniform(-8, 8) for _ in range(n)]))
    g /= np.dot(alpha,g)
    relative = float(np.dot(alpha*g, np.log(g)))
    assert relative >= -1e-13
    Dvalue = float(delta)+relative
    Jvalue = 7/8*math.log(7/6)+7/8*relative
    assert abs(Dvalue-(8*Jvalue/7-math.log(2)/7)) < 1e-13
    excesses.append(relative)

# A second finite-model implementation uses compressed block insertions and
# computes each probability as ||sqrt(E) K sqrt(F)||_2^2, including all outcomes.
# The only nonprojective effects are half of a rank-one projector in the final
# copy-two block, so sqrt(E)=E/sqrt(c) for its nonzero eigenvalue c=1/2.
num_base = {side: {q: {a: np.array(e, dtype=float) for a,e in effects.items()}
                         for q,effects in family.items()}
            for side,family in measurements.items()}

def sparse_model(levels):
    denominator = 4*16**levels
    U, V, vals = [], [], []
    position = 0
    for j in range(levels):
        nu, nv = 3*16**(levels-j), 12*16**(levels-j-1)
        U.append(np.arange(position,position+nu)); position += nu
        V.append(np.arange(position,position+nv)); position += nv
        vals.extend([7/6/4**j]*nu + [7/12/4**j]*nv)
    dimension = position
    assert dimension == 4*(16**levels-1)
    blocks = {1: [], 2: []}
    for j in range(levels):
        blocks[1].extend(([f, *u] for f,u in zip(V[j], U[j].reshape(-1,4))))
        flags = U[j+1] if j+1 < levels else np.full(len(V[j])//4,-1)
        blocks[2].extend(([f,*v] for f,v in zip(flags,V[j].reshape(-1,4))))
    boundary = sparse.csr_matrix((np.ones(len(U[0])),(U[0],U[0])),shape=(dimension,dimension))
    label_u, label_v = np.concatenate(U), np.concatenate(V)

    def lift(local, copy, root=False):
        rr,cc,vv = [],[],[]
        for block in blocks[copy]:
            present = np.array(block) >= 0
            small = local[np.ix_(present,present)]
            coords = np.array(block)[present]
            if root:
                # P^2=P in a full block, E^2=(1/2)E for cut sign effects.
                tr = np.trace(small)
                if tr > 1e-14:
                    small = small/math.sqrt(tr)
            a,b = np.nonzero(small)
            rr.extend(coords[a]); cc.extend(coords[b]); vv.extend(small[a,b])
        return sparse.csr_matrix((vv,(rr,cc)),shape=(dimension,dimension))

    full, roots = {"alice":{},"bob":{}}, {"alice":{},"bob":{}}
    max_sum_error, max_root_error = 0.,0.
    for side in full:
        for question, answers in TABLE[side].items():
            if question == "label":
                fx = [sparse.csr_matrix((np.ones(len(idx)),(idx,idx)),shape=(dimension,dimension))
                      for idx in (label_u,label_v)]
                sx = fx
            else:
                copy, q = question.split(":"); copy=int(copy)
                fx,sx = [],[]
                for raw in answers:
                    a = encode(raw)
                    fx.append(boundary if a == "bot" else lift(num_base[side][q][a],copy))
                    sx.append(boundary if a == "bot" else lift(num_base[side][q][a],copy,True))
            err = sum(fx)-sparse.eye(dimension)
            max_sum_error=max(max_sum_error,float(abs(err).max()))
            for e,r in zip(fx,sx):
                max_root_error=max(max_root_error,float(abs(r@r-e).max()))
            full[side][question], roots[side][question] = fx,sx
    assert max_sum_error < 1e-13 and max_root_error < 1e-13
    state_mass = 1-64**(-levels)
    coeff = sparse.diags(np.sqrt(np.array(vals)/(denominator*state_mass)),format="csr")
    per_pair, total_loss, normalization_error = {},0.,0.
    for pair in TABLE["supported_pairs"]:
        ae,be = roots["alice"][pair["alice"]], roots["bob"][pair["bob"]]
        bad = {tuple(z) for z in pair["forbidden"]}
        norm,loss=0.,0.
        for i,e in enumerate(ae):
            for j,f in enumerate(be):
                amplitude=e@coeff@f
                probability=float(amplitude.multiply(amplitude).sum())
                assert probability >= 0
                norm += probability
                if (i,j) in bad:
                    loss += probability
        normalization_error=max(normalization_error,abs(norm-1))
        if loss > 1e-20:
            per_pair[pair["alice"]+","+pair["bob"]]=loss
        total_loss += loss/52
    expected=Q(7,208*(64**levels-1))
    assert abs(total_loss-float(expected)) < 1e-14 and normalization_error < 1e-13
    return dict(K=levels,dimension=dimension,loss=total_loss,exact_loss=str(expected),
                max_probability_normalization_error=normalization_error,
                max_completeness_error=max_sum_error,max_sqrt_error=max_root_error,
                nonzero_losing_cells=per_pair)

finite = [sparse_model(k) for k in (1,2)]

# An exact dimension comparison at EVERY E4 point: use release index 3K-2.
# It has equal loss at K=1 and strictly smaller loss thereafter.
comparisons=[]
for k0 in range(1,11):
    d_new=4*(16**k0-1)
    n_old=3*k0-2
    ell_new=Q(7,208*(64**k0-1))
    ell_old=Q(1,624*(4**n_old-1))
    assert 20*n_old < d_new and ell_old <= ell_new
    comparisons.append(dict(K=k0,new_dimension=d_new,new_loss=str(ell_new),
                            release_index=n_old,release_dimension=20*n_old,release_loss=str(ell_old)))

peak={}
if os.name == "nt":
    class PMC(ctypes.Structure):
        _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong)]+[
            (n,ctypes.c_size_t) for n in ("PeakWorkingSetSize","WorkingSetSize",
             "QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage",
             "QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
    info=PMC(); info.cb=ctypes.sizeof(info)
    current=ctypes.windll.kernel32.GetCurrentProcess; current.restype=ctypes.c_void_p
    get=ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes=(ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong)
    assert get(current(),ctypes.byref(info),ctypes.sizeof(info))
    peak=dict(peak_working_set_bytes=info.PeakWorkingSetSize,peak_commit_bytes=info.PeakPagefileUsage)
    assert max(peak.values()) < 500_000_000

result=dict(event_count=event_count,exact_forbidden_ideal_products=ideal_zeros,
            primary_FMS_convention_conjugacy="Hadamard tensor Hadamard; all square and distinguished-vector labels checked exactly",
            exact_last_block_unweighted_loss=str(last_sum),last_block_cells=last_cells,
            exact_flat_block_loss=str(s.simplify(flat_sum)),
            trace_loss_upper_bound=str(trace_loss),
            delta_nats_60_digits=str(delta.evalf(60)),
            delta_bits_60_digits=str((delta/s.log(2)).evalf(60)),
            nonuniform_boundary_trials=len(excesses),minimum_sampled_entropy_excess=min(excesses),
            exact_entropy_identity="D + ln(T) - Delta = relative_entropy(normalized_boundary || uniform_boundary)",
            finite_models=finite,dimension_comparisons=comparisons,
            source_sha256={str(f.relative_to(SRC)):hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in (SRC/"main.tex",SRC/"anc/game.json")},**peak)
(OUT/"independent-results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
print("ALL INDEPENDENT CHECKS PASSED",flush=True)
