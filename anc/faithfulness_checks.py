import os
"""Independent, sequential, standard-library checks. All outputs go to the selected output directory."""
import ctypes
import hashlib
import itertools as it
import json
import math
from pathlib import Path
import random
import time
from fractions import Fraction as Q

START = time.perf_counter()
OUT = Path(os.environ.get('GAME_CHECK_OUTPUT', '.')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = Path(__file__).resolve().parent.parent
GAME = json.loads((SOURCE / 'anc/game.json').read_text(encoding='utf-8'))
result = {}

def answer(a):
    return tuple(a) if isinstance(a, list) else a

AA = {q: list(map(answer, a)) for q, a in GAME['alice'].items()}
BB = {q: list(map(answer, a)) for q, a in GAME['bob'].items()}
PAIRS = GAME['supported_pairs']

def oracle(x, y, a, b):
    h, x = x.split(':')
    if y == 'label':
        return ((a != 'f') if h == '1' else (a in ('f', 'bot'))) == (b == 'A')
    hy, y = y.split(':')
    assert h == hy
    if h == '2' and ('bot' in (a, b)):
        return a == b == 'bot'
    if x[0] == 'R' and y[0] == 'C':
        if 'f' in (a, b):
            return a == b == 'f'
        return a[int(y[1:])-1] == b[int(x[1:])-1]
    chosen = ((1, 1, 1), (1, -1, -1))
    if x[0] == 'T' and y == 'C1':
        return b in ('f', chosen[int(x[1:])]) if isinstance(a, int) else a == b
    if x == 'R1' and y[0] == 'U':
        return a in ('f', chosen[int(y[1:])]) if isinstance(b, int) else a == b
    if x[0] == 'T' and y[0] == 'U' and isinstance(a, int) and isinstance(b, int):
        return a*b == (-1)**(int(x[1:])*int(y[1:]))
    return True

events = 0
for pair in PAIRS:
    assert pair['probability'] == [1, 52]
    bad = set(map(tuple, pair['forbidden']))
    for ai, a in enumerate(AA[pair['alice']]):
        for bi, b in enumerate(BB[pair['bob']]):
            assert oracle(pair['alice'], pair['bob'], a, b) == ((ai, bi) not in bad)
            events += 1
assert len(PAIRS) == 52
result['predicate_crosscheck'] = {'supported_pairs': 52, 'answer_events': events,
    'canonical_game_sha256': hashlib.sha256((SOURCE/'anc/game.json').read_bytes()).hexdigest()}

# Exhaust the full Gamma game, using only the canonical forbidden tables.
# Once the shared label is fixed, the copies optimize independently. For each
# complete Alice assignment, optimize each internal Bob question separately.
copies = {}
for h in ('1', '2'):
    ax = [x for x in AA if x.startswith(h+':')]
    by = [y for y in BB if y.startswith(h+':')]
    tables = {}
    link = next(p for p in PAIRS if p['alice'] == h+':R1' and p['bob'] == 'label')
    link_bad = set(map(tuple, link['forbidden']))
    for p in PAIRS:
        if p['alice'].startswith(h+':') and p['bob'] != 'label':
            bad = set(map(tuple, p['forbidden']))
            tables[p['alice'], p['bob']] = [[int((i,j) not in bad)
                for j in range(len(BB[p['bob']]))] for i in range(len(AA[p['alice']]))]
    best = [-1, -1]
    witness = [None, None]
    count = 0
    best_internal = -1
    for av in it.product(*(range(len(AA[x])) for x in ax)):
        count += 1
        internal = sum(max(sum(tables[x,y][ai][bj] for x,ai in zip(ax,av))
                    for bj in range(len(BB[y]))) for y in by)
        best_internal = max(best_internal, internal)
        ai = av[ax.index(h+':R1')]
        for label in range(2):
            score = internal + int((ai, label) not in link_bad)
            if score > best[label]:
                best[label] = score
                witness[label] = dict(zip(ax, av))
    copies[h] = {'alice_assignments': count, 'best_internal': best_internal,
                 'best_with_label_A_B': best}
full_best = max(copies['1']['best_with_label_A_B'][i] +
                copies['2']['best_with_label_A_B'][i] for i in range(2))
assert full_best == 51
result['full_Gamma_exhaustion'] = {'copies': copies, 'best_score': [full_best,52],
    'exact_loss': str(Q(52-full_best,52)), 'method': 'Exact integer optimization of both copies and the shared label'}

def eye(n): return [[complex(i == j) for j in range(n)] for i in range(n)]
def add(a,b): return [[x+y for x,y in zip(r,s)] for r,s in zip(a,b)]
def scale(a,c): return [[c*x for x in r] for r in a]
def mul(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
def adj(a): return [[a[j][i].conjugate() for j in range(len(a))] for i in range(len(a[0]))]
def kron(a,b):
    return [[a[i][j]*b[k][l] for j in range(len(a[0])) for l in range(len(b[0]))]
        for i in range(len(a)) for k in range(len(b))]
def outer(v): return [[a*b.conjugate() for b in v] for a in v]
def mv(a,v): return [sum(x*y for x,y in zip(r,v)) for r in a]
def norm(v): return math.sqrt(sum(abs(x)**2 for x in v))
def normal(v): return [x/norm(v) for x in v]
def dist(a,b): return max(abs(x-y) for r,s in zip(a,b) for x,y in zip(r,s))
def embed(a): return [[0j]*5] + [[0j]+r for r in a]

I2 = eye(2)
X = [[0j,1+0j],[1+0j,0j]]
Y = [[0j,-1j],[1j,0j]]
Z = [[1+0j,0j],[0j,-1+0j]]
SQUARE = [[kron(X,I2),kron(I2,X),kron(X,X)],
          [kron(I2,Z),kron(Z,I2),kron(Z,Z)],
          [kron(X,Z),kron(Z,X),kron(Y,Y)]]
PF = outer([1+0j,0j,0j,0j,0j])
def square_projector(observables, signs):
    return scale(mul(add(eye(4),scale(observables[0],signs[0])),
                     add(eye(4),scale(observables[1],signs[1]))), .25)
chosen_c = [[1/math.sqrt(2),0,1/math.sqrt(2),0], [0,1/math.sqrt(2),0,1/math.sqrt(2)]]
chosen_r = [[.5,.5,.5,.5],[.5,-.5,.5,-.5]]
def ideal_effect(q,a,alice):
    q = q.split(':')[1]
    if a == 'f': return PF
    if q[0] == 'R': return embed(square_projector(SQUARE[int(q[1:])-1], a))
    if q[0] == 'C': return embed(square_projector([SQUARE[i][int(q[1:])-1] for i in range(3)],a))
    if isinstance(a,int):
        v = (chosen_c if alice else chosen_r)[int(q[1:])]
        return outer([complex(1/math.sqrt(2))]+[complex(a*x/math.sqrt(2)) for x in v])
    obs = [SQUARE[i][0] for i in range(3)] if alice else SQUARE[0]
    return embed(square_projector(obs,a))

D = [1/3]+[math.sqrt(2)/3]*4
fms_loss = 0.0
largest_nonzero_forbidden_product_trace = 0.0
completeness_error = 0.0
for side, questions in ((True,AA),(False,BB)):
    for q,ans in questions.items():
        if not q.startswith('1:'): continue
        total = [[0j]*5 for _ in range(5)]
        for a in ans:
            e = ideal_effect(q,a,side)
            assert dist(mul(e,e),e) < 2e-15
            total = add(total,e)
        completeness_error = max(completeness_error,dist(total,eye(5)))
for p in PAIRS:
    if not p['alice'].startswith('1:') or p['bob']=='label': continue
    for ai,bi in p['forbidden']:
        e = ideal_effect(p['alice'],AA[p['alice']][ai],True)
        f = ideal_effect(p['bob'],BB[p['bob']][bi],False)
        probability = sum(D[i]*D[j]*e[i][j]*f[i][j] for i in range(5) for j in range(5)).real
        assert abs(probability) < 2e-15
        fms_loss += probability/25
        tr_product = (sum(e[i][i] for i in range(5))*sum(f[i][i] for i in range(5))).real
        largest_nonzero_forbidden_product_trace = max(largest_nonzero_forbidden_product_trace,tr_product)
assert largest_nonzero_forbidden_product_trace > .99
result['local_faithfulness_counterexample'] = {'ideal_FMS_loss_roundoff': fms_loss,
    'reduced_state_eigenvalues': [1/9]+[2/9]*4,
    'forbidden_operator_product_nonzero_trace': largest_nonzero_forbidden_product_trace,
    'PVM_completeness_max_error': completeness_error,
    'meaning': 'Perfect nonclassical play and faithful marginals coexist; the joint state is not faithful.'}

# Test the effect identity for genuinely nonprojective commuting effects.
E = add(scale(eye(2),.2),scale(outer(normal([1+1j,2-1j])),.6))
F = add(scale(eye(2),.1),scale(outer(normal([2+1j,1-2j])),.7))
P, R = kron(E,I2), kron(I2,F)
I4 = eye(4)
lhs = add(mul(P,add(I4,scale(R,-1))),mul(add(I4,scale(P,-1)),R))
diff = add(P,scale(R,-1))
rhs = add(mul(diff,diff),add(add(P,scale(mul(P,P),-1)),add(R,scale(mul(R,R),-1))))
identity_error = dist(lhs,rhs)
assert identity_error < 1e-14
result['zero_mismatch_identity'] = {'nonprojective_effects': True, 'max_error': identity_error}

# A non-self-adjoint alleged double does not imply centralizer membership.
a,b = math.sqrt(1/3),math.sqrt(2/3)
psi = [a,0j,0j,b]
B = [[0j,a/b],[b/a,0j]]
u,v = mv(kron(X,I2),psi),mv(kron(I2,B),psi)
mirror_error = max(abs(x-y) for x,y in zip(u,v))
comm = add(mul(X,[[1/3,0j],[0j,2/3]]),scale(mul([[1/3,0j],[0j,2/3]],X),-1))
assert mirror_error < 1e-15 and dist(B,adj(B)) > .5
assert max(abs(x) for row in comm for x in row) > .3
result['E1_self_adjointness_boundary'] = {'mirror_error': mirror_error,
    'B_adjoint_error': dist(B,adj(B)), 'A_reduced_density_commutator_max': 1/3}

# Explicit finite-Alice compression on Bob's Schmidt support.
rng = random.Random(29012026)
K = [[complex(rng.uniform(-1,1),rng.uniform(-1,1)) for _ in range(5)] for _ in range(2)]
size = math.sqrt(sum(abs(z)**2 for row in K for z in row))
K = [[z/size for z in row] for row in K]
basis=[]
for row in K:
    v = row[:]
    for w in basis:
        coeff = sum(z.conjugate()*t for z,t in zip(w,v))
        v = [t-coeff*z for z,t in zip(w,v)]
    basis.append(normal(v))
V = [[basis[j][i] for j in range(2)] for i in range(5)]
Ksmall = [[sum(w.conjugate()*z for w,z in zip(basis[j],row)) for j in range(2)] for row in K]
def prob(k,e,f):
    return sum(k[i][j].conjugate()*e[i][a]*f[j][b]*k[a][b]
        for i in range(len(k)) for j in range(len(k[0]))
        for a in range(len(k)) for b in range(len(k[0])))
compression_error=0.0
for _ in range(10):
    e = outer(normal([complex(rng.random(),rng.random()) for _ in range(2)]))
    f = outer(normal([complex(rng.random(),rng.random()) for _ in range(5)]))
    small_f = mul(mul(adj(V),f),V)
    for ee in (e,add(eye(2),scale(e,-1))):
        for ff,sf in ((f,small_f),(add(eye(5),scale(f,-1)),add(eye(2),scale(small_f,-1)))):
            compression_error = max(compression_error,abs(prob(K,ee,ff)-prob(Ksmall,ee,sf)))
assert compression_error < 2e-14
result['finite_Alice_compression'] = {'original_dimensions':[2,5], 'compressed_dimensions':[2,2],
    'probabilities_compared':40,'max_error':compression_error}

def classical_beta(s,w,t):
    remain, value = t,Q(0)
    for i in sorted((i for i,x in enumerate(s) if x),key=lambda i:w[i]/s[i]):
        take = min(Q(1),remain/s[i])
        value += w[i]*take
        remain -= s[i]*take
        if remain == 0: return value
    raise AssertionError('infeasible test')

testing=[]
for lam in (Q(1,1000),Q(1,10),Q(1)):
    s = [Q(1,25)]*25
    w = [1-lam+lam/25]+[lam/25]*24
    beta = classical_beta(s,w,Q(1,25))
    assert beta == lam/25
    assert classical_beta(w,w,Q(1,25)) == Q(1,25)
    testing.append({'noise_weight':str(lam),'t':'1/25','beta':str(beta)})
result['exact_commuting_testing_problems'] = testing
result['rapidity_box_approximate_invariants'] = [{'R':R,'unit_shift_overlap':max(0,1-1/(2*R))}
    for R in (1,10,100,1000)]

result['elapsed_seconds'] = time.perf_counter()-START
try:
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    p=PMC(); p.cb=ctypes.sizeof(p)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(p),p.cb)
    result['peak_working_set_bytes']=int(p.PeakWorkingSetSize)
    assert p.PeakWorkingSetSize < 1_000_000_000
except (AttributeError,OSError):
    result['peak_working_set_bytes']='Unavailable on this platform'
result['status']='PASS'
result['limitations']='Finite checks do not prove infinite-dimensional structural or modular theorems.'
(OUT/'INDEPENDENT-CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
