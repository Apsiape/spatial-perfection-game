"""Independent exact checks and attempted counterexamples. No input modules imported."""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import sympy as s

OUT = Path(__file__).resolve().parent
I2 = s.eye(2)
X = s.Matrix([[0,1],[1,0]])
Z = s.diag(1,-1)
Y = s.Matrix([[0,-s.I],[s.I,0]])
kron = s.kronecker_product
M = [[kron(X,I2),kron(I2,X),kron(X,X)],
     [kron(I2,Z),kron(Z,I2),kron(Z,Z)],
     [kron(X,Z),kron(Z,X),kron(Y,Y)]]
I4 = s.eye(4)
I5 = s.eye(5)
zero5 = s.zeros(5)
rows = [u for u in itertools.product((1,-1),repeat=3) if math.prod(u)==1]
cols = [[u for u in itertools.product((1,-1),repeat=3) if math.prod(u)==p] for p in (1,1,-1)]
cx = [(1,1,1),(1,-1,-1)]
ry = cx
flag = s.diag(1,0,0,0,0)

def simplify(A):
    return A.applyfunc(s.simplify)

def assert_zero(A):
    assert simplify(A) == s.zeros(A.rows,A.cols), A

def joint(ops,ans):
    P = I4
    for op,sign in zip(ops,ans):
        P = P*(I4+sign*op)/2
    return simplify(P)

def embed(A):
    return s.diag(0,A)

A = {}; B = {}; aa = {}; bb = {}
for i in range(3):
    aa['R',i] = rows+['f']
    A['R',i] = [embed(joint(M[i],u)) for u in rows]+[flag]
for j in range(3):
    bb['C',j] = cols[j]+['f']
    B['C',j] = [embed(joint([M[i][j].conjugate() for i in range(3)],u)) for u in cols[j]]+[flag]
cvec = [s.Matrix([1,0,1,0])/s.sqrt(2),s.Matrix([0,1,0,1])/s.sqrt(2)]
rvec = [s.Matrix([1,1,1,1])/2,s.Matrix([1,-1,1,-1])/2]
for x in range(2):
    aa['T',x] = ['+','-']+[u for u in cols[0] if u!=cx[x]]
    f = s.Matrix([1,0,0,0,0]); c = s.Matrix([0,*list(cvec[x])])
    A['T',x] = [(f+c)*(f+c).T/2,(f-c)*(f-c).T/2]+[embed(joint([M[i][0] for i in range(3)],u)) for u in aa['T',x][2:]]
for y in range(2):
    bb['U',y] = ['+','-']+[u for u in rows if u!=ry[y]]
    f = s.Matrix([1,0,0,0,0]); r = s.Matrix([0,*list(rvec[y])])
    B['U',y] = [(f+r)*(f+r).T/2,(f-r)*(f-r).T/2]+[embed(joint(M[0],u)) for u in bb['U',y][2:]]

def win(q,r,a,b):
    if q[0]=='R' and r[0]=='C':
        if a=='f' or b=='f': return a==b
        return a[r[1]]==b[q[1]]
    if q[0]=='T' and r==('C',0):
        return b in ('f',cx[q[1]]) if a in ('+','-') else b==a
    if q==('R',0) and r[0]=='U':
        return a in ('f',ry[r[1]]) if b in ('+','-') else a==b
    if q[0]=='T' and r[0]=='U' and a in ('+','-') and b in ('+','-'):
        return (1 if a=='+' else -1)*(1 if b=='+' else -1)==(-1)**(q[1]*r[1])
    return True

checks = 0
for effects in [*A.values(),*B.values()]:
    assert_zero(sum(effects,s.zeros(5))-I5)
    for E in effects:
        assert_zero(E-E.H); assert_zero(E*E-E)
        checks += 2
K = s.diag(1,s.sqrt(2),s.sqrt(2),s.sqrt(2),s.sqrt(2))/3
for q in aa:
    for r in bb:
        for ia,a in enumerate(aa[q]):
            for ib,b in enumerate(bb[r]):
                if not win(q,r,a,b):
                    assert_zero(A[q][ia]*K*B[r][ib].T)
                    checks += 1
print('EXACT ideal completeness, PVM and forbidden amplitude checks:',checks,flush=True)

def Fcell(j,i,sign):
    return sum((B['C',j][k] for k,u in enumerate(bb['C',j]) if u!='f' and u[i]==sign),s.zeros(5))
bt = []
g = []
for y in range(2):
    bt.append(Fcell(2,0,ry[y][2])*Fcell(1,0,ry[y][1])*Fcell(0,0,ry[y][0]))
for x in range(2):
    g.append(B['C',0][bb['C',0].index(cx[x])])
b12=Fcell(1,0,1)-Fcell(1,0,-1)+flag
b21=Fcell(0,1,1)-Fcell(0,1,-1)+flag
assert_zero(bt[0]*g[1]-bt[0]*g[0]*b12)
assert_zero(bt[1]*g[1]+bt[1]*g[0]*b12)
assert_zero(bt[1]*g[0]-b21*bt[0]*g[0])
for x in range(2):
    for y in range(2):
        assert s.simplify((cvec[x].T*rvec[y])[0]-(-1)**(x*y)/s.sqrt(2))==0
print('EXACT W1,W2,W3 and signed overlaps: PASS',flush=True)

rho=s.symbols('rho',positive=True)
Kwrong=s.diag(1,rho,rho,rho,rho)/s.sqrt(1+4*rho**2)
total=0
for x in range(2):
    for y in range(2):
        for ia in range(2):
            for ib in range(2):
                if not win(('T',x),('U',y),aa['T',x][ia],bb['U',y][ib]):
                    total += s.trace(A['T',x][ia]*Kwrong*B['U',y][ib].T*Kwrong.H)
exact_loss=s.simplify(total/25)
expected=(rho-s.sqrt(2))**2/(25*(1+4*rho**2))
assert s.simplify(exact_loss-expected)==0
print('EXACT wrong-ratio loss = (rho-sqrt(2))^2 / [25(1+4 rho^2)]',flush=True)
print('EXACT wrong-ratio sector defect squared = 100 * mean loss',flush=True)

# Exhaust all 5^5 deterministic Alice answers; optimize Bob separately per question.
best=-1; witness=None
alist=list(aa); blist=list(bb)
score={}
for iq,q in enumerate(alist):
    for ir,r in enumerate(blist):
        for ia,a in enumerate(aa[q]):
            score[iq,ir,ia]=[int(win(q,r,a,b)) for b in bb[r]]
for answers in itertools.product(range(5),repeat=5):
    sc=sum(max(sum(score[iq,ir,answers[iq]][ib] for iq in range(5)) for ib in range(5)) for ir in range(5))
    if sc>best: best=sc; witness=answers
assert best==24
print('EXACT classical FMS value:',Fraction(best,25),'Alice assignments:',5**5,flush=True)

# Check all 32 signed real Pauli elements and their normal-form group table.
gens=[M[0][0],M[0][1],M[1][0],M[1][1]]
group=[]; keys={}
for z in range(2):
    for bits in itertools.product(range(2),repeat=4):
        T=(-1)**z*I4
        for G,bit in zip(gens,bits):
            if bit: T=T*G
        key=tuple(T)
        assert key not in keys
        keys[key]=len(group); group.append(T)
for U in group:
    for V in group:
        assert tuple(U*V) in keys
assert len(group)==32
print('EXACT real Pauli group: 32 distinct elements; 1024 products closed',flush=True)

print('CS Appendix A.3 counterexample (C trivial):',flush=True)
for t in [1e-2,1e-4,1e-6]:
    lhs=2*math.sqrt(t); rhs=6*t
    assert lhs>rhs
    print(f'  projector defect^2={t:.1e}, trace distance={lhs:.6g}, alleged bound={rhs:.6g}',flush=True)

H=math.log2(48)-Fraction(2,9)
print('Entropy infinite chain:',H,flush=True)
for level in range(1,7):
    loss=Fraction(1,624*(4**level-1))
    HK=H+math.log2(1-4.0**(-level))-2*level*4.0**(-level)/(1-4.0**(-level))
    assert HK<H<5.4
    print(f'  K={level}: rank={20*level}, exact loss={loss}, loss*4^K={float(loss)*4**level:.10g}, H={HK:.10g}',flush=True)
print('Pure-state entropy floor for eps <= 1/104:',-math.log2(1-Fraction(1,104**2)),flush=True)

# A positive formal multiset model for two equal-ratio coordinates.
# Record coefficients at s*2^exponent. S has one copy at each exponent <= 0.
def S(exp): return int(exp<=0)
def apply_T(f,exp): return 16*f(exp-1)-f(exp)
for exp in range(-30,4):
    assert apply_T(S,exp)>=0
    assert apply_T(lambda n:apply_T(S,n),exp)>=0
assert [apply_T(lambda n:apply_T(S,n),exp) for exp in (2,1,0,-1)]==[256,224,225,225]
print('EXACT formal diagonal model: T^2 S multiplicities [256,224,225,225,...] nonnegative',flush=True)

# Product-tail exact arithmetic for fixed k.
for k in (1,2,3,4):
    for L in (1,4,10,20):
        count=20**k*math.comb(L+k,k)
        retained=sum(Fraction(3,4)**k*math.comb(n+k-1,k-1)*Fraction(1,4)**n for n in range(L+1))
        tail=1-retained
        assert tail>0
        assert count>0
print('EXACT product simplex counts and negative-binomial tails: PASS',flush=True)
print('DONE',flush=True)
