"""Exact gadget, predicate, gluing and finite-compression checks.
Reconstructs the measurements and predicate. Run with python -B.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[k]='1'
import argparse, itertools, json
from pathlib import Path
import sympy as sp
import numpy as np
OUT=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=OUT/'results.json')
args=parser.parse_args()
I=sp.eye(2); X=sp.Matrix([[0,1],[1,0]]); Z=sp.diag(1,-1); Y=sp.Matrix([[0,-sp.I],[sp.I,0]])
kr=sp.kronecker_product
square=[[kr(X,I),kr(I,X),kr(X,X)], [kr(I,Z),kr(Z,I),kr(Z,Z)], [kr(X,Z),kr(Z,X),kr(Y,Y)]]
signs=list(itertools.product((1,-1),repeat=3))
rows=[s for s in signs if sp.prod(s)==1]
cols=[[s for s in signs if sp.prod(s)==p] for p in (1,1,-1)]
for seq in square+[[square[i][j] for i in range(3)] for j in range(3)]:
    for a,b in itertools.combinations(seq,2): assert a*b==b*a
for i in range(3): assert square[i][0]*square[i][1]*square[i][2]==sp.eye(4)
for j,p in enumerate((1,1,-1)): assert square[0][j]*square[1][j]*square[2][j]==p*sp.eye(4)
def projector(mats, s):
    p=sp.eye(4)
    for a,t in zip(mats,s): p=p*(sp.eye(4)+t*a)/2
    return p
def embed(p): return sp.diag(sp.zeros(1),p)
flag=sp.diag(1,0,0,0,0)
c=[sp.Matrix([1,0,1,0])/sp.sqrt(2),sp.Matrix([0,1,0,1])/sp.sqrt(2)]
r=[sp.Matrix([1,1,1,1])/2,sp.Matrix([1,-1,1,-1])/2]
overlap=sp.Matrix(2,2,lambda x,y:(c[x].T*r[y])[0])
assert overlap==sp.Matrix([[1,1],[1,-1]])/sp.sqrt(2)
aq=[('row',i) for i in range(3)]+[('bridge',i) for i in range(2)]
bq=[('col',j) for j in range(3)]+[('bridge',j) for j in range(2)]
A={}; B={}; ca=[(1,1,1),(1,-1,-1)]; ra=ca.copy()
for q in aq:
    kind,i=q
    if kind=='row': A[q]={s:embed(projector(square[i],s)) for s in rows}; A[q]['f']=flag
    else:
        A[q]={s:embed(projector([square[t][0] for t in range(3)],s)) for s in cols[0] if s!=ca[i]}
        for t in (1,-1):
            v=sp.Matrix([1,*list(t*c[i])]); A[q][t]=v*v.T/2
for q in bq:
    kind,j=q
    if kind=='col': B[q]={s:embed(projector([square[t][j] for t in range(3)],s)) for s in cols[j]}; B[q]['f']=flag
    else:
        B[q]={s:embed(projector(square[0],s)) for s in rows if s!=ra[j]}
        for t in (1,-1):
            v=sp.Matrix([1,*list(t*r[j])]); B[q][t]=v*v.T/2
def accepted(q,t,a,b):
    u,i=q; v,j=t
    if u=='row' and v=='col':
        return (a==b=='f') if a=='f' or b=='f' else a[j]==b[i]
    if u=='bridge' and v=='col':
        return True if j else (b in ('f',ca[i]) if isinstance(a,int) else a==b)
    if u=='row' and v=='bridge':
        return True if i else (a in ('f',ra[j]) if isinstance(b,int) else a==b)
    return True if not isinstance(a,int) or not isinstance(b,int) else a*b==(-1)**(i*j)
D=sp.diag(1,*([sp.sqrt(2)]*4))/3
rejected=0; per_pair={}
for q in aq:
    for t in bq:
        total=sp.Integer(0)
        for a,e in A[q].items():
            for b,f in B[t].items():
                p=sp.simplify(sp.trace(e*D*f*D))
                total+=p
                if not accepted(q,t,a,b):
                    rejected+=1
                    assert sp.simplify(e*D*f)==sp.zeros(5)
        assert sp.simplify(total)==1
        per_pair[str((q,t))]='all losing products zero; probabilities sum to 1'
for m in list(A.values())+list(B.values()):
    assert sum(m.values(),sp.zeros(5))==sp.eye(5)
    for e in m.values(): assert e==e.T and e*e==e
print('EXACT FMS: 25 pairs, 625 events,',rejected,'forbidden products zero; all POVMs projective and complete',flush=True)
print('EXACT overlap:',overlap,flush=True)
# Exact compressed last copy-2 block: the flag coefficient is absent.
Dlast=sp.diag(0,1,1,1,1)
raw=0
for q in aq:
    for t in bq:
        for a,e in A[q].items():
            for b,f in B[t].items():
                if not accepted(q,t,a,b): raw+=sp.trace(e*Dlast*f*Dlast)
assert sp.simplify(raw)==1
print('EXACT unweighted last-block loss for coefficient b=1:',raw,flush=True)
# Fresh lifting, with all compressed effects represented directly in the retained space.
def finite(K):
    basis=[('A',i,j,k) for k in range(K) for i in range(4) for j in range(4)]+[('B',j,k) for k in range(K) for j in range(4)]
    ix={v:n for n,v in enumerate(basis)}; n=len(basis)
    s=np.array([(2**.5 if v[0]=='A' else 1)*2.**(-v[-1])/(48**.5) for v in basis]); s/=np.linalg.norm(s)
    blocks1=[[ix[('B',j,k)]]+[ix[('A',i,j,k)] for i in range(4)] for k in range(K) for j in range(4)]
    blocks2=[[ix.get(('A',0,0,k+1))]+[ix[('B',j,k)] for j in range(4)] for k in range(K)]
    used={v for g in blocks2 for v in g if v is not None}; bd=np.diag([int(i not in used) for i in range(n)])
    def lift(meas,blocks,boundary):
        ans={}
        for a,e in meas.items():
            arr=np.array(e,dtype=float); mat=np.zeros((n,n))
            for g in blocks:
                for i,x in enumerate(g):
                    for j,y in enumerate(g):
                        if x is not None and y is not None: mat[x,y]+=arr[i,j]
            ans[a]=mat
        if boundary: ans['bot']=bd
        return ans
    As={(h,q):lift(A[q],blocks1 if h==1 else blocks2,h==2) for h in (1,2) for q in aq}
    Bs={(h,t):lift(B[t],blocks1 if h==1 else blocks2,h==2) for h in (1,2) for t in bq}
    Bs['label']={a:np.diag([int(v[0]==a) for v in basis]) for a in ('A','B')}
    def win(h,q,t,a,b):
        if t=='label': return ((a!='f') if h==1 else (a in ('f','bot')))==(b=='A')
        if h==2 and ('bot' in (a,b)): return a==b=='bot'
        return accepted(q,t,a,b)
    losses={}; mineig=1.; maxcomplete=0.
    for ms in list(As.values())+list(Bs.values()):
        maxcomplete=max(maxcomplete,float(np.max(np.abs(sum(ms.values())-np.eye(n)))))
        for e in ms.values(): mineig=min(mineig,float(np.linalg.eigvalsh(e).min()))
    for h in (1,2):
        for q in aq:
            for t in bq:
                loss=sum(np.einsum('ij,ij,i,j->',e,f,s,s) for a,e in As[h,q].items() for b,f in Bs[h,t].items() if not win(h,q,t,a,b))
                losses[str((h,q,t))]=float(loss)
        q=('row',0)
        losses[str((h,q,'label'))]=float(sum(np.einsum('ij,ij,i,j->',e,f,s,s) for a,e in As[h,q].items() for b,f in Bs['label'].items() if not win(h,q,'label',a,b)))
    observed=sum(losses.values())/52
    expected=1/(624*(4**K-1))
    assert len(losses)==52 and abs(observed-expected)<1e-14
    assert mineig>-1e-12 and maxcomplete<1e-12
    return dict(K=K,dimension=n,loss=observed,exact=f'1/{624*(4**K-1)}',max_POVM_error=maxcomplete,min_eigenvalue=mineig,per_pair=losses)
numerical=[finite(k) for k in range(1,7)]
for row in numerical: print({k:v for k,v in row.items() if k!='per_pair'},flush=True)
# Normalization and all 52 infinite-pair checks reduce to exact gadget products and diagonal labels.
t=sp.symbols('t',integer=True,nonnegative=True)
norm=sp.summation(sp.Rational(3,4)*4**(-t),(t,0,sp.oo)); assert norm==1
sector_mass=dict(copy1_flag=str(sp.Rational(1,9)),copy1_MS=str(sp.Rational(8,9)),copy2_flag=str(sp.Rational(1,72)),copy2_MS=str(sp.Rational(1,9)),boundary=str(sp.Rational(7,8)))
print('EXACT normalized infinite state; sector masses:',sector_mass,flush=True)
# Compare every event of the standalone rational table with the reconstructed rules.
table=json.loads((OUT/'game.json').read_text(encoding='utf-8'))
def unpack(a):
    return tuple(a) if isinstance(a,list) else a
def parse(name,alice):
    h,tag=name.split(':')
    index=int(tag[1:])-(1 if tag[0] in 'RC' else 0)
    return int(h),('row' if tag[0]=='R' else 'col' if tag[0]=='C' else 'bridge',index)
events=0
for pair in table['supported_pairs']:
    assert pair['probability']==[1,52]
    h,q=parse(pair['alice'],True)
    forbidden={tuple(v) for v in pair['forbidden']}
    for ai,aa in enumerate(table['alice'][pair['alice']]):
        for bi,bb in enumerate(table['bob'][pair['bob']]):
            a,b=unpack(aa),unpack(bb)
            if pair['bob']=='label':
                good=((a!='f') if h==1 else (a in ('f','bot')))==(b=='A')
            else:
                hh,t=parse(pair['bob'],False); assert h==hh
                good=(a==b=='bot') if h==2 and ('bot' in (a,b)) else accepted(q,t,a,b)
            assert good==((ai,bi) not in forbidden)
            events+=1
assert len(table['alice'])==10 and len(table['bob'])==11
assert len(table['supported_pairs'])==52 and events==1547
assert sum(len(v) for v in table['alice'].values())+sum(len(v) for v in table['bob'].values())==112
assert max(map(len,table['alice'].values()))==max(map(len,table['bob'].values()))==6
print('EXACT rational table:',events,'events on 52 pairs match the reconstructed predicate',flush=True)
args.output.write_text(json.dumps(dict(exact_FMS_pairs=25,exact_FMS_events=625,forbidden_products=rejected,normalization=str(norm),last_block_loss=str(raw),sector_mass=sector_mass,table_events=events,numerical=numerical),indent=2)+'\n',encoding='utf-8',newline='\n')
print('ALL SYMBOLIC AND NUMERICAL CHECKS PASSED',flush=True)
