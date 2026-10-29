"""Check the halving descent, occurrence blocks and rank-preserving dilation."""
import math
import numpy as np

rng=np.random.default_rng(720319)
worst_error=0.0
retained=0
for levels in (8,12,20,32):
    a=2.0**(-np.arange(levels))
    b=np.repeat(math.sqrt(2)*a,4)
    c=np.repeat(2*a,16)
    t=np.concatenate((np.repeat(2.0,16),np.repeat(a[:-1],15)))
    for perturbation in (0,0.05,0.2):
        delta=2*a[-1]
        aa=np.sort(a+rng.uniform(-perturbation,perturbation,len(a))*delta)[::-1]
        bb=np.sort(b+rng.uniform(-perturbation,perturbation,len(b))*delta)[::-1]
        cc=np.sort(c+rng.uniform(-perturbation,perturbation,len(c))*delta)[::-1]
        tt=np.sort(t+rng.uniform(-perturbation,perturbation,len(t))*delta)[::-1]
        union=np.sort(np.concatenate((aa,tt)))[::-1]
        n=max(len(union),len(cc))
        pad=lambda x:np.pad(x,(0,n-len(x)))
        delta=max(float(np.max(np.abs(pad(union)-pad(cc)))),
                  float(np.max(np.abs(cc-np.repeat(math.sqrt(2)*bb,4)))),
                  float(np.max(np.abs(bb-np.repeat(math.sqrt(2)*aa,4)))))
        delta=max(delta,1e-15)
        tagged=sorted([(float(v),0,i) for i,v in enumerate(aa)]+
                      [(float(v),1,i) for i,v in enumerate(tt)],
                      key=lambda q:-q[0])
        position={i:p for p,(_,kind,i) in enumerate(tagged) if kind==0}
        count=max(0,math.floor(math.log2(aa[0]/(64*delta)))+1)
        indices=[]; index=0
        for k in range(count):
            value=aa[index]
            error=abs(value-aa[0]*2**(-k))
            assert error <= (2+math.sqrt(2))*delta+1e-14
            assert value>0
            if indices: assert value<aa[indices[-1]]
            assert index not in indices
            indices.append(index)
            block_b=bb[4*index:4*(index+1)]
            block_c=cc[16*index:16*(index+1)]
            assert len(block_b)==4 and len(block_c)==16
            assert np.max(abs(block_b-math.sqrt(2)*value))<=delta+1e-14
            assert np.max(abs(block_c-2*value))<=(1+math.sqrt(2))*delta+1e-14
            assert min(block_b)>delta and min(block_c)>delta
            worst_error=max(worst_error,error/delta)
            p=position[index]
            index=p//16
        assert count>=math.log2(aa[0]/(64*delta))
        retained+=count
print('Descent: 12 finite exact/perturbed sequence families passed')
print('Selected positive levels:',retained)
print('Largest accumulated error / delta:',worst_error)

E=np.array([[.7,.1],[.1,.2]])
evals,U=np.linalg.eigh(E)
sqrtE=(U*np.sqrt(evals))@U.T
sqrtC=(U*np.sqrt(1-evals))@U.T
W=np.concatenate((sqrtE,sqrtC),axis=0)
P=np.diag([1,1,0,0])
assert np.linalg.norm(W.T@W-np.eye(2))<1e-14
assert np.linalg.norm(W.T@P@W-E)<1e-14
K=np.diag(np.sqrt([.8,.2]))
Kd=W@K@W.T
assert np.linalg.matrix_rank(K)==np.linalg.matrix_rank(Kd)==2
assert np.linalg.norm(np.linalg.svd(Kd,compute_uv=False)[:2]-
                      np.linalg.svd(K,compute_uv=False))<1e-14
assert abs(np.trace(E@K@E.T@K.T)-np.trace(P@Kd@P.T@Kd.T))<1e-14
print('Nonprojective local dilation: correlations and Schmidt rank 2 preserved')
print('Dilated local ambient dimension: 4; original local dimension: 2')
print('PASS')
