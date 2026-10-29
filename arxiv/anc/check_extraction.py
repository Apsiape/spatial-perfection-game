"""Independent numerical stress check of the direct magic-square extraction."""
import itertools
import numpy as np
from threadpoolctl import threadpool_info

rng=np.random.default_rng(781123)
I=np.eye(2); X=np.array([[0,1],[1,0]]); Z=np.diag([1,-1]); Y=np.array([[0,-1j],[1j,0]])
kr=np.kron
MS=[[kr(X,I),kr(I,X),kr(X,X)],
    [kr(I,Z),kr(Z,I),kr(Z,Z)],
    [kr(X,Z),kr(Z,X),kr(Y,Y)]]
norm=np.linalg.norm
labels=list(itertools.product(range(2),repeat=5))

def forms(gens):
    result=[]
    for central,*bits in labels:
        T=(-1)**central*np.eye(gens[0].shape[0],dtype=complex)
        for op,bit in zip(gens,bits):
            if bit: T=T@op
        result.append(T)
    return result

ideal=forms([MS[0][0],MS[0][1],MS[1][0],MS[1][1]])
lookup={tuple(np.rint(T.real).astype(int).ravel()):j for j,T in enumerate(ideal)}
mul=np.array([[lookup[tuple(np.rint((U@V).real).astype(int).ravel())] for V in ideal] for U in ideal])
sgens=[lookup[tuple(G.real.astype(int).ravel())] for G in [MS[0][0],MS[0][1],MS[1][0],MS[1][1]]]
J=lookup[tuple((-np.eye(4,dtype=int)).ravel())]

def unitary(d,t):
    T=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)); H=(T+T.conj().T)/2
    w,v=np.linalg.eigh(H)
    return (v*np.exp(1j*t*w))@v.conj().T

worst_rel=0; worst_prod=0; worst_extraction=0; worst_pvm_spectrum=0
for m in (1,2,3):
    d=4*m
    for t in (1e-1,1e-2,1e-4):
        weights=rng.exponential(size=m); weights=np.sqrt(weights/weights.sum())
        K=kr(np.eye(4)/2,np.diag(weights)).astype(complex)
        K += t*(rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)))/d
        K /= norm(K)
        A=[]; B=[[None]*3 for _ in range(3)]
        for i in range(3):
            U=unitary(d,t)
            A.append([U@kr(MS[i][j],np.eye(m))@U.conj().T for j in range(3)])
        for j in range(3):
            U=unitary(d,t)
            for i in range(3): B[i][j]=U@kr(MS[i][j],np.eye(m))@U.conj().T
        err=max(norm(A[i][j]@K-K@B[i][j].T) for i in range(3) for j in range(3))
        a,b,c,dd=A[0][0],A[0][1],A[1][0],A[1][1]
        anti_bc=norm((b@c+c@b)@K); anti_ad=norm((a@dd+dd@a)@K)
        assert anti_bc <= 15*err+1e-12
        assert anti_ad <= 51*err+1e-12
        worst_rel=max(worst_rel,anti_bc/err,anti_ad/err)
        fa=forms([a,b,c,dd]); fb=forms([B[0][0],B[0][1],B[1][0],B[1][1]])
        for g in range(32):
            for h in range(32):
                prod=max(norm((fa[g]@fa[h]-fa[mul[g,h]])@K),norm(K@(fb[g]@fb[h]-fb[mul[g,h]]).T))
                assert prod<=4096*err+1e-11
                worst_prod=max(worst_prod,prod/err)
        WA=np.concatenate(fa,axis=0)/np.sqrt(32); WB=np.concatenate(fb,axis=0)/np.sqrt(32)
        assert norm(WA.conj().T@WA-np.eye(d))<1e-12
        assert norm(WB.conj().T@WB-np.eye(d))<1e-12
        v=(WA@K@WB.T).reshape(32,d,32,d)
        assert norm(v[mul[:,J],:,:,:]+v)<1e-12
        assert norm(v[:,:,mul[:,J],:]+v)<1e-12
        pv=v.copy()
        for h in sgens:
            both=pv[mul[:,h],:,:,:][:,:,mul[:,h],:]
            pv=(pv+both)/2
        q=norm(v-pv)
        assert q<=8193*err+1e-11
        worst_extraction=max(worst_extraction,q/err)
        sch=np.linalg.svd(pv.reshape(32*d,32*d),compute_uv=False)
        spectrum=max(float(np.ptp(sch[i:i+4])) for i in range(0,len(sch),4))
        worst_pvm_spectrum=max(worst_pvm_spectrum,spectrum)
        assert spectrum<1e-12
        print(f'm={m} t={t:g} mirror_error={err:.4g} projection_error={q:.4g} q/error={q/err:.4g}',flush=True)
print('worst anticommutator/error',worst_rel)
print('worst normal-form product/error',worst_prod)
print('worst invariant projection/error',worst_extraction)
print('worst fourfold Schmidt splitting of invariant projection',worst_pvm_spectrum)
print('BLAS pools',[(p['internal_api'],p['num_threads']) for p in threadpool_info()])
print('DONE')
