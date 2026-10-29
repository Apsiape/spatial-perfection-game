"""Check every vector identity used in the robust bridge proof , exactly on the ideal FMS
strategy and as O(sqrt loss) on perturbed strategies with junk.  Vectors are coefficient matrices:
Alice operator M acts as M @ K, Bob operator N acts as K @ N.T.
"""
import numpy as np
from fms_core import (ideal_strategy, loss, QA, QB, ROW_ANS, col_answers, RY, CX, sv, pad)
from check_rsl import with_junk, perturb, rng


def idx(lst, a):
    return lst.index(a)


def ops(A, B, d):
    o = {}
    o['P'] = A[('R', 0)][-1]
    o['Q'] = B[('C', 0)][-1]
    for y in range(2):
        o['H', y] = A[('R', 0)][idx(QA[('R', 0)], RY[y])]
        o['S\'', y] = B[('U', y)][0] + B[('U', y)][1]
        o['Y', y] = B[('U', y)][0] - B[('U', y)][1]
    for x in range(2):
        o['G', x] = B[('C', 0)][idx(QB[('C', 0)], CX[x])]
        o['S', x] = A[('T', x)][0] + A[('T', x)][1]
        o['X', x] = A[('T', x)][0] - A[('T', x)][1]

    def Ecell(i, j, s):  # Alice R_i non-flag answers with cell j value s
        return sum(A[('R', i)][k] for k, a in enumerate(QA[('R', i)]) if a != 'f' and a[j] == s)

    def Fcell(j, i, s):  # Bob C_j non-flag answers with cell i value s
        return sum(B[('C', j)][k] for k, b in enumerate(QB[('C', j)]) if b != 'f' and b[i] == s)
    o['A12'] = Ecell(0, 1, 1) - Ecell(0, 1, -1) + A[('R', 0)][-1]
    o['B12'] = Fcell(1, 0, 1) - Fcell(1, 0, -1) + B[('C', 1)][-1]
    o['B21'] = Fcell(0, 1, 1) - Fcell(0, 1, -1) + B[('C', 0)][-1]
    for y in range(2):
        s1, s2, s3 = RY[y]
        o['Bt', y] = Fcell(2, 0, s3) @ Fcell(1, 0, s2) @ Fcell(0, 0, s1)
    o['I'] = np.eye(d)
    return o


def defects(K, A, B):
    d = K.shape[0]
    o = ops(A, B, d)
    I = o['I']; P = o['P']
    al = lambda M, V: M @ V
    bo = lambda N, V: V @ N.T
    f = P @ K
    zeta = {(x, y): bo(o['S\'', y], o['S', x] @ (I - P) @ K) for x in range(2) for y in range(2)}
    out = {}
    for x in range(2):
        for y in range(2):
            out[f'zeta{x}{y} vs S_xH_y psi'] = np.linalg.norm(zeta[x, y] - o['S', x] @ o['H', y] @ K)
            out[f'zeta{x}{y} vs Bt_yG_x psi'] = np.linalg.norm(zeta[x, y] - bo(o['Bt', y] @ o['G', x], K))
            out[f'P zeta{x}{y}'] = np.linalg.norm(P @ zeta[x, y])
            out[f'Q zeta{x}{y}'] = np.linalg.norm(bo(o['Q'], zeta[x, y]))
    out['(4a) zeta10 vs A12 zeta00'] = np.linalg.norm(zeta[1, 0] - o['A12'] @ zeta[0, 0])
    out['(4b) zeta11 vs -A12 zeta01'] = np.linalg.norm(zeta[1, 1] + o['A12'] @ zeta[0, 1])
    out['(4c) zeta01 vs B21 zeta00'] = np.linalg.norm(zeta[0, 1] - bo(o['B21'], zeta[0, 0]))
    out['(W1)'] = np.linalg.norm(bo(o['Bt', 0] @ o['G', 1], K) - bo(o['Bt', 0] @ o['G', 0] @ o['B12'], K))
    out['(W2)'] = np.linalg.norm(bo(o['Bt', 1] @ o['G', 1], K) + bo(o['Bt', 1] @ o['G', 0] @ o['B12'], K))
    out['(W3)'] = np.linalg.norm(bo(o['Bt', 1] @ o['G', 0], K) - bo(o['B21'] @ o['Bt', 0] @ o['G', 0], K))
    out['(F6a) X0Y0 f vs zeta00'] = np.linalg.norm(bo(o['Y', 0], o['X', 0] @ f) - zeta[0, 0])
    out['(F6b) X0Y0 zeta00 vs f'] = np.linalg.norm(bo(o['Y', 0], o['X', 0] @ zeta[0, 0]) - f)
    s_f = sv(f); s_z = sv(zeta[0, 0]); n = max(len(s_f), len(s_z))
    out['Sch(f) vs Sch(zeta00)'] = np.linalg.norm(pad(s_f, n) - pad(s_z, n))
    s_m = sv((I - P) @ K); tgt = np.sort(np.repeat(np.sqrt(2) * s_z, 4))[::-1]; n = max(len(s_m), len(tgt))
    out['Sch(psi_m) vs 4x sqrt2 Sch(zeta00)'] = np.linalg.norm(pad(s_m, n) - pad(tgt, n))
    return out


if __name__ == '__main__':
    K0, A0, B0 = ideal_strategy()
    print('IDEAL (all should be ~0):')
    for k, v in defects(K0, A0, B0).items():
        print(f'   {k:40s} {v:.2e}')
    worst = {}
    for m in [1, 2, 3]:
        for trial in range(5):
            lam = rng.exponential(size=m); lam = np.sqrt(lam / lam.sum())
            U = np.linalg.qr(rng.normal(size=(m, m)))[0]; V = np.linalg.qr(rng.normal(size=(m, m)))[0]
            chi = U @ np.diag(lam) @ V.T
            K, A, B = with_junk(K0, A0, B0, chi)
            for t in [3e-2, 1e-2, 3e-3, 1e-3]:
                K2, A2, B2 = perturb(K, A, B, t)
                l = loss(K2, A2, B2)
                for k, v in defects(K2, A2, B2).items():
                    worst[k] = max(worst.get(k, 0), v / np.sqrt(l))
    print('PERTURBED: worst defect / sqrt(loss) over 60 instances (m = 1,2,3; t = 3e-2..1e-3):')
    for k, v in worst.items():
        print(f'   {k:40s} {v:.2f}')
    print('DONE')
