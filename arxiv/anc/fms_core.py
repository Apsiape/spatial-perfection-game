"""Flagged Magic Square (Mancinska arXiv 2610.06512) in the conventions of the accompanying paper, re-derived here.

Local space C^5 = span{f} (+) C^4, index 0 = flag, 1..4 = |00>,|01>,|10>,|11>.
A strategy is (K, A, B): K = dA x dB coefficient matrix of psi (psi = sum K_ij |i>|j>),
A[q] = list of Alice effects, B[r] = list of Bob effects (POVMs).  <psi|E (x) F|psi> = tr(E K F^T K^dag).
"""
import itertools
import numpy as np

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], float); Z = np.diag([1., -1.]); Y = np.array([[0, -1j], [1j, 0]])
kr = np.kron
MS = [[kr(X, I2), kr(I2, X), kr(X, X)],
      [kr(I2, Z), kr(Z, I2), kr(Z, Z)],
      [kr(X, Z), kr(Z, X), kr(Y, Y)]]
COLSIGN = [1, 1, -1]
SIGMA = [[1, 1], [1, -1]]

ROW_ANS = [t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == 1]


def col_answers(j):
    return [t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == COLSIGN[j]]


R0 = (1, 1, 1); R1 = (1, -1, -1)
C0 = (1, 1, 1); C1 = (1, -1, -1)
RY = [R0, R1]; CX = [C0, C1]


def joint_proj(ops, signs):
    P = np.eye(4, dtype=complex)
    for O, s in zip(ops, signs):
        P = P @ (np.eye(4) + s * O) / 2
    return P


def embed4(P4):
    P = np.zeros((5, 5), dtype=complex)
    P[1:, 1:] = P4
    return P


FLAG = np.zeros((5, 5), dtype=complex); FLAG[0, 0] = 1


def questions():
    """Answer lists for every question."""
    QA = {}
    for i in range(3):
        QA[('R', i)] = ROW_ANS + ['f']
    for x in range(2):
        QA[('T', x)] = ['+', '-'] + [c for c in col_answers(0) if c != CX[x]]
    QB = {}
    for j in range(3):
        QB[('C', j)] = col_answers(j) + ['f']
    for y in range(2):
        QB[('U', y)] = ['+', '-'] + [r for r in ROW_ANS if r != RY[y]]
    return QA, QB


def wins(q, r, a, b):
    if q[0] == 'R' and r[0] == 'C':
        i, j = q[1], r[1]
        if a == 'f' and b == 'f':
            return True
        if (a == 'f') != (b == 'f'):
            return False
        return a[j] == b[i]
    if q[0] == 'T' and r == ('C', 0):
        x = q[1]
        if a in ('+', '-'):
            return b == 'f' or b == CX[x]
        return b == a
    if q == ('R', 0) and r[0] == 'U':
        y = r[1]
        if b in ('+', '-'):
            return a == 'f' or a == RY[y]
        return a == b
    if q[0] == 'T' and r[0] == 'U':
        x, y = q[1], r[1]
        if a in ('+', '-') and b in ('+', '-'):
            sa = 1 if a == '+' else -1
            sb = 1 if b == '+' else -1
            return sa * sb == SIGMA[x][y]
        return True
    return True


QA, QB = questions()
PAIRS = [(q, r) for q in QA for r in QB]  # uniform on 25 pairs


def ideal_strategy():
    K = np.zeros((5, 5), dtype=complex)
    K[0, 0] = 1 / 3
    for z in range(1, 5):
        K[z, z] = np.sqrt(2) / 3
    A = {}; B = {}
    for i in range(3):
        A[('R', i)] = [embed4(joint_proj(MS[i], s)) for s in ROW_ANS] + [FLAG.copy()]
    for j in range(3):
        ops = [MS[i][j] for i in range(3)]
        # Bob measures the complex conjugates (transposes) so that psi ~ Phi mirrors Alice
        B[('C', j)] = [embed4(joint_proj([o.conj() for o in ops], s)) for s in col_answers(j)] + [FLAG.copy()]
    colvec = {}
    for s in col_answers(0):
        P = joint_proj([MS[i][0] for i in range(3)], s)
        w, v = np.linalg.eigh(P)
        colvec[s] = v[:, -1]
    rowvec = {}
    for s in ROW_ANS:
        P = joint_proj(MS[0], s)
        w, v = np.linalg.eigh(P)
        rowvec[s] = v[:, -1]
    # fix phases: real positive first nonzero... choose c0=|+0>, c1=|+1>, r0=|++>, r1=|+->
    h = 1 / np.sqrt(2)
    colvec[C0] = np.array([h, 0, h, 0]); colvec[C1] = np.array([0, h, 0, h])
    rowvec[R0] = np.array([.5, .5, .5, .5]); rowvec[R1] = np.array([.5, -.5, .5, -.5])
    for x in range(2):
        c = np.zeros(5, complex); c[1:] = colvec[CX[x]]
        f = np.zeros(5, complex); f[0] = 1
        plus = (f + c) / np.sqrt(2); minus = (f - c) / np.sqrt(2)
        effs = [np.outer(plus, plus.conj()), np.outer(minus, minus.conj())]
        for s in QA[('T', x)][2:]:
            v = np.zeros(5, complex); v[1:] = colvec[s]
            effs.append(np.outer(v, v.conj()))
        A[('T', x)] = effs
    for y in range(2):
        r = np.zeros(5, complex); r[1:] = rowvec[RY[y]].conj()
        f = np.zeros(5, complex); f[0] = 1
        plus = (f + r) / np.sqrt(2); minus = (f - r) / np.sqrt(2)
        effs = [np.outer(plus, plus.conj()), np.outer(minus, minus.conj())]
        for s in QB[('U', y)][2:]:
            v = np.zeros(5, complex); v[1:] = rowvec[s].conj()
            effs.append(np.outer(v, v.conj()))
        B[('U', y)] = effs
    return K, A, B


def loss(K, A, B, per_pair=False):
    tot = 0.0; pp = {}
    KH = K.conj().T
    for (q, r) in PAIRS:
        l = 0.0
        for ia, a in enumerate(QA[q]):
            EK = A[q][ia] @ K
            for ib, b in enumerate(QB[r]):
                if not wins(q, r, a, b):
                    l += np.real(np.trace(EK @ B[r][ib].T @ KH))
        pp[(q, r)] = l
        tot += l
    tot /= len(PAIRS)
    return (tot, pp) if per_pair else tot


def psd_sqrt(E):
    w, v = np.linalg.eigh((E + E.conj().T) / 2)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.conj().T


def sv(M):
    return np.sort(np.linalg.svd(M, compute_uv=False))[::-1]


def pad(v, n):
    out = np.zeros(n); out[:min(n, len(v))] = v[:n]
    return out


def sector_law_defect(K, A):
    """|| Sch(psi_MS) - (4 x sqrt2 Sch(psi_f)) ||_2 with the Naimark-consistent sector vectors."""
    Ef = A[('R', 0)][-1]
    d = K.shape[0]
    Kf = psd_sqrt(Ef) @ K
    Km = psd_sqrt(np.eye(d) - Ef) @ K
    sf = sv(Kf); sm = sv(Km)
    target = np.sort(np.repeat(np.sqrt(2) * sf, 4))[::-1]
    n = max(len(sm), len(target))
    return np.linalg.norm(pad(sm, n) - pad(target, n)), sf, sm


if __name__ == '__main__':
    K, A, B = ideal_strategy()
    print('ideal loss', loss(K, A, B))
    print('norm', np.linalg.norm(K))
    for q in A:
        S = sum(A[q]); print(q, 'POVM err', np.abs(S - np.eye(5)).max())
    for r in B:
        S = sum(B[r]); print(r, 'POVM err', np.abs(S - np.eye(5)).max())
    print('sector defect', sector_law_defect(K, A)[0])
