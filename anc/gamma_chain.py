"""The glued flagged magic-square strategy and its approximate spectral relations.
For K levels: truncated explicit strategy (state truncated to levels < K, effects of the (K+1)-level model, which
equals the compressed strategy of the tail-to-loss bound).  Reports: loss, Schmidt rank r = 20K, the l2 defects of the three
chain relations (link, copy-1 law, copy-2 law), the descent chain found in a = Sch(copy-2 flag), and the count bound.
"""
import numpy as np
from fms_core import ideal_strategy, QA as FQA, QB as FQB, wins as fwins, sv, pad

K0, A0, B0 = ideal_strategy()   # 5x5 blocks: index 0 flag, 1..4 MS


def build(L):
    """L levels k = 0..L-1. Basis: label A |i,j,k> (16 per level), label B |j,k> (4 per level)."""
    idxA = {}; idxB = {}; n = 0
    for k in range(L):
        for i in range(4):
            for j in range(4):
                idxA[i, j, k] = n; n += 1
        for j in range(4):
            idxB[j, k] = n; n += 1
    return idxA, idxB, n


def strategy(Kt, L):
    idxA, idxB, n = build(L)
    b = np.array([2.0 ** (-k) / (4 * np.sqrt(3)) for k in range(L)]); a = np.sqrt(2) * b
    psi = np.zeros((n, n))
    for (i, j, k), p in idxA.items():
        if k < Kt:
            psi[p, p] = a[k]
    for (j, k), p in idxB.items():
        if k < Kt:
            psi[p, p] = b[k]
    psi /= np.linalg.norm(psi)
    # copy-1 blocks: for each (j,k): [B(j,k), A(0,j,k), ..., A(3,j,k)]
    blocks1 = [[idxB[j, k]] + [idxA[i, j, k] for i in range(4)] for k in range(L) for j in range(4)]
    # copy-2 blocks: for each k with k+1 < L: [A(0,0,k+1), B(0,k), ..., B(3,k)]
    blocks2 = [[idxA[0, 0, k + 1]] + [idxB[j, k] for j in range(4)] for k in range(L - 1)]
    covered2 = set(p for bl in blocks2 for p in bl)
    boundary = [p for p in range(n) if p not in covered2]

    def lift(effs5, blocks, extra=None):
        out = []
        for e in effs5:
            M = np.zeros((n, n), complex)
            for bl in blocks:
                M[np.ix_(bl, bl)] += e
            out.append(M)
        return out
    QA = {}; QB = {}; A = {}; B = {}
    for q in FQA:
        QA[(1, q)] = list(FQA[q]); A[(1, q)] = lift(A0[q], blocks1)
        QA[(2, q)] = list(FQA[q]) + ['bot']
        effs = lift(A0[q], blocks2)
        Pb = np.zeros((n, n), complex)
        for p in boundary:
            Pb[p, p] = 1
        A[(2, q)] = effs + [Pb]
    for r in FQB:
        QB[(1, r)] = list(FQB[r]); B[(1, r)] = lift(B0[r], blocks1)
        QB[(2, r)] = list(FQB[r]) + ['bot']
        effs = lift(B0[r], blocks2)
        Pb = np.zeros((n, n), complex)
        for p in boundary:
            Pb[p, p] = 1
        B[(2, r)] = effs + [Pb]
    LA = np.zeros((n, n), complex); LB = np.zeros((n, n), complex)
    for p in idxA.values():
        LA[p, p] = 1
    for p in idxB.values():
        LB[p, p] = 1
    QB['Lab'] = ['A', 'B']; B['Lab'] = [LA, LB]
    pairs = [((1, q), (1, r)) for q in FQA for r in FQB] + [((2, q), (2, r)) for q in FQA for r in FQB] + \
            [((1, ('R', 0)), 'Lab'), ((2, ('R', 0)), 'Lab')]
    return psi, A, B, QA, QB, pairs


def win(q, r, a, b):
    if r == 'Lab':
        if q[0] == 1:
            return (a != 'f') == (b == 'A')
        return (a in ('f', 'bot')) == (b == 'A')
    if q[0] == 1:
        return fwins(q[1], r[1], a, b)
    if a == 'bot' and b == 'bot':
        return True
    if (a == 'bot') != (b == 'bot'):
        return False
    return fwins(q[1], r[1], a, b)


def loss(psi, A, B, QA, QB, pairs):
    tot = 0.0; KH = psi.conj().T
    for (q, r) in pairs:
        for ia, a in enumerate(QA[q]):
            EK = A[q][ia] @ psi
            for ib, b in enumerate(QB[r]):
                if not win(q, r, a, b):
                    tot += np.real(np.trace(EK @ B[r][ib].T @ KH))
    return tot / len(pairs)


def chain_report(psi, A, B, QA):
    """Robust chain relations of approximate gluing: c = Sch(alpha), b = Sch(beta), a = Sch(phi_2)."""
    LA, LB = B['Lab']
    alpha = psi @ LA.T; beta = psi @ LB.T
    q2 = (2, ('R', 0))
    Ef2 = A[q2][QA[q2].index('f')]
    phi2 = Ef2 @ psi
    c = sv(alpha); b = sv(beta); a = sv(phi2)
    n = len(c) + len(b) + len(a)
    tA = np.sort(np.concatenate([a, sv(A[q2][-1] @ psi @ B[(2, ('C', 0))][-1].T)]))[::-1]
    d_link = np.linalg.norm(pad(c, n) - pad(tA, n))
    d_c1 = np.linalg.norm(pad(c, n) - pad(np.sort(np.repeat(np.sqrt(2) * b, 4))[::-1], n))
    d_c2 = np.linalg.norm(pad(b, n) - pad(np.sort(np.repeat(np.sqrt(2) * a, 4))[::-1], n))
    return a[a > 1e-14], b, c, d_link, d_c1, d_c2


if __name__ == '__main__':
    print('Kt  r=20Kt  loss        loss*4^Kt   d_link     d_copy1    d_copy2    max-defect/sqrt(loss)  #chain(a)', flush=True)
    for Kt in range(1, 7):
        psi, A, B, QA, QB, pairs = strategy(Kt, Kt + 1)
        l = loss(psi, A, B, QA, QB, pairs)
        a, b, c, dl, d1, d2 = chain_report(psi, A, B, QA)
        print(f'{Kt:2d}  {20*Kt:5d}  {l:.4e}  {l*4**Kt:.4e}  {dl:.3e}  {d1:.3e}  {d2:.3e}  '
              f'{max(dl, d1, d2)/np.sqrt(l):8.3f}               {len(a)}', flush=True)
    # sanity: Schmidt rank of the truncated state and the a-chain values
    psi, A, B, QA, QB, pairs = strategy(5, 6)
    a, b, c, *_ = chain_report(psi, A, B, QA)
    print('a-chain (copy-2 flag Schmidt values) at Kt=5:', np.round(a, 6))
    print('ratios a_k/a_{k+1}:', np.round(a[:-1] / a[1:], 6))
    print('DONE')
