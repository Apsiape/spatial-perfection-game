"""Refutation-first checks of the robust FMS sector law (RSL):
    || Sch(psi_MS) - (4 x sqrt2 Sch(psi_f)) ||_2 <= C sqrt(loss)    (dimension-free C).
perturbation_cases: random local perturbations of ideal (x) junk (junk with random spectrum); ratio defect/sqrt(loss).
optimized_cases: states that VIOLATE the law (wrong flag/MS ratio, or mismatched junk spectra); measurements optimized by a
    see-saw of SDPs (POVMs).  A refutation would be: small loss with a large defect.
Single-threaded, small.
"""
import sys
import numpy as np
import cvxpy as cp
from fms_core import (ideal_strategy, loss, sector_law_defect, QA, QB, PAIRS, wins)

rng = np.random.default_rng(12345)


def rand_herm(d):
    M = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return (M + M.conj().T) / 2


def expm_h(H, t):
    w, v = np.linalg.eigh(H)
    return (v * np.exp(1j * t * w)) @ v.conj().T


def with_junk(K, A, B, chi):
    m = chi.shape[0]
    K2 = np.kron(K, chi)
    A2 = {q: [np.kron(E, np.eye(m)) for E in A[q]] for q in A}
    B2 = {r: [np.kron(F, np.eye(m)) for F in B[r]] for r in B}
    return K2, A2, B2


def perturb(K, A, B, t):
    d = K.shape[0]
    A2 = {}
    for q in A:
        U = expm_h(rand_herm(d), t)
        A2[q] = [U @ E @ U.conj().T for E in A[q]]
    B2 = {}
    for r in B:
        U = expm_h(rand_herm(d), t)
        B2[r] = [U @ F @ U.conj().T for F in B[r]]
    K2 = K + t * (rng.normal(size=K.shape) + 1j * rng.normal(size=K.shape)) / d
    K2 /= np.linalg.norm(K2)
    return K2, A2, B2


def perturbation_cases():
    print('perturbation_cases: perturbations of ideal (x) junk', flush=True)
    K0, A0, B0 = ideal_strategy()
    worst = 0
    for m in [1, 2, 3]:
        for trial in range(6):
            lam = rng.exponential(size=m); lam = np.sqrt(lam / lam.sum())
            U = np.linalg.qr(rng.normal(size=(m, m)))[0]; V = np.linalg.qr(rng.normal(size=(m, m)))[0]
            chi = U @ np.diag(lam) @ V.T
            K, A, B = with_junk(K0, A0, B0, chi)
            for t in [1e-1, 3e-2, 1e-2, 3e-3]:
                K2, A2, B2 = perturb(K, A, B, t)
                l = loss(K2, A2, B2)
                dfc = sector_law_defect(K2, A2)[0]
                r = dfc / np.sqrt(max(l, 1e-30))
                worst = max(worst, r)
            print(f'  m={m} trial={trial}: last t={t:.0e} loss={l:.2e} defect={dfc:.2e} ratio={r:.2f}', flush=True)
    print(f'perturbation_cases worst defect/sqrt(loss) = {worst:.3f}', flush=True)
    return worst


def best_alice(K, B):
    d = K.shape[0]; KH = K.conj().T
    A = {}
    tot = 0
    for q in QA:
        W = [np.zeros((d, d), complex) for _ in QA[q]]
        for r in QB:
            for ia, a in enumerate(QA[q]):
                for ib, b in enumerate(QB[r]):
                    if not wins(q, r, a, b):
                        W[ia] += K @ B[r][ib].T @ KH
        n = len(QA[q])
        Es = [cp.Variable((d, d), hermitian=True) for _ in range(n)]
        cons = [E >> 0 for E in Es] + [sum(Es) == np.eye(d)]
        obj = cp.Minimize(cp.real(sum(cp.trace(E @ ((W[i] + W[i].conj().T) / 2)) for i, E in enumerate(Es))))
        prob = cp.Problem(obj, cons); prob.solve(solver=cp.CLARABEL, max_threads=1)
        A[q] = [(E.value + E.value.conj().T) / 2 for E in Es]
        tot += prob.value
    return A, tot / len(PAIRS)


def best_bob(K, A):
    d = K.shape[1]; KH = K.conj().T
    B = {}
    tot = 0
    for r in QB:
        W = [np.zeros((d, d), complex) for _ in QB[r]]
        for q in QA:
            for ia, a in enumerate(QA[q]):
                M = (KH @ A[q][ia] @ K).T
                for ib, b in enumerate(QB[r]):
                    if not wins(q, r, a, b):
                        W[ib] += M
        n = len(QB[r])
        Fs = [cp.Variable((d, d), hermitian=True) for _ in range(n)]
        cons = [F >> 0 for F in Fs] + [sum(Fs) == np.eye(d)]
        obj = cp.Minimize(cp.real(sum(cp.trace(F @ ((W[i] + W[i].conj().T) / 2)) for i, F in enumerate(Fs))))
        prob = cp.Problem(obj, cons); prob.solve(solver=cp.CLARABEL, max_threads=1)
        B[r] = [(F.value + F.value.conj().T) / 2 for F in Fs]
        tot += prob.value
    return B, tot / len(PAIRS)


def seesaw(K, A, B, iters=30):
    l = loss(K, A, B)
    for it in range(iters):
        A, _ = best_alice(K, B)
        B, l2 = best_bob(K, A)
        if l - l2 < 1e-12 * max(1, l):
            l = l2; break
        l = l2
    return A, B, l


def optimized_cases():
    print('optimized_cases: law-violating states, see-saw optimized measurements', flush=True)
    K0, A0, B0 = ideal_strategy()
    worst = 0
    rows = []
    # (a) wrong flag weight, no junk
    for rho in [1.0, 1.2, 1.35, 1.40, 1.414, 1.43, 1.5, 1.7, 2.0]:
        K = np.zeros((5, 5), complex); K[0, 0] = 1
        for z in range(1, 5):
            K[z, z] = rho
        K /= np.linalg.norm(K)
        A, B, l = seesaw(K, A0, B0)
        dfc = sector_law_defect(K, A)[0]
        r = dfc / np.sqrt(max(l, 1e-30)); worst = max(worst, r)
        rows.append(('ratio', rho, l, dfc, r))
        print(f'  MS/flag ratio {rho:.3f}: seesaw loss {l:.3e}  defect {dfc:.3e}  defect/sqrt(loss) {r:.2f}', flush=True)
    # (b) mismatched junk spectra: flag junk (1, u), MS junk (1, v) -> law needs u = v
    for (u, v) in [(1.0, 1.0), (0.5, 0.5), (0.5, 0.45), (0.5, 0.3), (0.5, 0.0), (1.0, 0.5)]:
        m = 2
        K = np.zeros((10, 10), complex)
        jf = np.array([1, u]) / np.sqrt(1 + u * u); jm = np.array([1, v]) / np.sqrt(1 + v * v)
        for k in range(m):
            K[0 * m + k, 0 * m + k] = (1 / 3) * jf[k]
            for z in range(1, 5):
                K[z * m + k, z * m + k] = (np.sqrt(2) / 3) * jm[k]
        K /= np.linalg.norm(K)
        Aj = {q: [np.kron(E, np.eye(m)) for E in A0[q]] for q in A0}
        Bj = {r: [np.kron(F, np.eye(m)) for F in B0[r]] for r in B0}
        A, B, l = seesaw(K, Aj, Bj)
        dfc = sector_law_defect(K, A)[0]
        r = dfc / np.sqrt(max(l, 1e-30)); worst = max(worst, r)
        print(f'  junk (1,{u})/(1,{v}): seesaw loss {l:.3e}  defect {dfc:.3e}  defect/sqrt(loss) {r:.2f}', flush=True)
    print(f'optimized_cases worst defect/sqrt(loss) = {worst:.3f}', flush=True)
    return worst


if __name__ == '__main__':
    w1 = perturbation_cases()
    w2 = optimized_cases()
    print('DONE', w1, w2)
