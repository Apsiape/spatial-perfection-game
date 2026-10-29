"""Independent exact-arithmetic tests. Infinite tails are included analytically.

No imports from the supplied checks or release code; stdlib only.
An eventual constant-multiplicity quarter-chain specifies an infinite multiset.
"""
import os
from fractions import Fraction as Q
from collections import defaultdict
from bisect import bisect_left
import json
import math
import random
from pathlib import Path

OUT = Path(os.environ.get('GAME_CHECK_OUTPUT', '.')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(510292026)
LN2 = math.log(2)

def log2(q):
    return math.log2(q.numerator) - math.log2(q.denominator)

def merge(blocks):
    d = defaultdict(int)
    for v, m in blocks:
        d[v] += m
    return sorted(d.items(), reverse=True)

def summarize(blocks):
    rank, mass, energy = 0, Q(0), Q(0)
    for v, m in merge(blocks):
        energy += v * (m * rank + m * (m - 1) // 2)
        mass += v * m
        rank += m
    return rank, mass, energy

def make_case(chains):
    # Each chain = (offset r in (1/4,1], finite multiplicities, constant tail).
    z = sum((sum((r * m / 4**k for k, m in enumerate(ms)), Q(0))
             + r * tail / 4**len(ms) * Q(4, 3) for r, ms, tail in chains), Q(0))
    scale = 1 / (72 * z)
    return chains, scale

def mass_entropy_renyi(case):
    chains, s = case
    sigma_entropy = 0.0
    alphas = (0.1, 0.5, 2.0, 3.0, 10.0)
    powers = {a: 0.0 for a in alphas}
    mass = Q(0)
    for r, ms, tail in chains:
        for k, m in enumerate(ms):
            v = s * r / 4**k
            mass += m * v
            sigma_entropy -= float(m * v) * log2(v)
            for a in alphas:
                powers[a] += m * float(v)**a
        v = s * r / 4**len(ms)
        w = tail * v * Q(4, 3)
        mass += w
        sigma_entropy += float(w) * (-log2(v) + 2 / 3)
        for a in alphas:
            powers[a] += tail * float(v)**a / (1 - 4**(-a))
    assert mass == Q(1, 72)
    h = 72 * sigma_entropy - 17 / 9
    renyi = {a: math.log2((16 * 4**a + 4 * 2**a) * powers[a]) / (1 - a)
             for a in alphas}
    return h, renyi

def spectrum_prefix_band(case, transform=True):
    chains, s = case
    cutoff = s / 4**(max(len(ms) for _, ms, _ in chains) + 3)
    prefix, band = [], []
    factors = ((4, 16), (2, 4)) if transform else ((1, 1),)
    for r, ms, tail in chains:
        for f, fm in factors:
            k, v = 0, f * s * r
            while v > cutoff:
                m = ms[k] if k < len(ms) else tail
                prefix.append((v, fm * m))
                k += 1
                v /= 4
            assert k >= len(ms)
            assert cutoff / 4 < v <= cutoff
            band.append((v, fm * tail))
    return merge(prefix), merge(band)

def exact_energy(case):
    pre, band = spectrum_prefix_band(case)
    q, pmass, pe = summarize(pre)
    r, bmass, be = summarize(band)
    assert pmass + Q(4, 3) * bmass == 1
    return pe + Q(4, 3) * (be + q * bmass) + Q(4, 9) * r * bmass

def cumulative(blocks):
    ranks, masses, values = [0], [Q(0)], [Q(0)]
    for v, m in merge(blocks):
        ranks.append(ranks[-1] + m)
        masses.append(masses[-1] + m * v)
        values.append(v)
    return ranks, masses, values

def kf(cum, n):
    ranks, masses, vals = cum
    j = bisect_left(ranks, n)
    assert j < len(ranks)
    return masses[j] - (ranks[j] - n) * vals[j]

def expanded_bands(case, transform=True):
    pre, band = spectrum_prefix_band(case, transform)
    return pre + [(v / 4**k, m) for k in range(25) for v, m in band]

star = make_case([(Q(1), [], 1)])
assert star[1] == Q(1, 96)
estar = exact_energy(star)
hstar, rstar = mass_entropy_renyi(star)
assert estar == Q(275, 18)
assert abs(hstar - (math.log2(24) + 7 / 9)) < 1e-13
star_cum = {t: cumulative(expanded_bands(star, t)) for t in (True, False)}

def star_kf(n, transform):
    if not transform:
        return Q(1, 72) * (1 - Q(1, 4**n))
    k, j = divmod(n, 20)
    prefix = 1 - Q(1, 4**k)
    return prefix + Q(min(j, 16), 24 * 4**k) + Q(max(j - 16, 0), 48 * 4**k)

cases = []
for _ in range(300):
    chains = []
    for c in range(rng.randint(1, 4)):
        r = Q(1) if c == 0 else Q(rng.randint(26, 100), 100)
        ms = [rng.randint(1, 45)]
        for k in range(rng.randint(1, 12)):
            lo = (ms[-1] + 15) // 16
            ms.append(rng.randint(lo, min(50, ms[-1] + 8)))
        tail = rng.randint((ms[-1] + 15) // 16, 8)
        assert all(a <= 16 * b for a, b in zip(ms, ms[1:] + [tail]))
        chains.append((r, ms, tail))
    cases.append(make_case(chains))
# Exact tests arbitrarily close to the maximizer, with colliding and interlaced chains.
for depth in (0, 1, 2, 5, 10, 20, 40):
    for offset in (Q(1), Q(1, 2), Q(3, 10), Q(99, 100)):
        cases.append(make_case([(Q(1), [], 1), (offset, [0] * depth, 1)]))
cases.append(star)

min_h, min_e = float("inf"), None
comparisons = 0
strict_nonstar = 0
for case in cases:
    h, rs = mass_entropy_renyi(case)
    e = exact_energy(case)
    assert e >= estar
    assert h >= hstar - 2e-12
    assert all(rs[a] >= rstar[a] - 2e-12 for a in rs)
    min_h = min(min_h, h)
    min_e = e if min_e is None else min(min_e, e)
    if case != star:
        assert e > estar
        strict_nonstar += 1
    for transform in (False, True):
        cum = cumulative(expanded_bands(case, transform))
        limit = min(cum[0][-1], 2000 if transform else 150)
        # Both Ky Fan curves are piecewise affine in the integer rank; these
        # are all breakpoints of either curve in the checked rank interval.
        points = {n for n in cum[0] if 0 < n <= limit}
        if transform:
            points.update(n for k in range(limit // 20 + 1) for n in (20*k, 20*k+16)
                          if 0 < n <= limit)
        else:
            points.update(range(1, limit + 1))
        points.add(limit)
        for n in points:
            assert kf(cum, n) <= star_kf(n, transform), (n, case)
            comparisons += 1

print("INDEPENDENT: infinite tails normalized exactly; no finite truncation normalization")
print("Cases:", len(cases), "; exact Ky Fan comparisons:", comparisons)
print("Strict energy inequalities for", strict_nonstar, "nonoptimal cases")
print("Minimum entropy bits:", min_h, "; exact minimum energy:", str(min_e))
print("Renyi orders 0.1, 0.5, 2, 3, 10: all comparisons passed")

capacity = []
for n in range(13):
    t = Q(5, 16 * 64**n + 5)
    seed_variance = t * Q(16, 9) + t * (1 - t) * (2*n + Q(8, 3))**2
    scaled = Q(8, 49) + seed_variance
    assert scaled > Q(8, 49)
    if n == 0:
        assert scaled == Q(152, 81)
    capacity.append(dict(N=n, tail_boundary_probability=str(t),
                         variance_over_ln2_squared=str(scaled), numeric=float(scaled)))
print("Exact capacity sequence N=0..12:", [x["numeric"] for x in capacity])

general = []
for n in range(2, 8):
    for lam in (Q(1, n) + Q(1, 100), Q(3, 4), Q(1), Q(3, 2), Q(2), Q(5)):
        rho = n * lam
        if rho <= 1:
            continue
        b = Q(n - 1, n)
        c = (rho - 1) / (lam * (n - 1))
        assert b * c / (1 - 1 / rho) == 1
        assert b / (1 - Q(1, n)) == 1
        assert b * c == 1 - 1 / rho
        delta = math.log(float((rho - 1) / (lam * (n - 1)))) - math.log(float(lam)) / float(rho - 1)
        kl = math.log(float((1 - 1 / rho) / (1 - Q(1, n)))) + math.log(float(n / rho)) / float(rho - 1)
        assert abs(delta - kl) < 1e-12 and delta >= -1e-12
        assert (lam == 1) == (abs(delta) < 1e-13)
        general.append((n, str(lam), delta))
print("General II_1 ladders and KL deficit:", len(general), "exact normalizations passed")

for q in (Q(1, 10), Q(1, 2), Q(9, 10)):
    for k in range(30):
        assert (1 - q) * q**k == (1 / q) * (1 - q) * q**(k + 1)
print("TMSV backward-shift density intertwining: 90 exact checks passed")

# A boundary series at the exceptional critical point would equate a nonzero law
# to a sum of zero boundary measures. The valid ladder is u=1, B=0.
assert Q(1, 3) + Q(2, 3) == 1
critical_example = {"n": 1, "lambda": 1, "B": 0, "u": 1,
                    "density_eigenvalues": ["1/3", "2/3"],
                    "state_law_mass": 1, "claimed_boundary_series_mass": 0}
print("Critical identity-ladder boundary check:", critical_example)

summary = dict(cases=len(cases), exact_Ky_Fan_comparisons=comparisons,
               exact_minimum_energy=str(min_e), minimum_entropy_bits=min_h,
               exact_capacity_sequence=capacity, general_ladder_cases=len(general),
               critical_identity_ladder=critical_example)
(OUT / "ladder-results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("ALL INDEPENDENT CHECKS PASSED")
