"""Exact oscillator resource identities and finite entropy tail checks."""
import os,json,math
from pathlib import Path
from fractions import Fraction as Q
import sympy as s
z=s.symbols('z')
pgf=(sum(z**r for r in range(16))/24+sum(z**r for r in range(16,20))/48)/(1-z**20/4)
assert s.simplify(pgf.subs(z,1))==1
moments=[];g=pgf
for k in range(1,7):
    g=z*s.diff(g,z);moments.append(s.factor(g.subs(z,1)))
assert moments[0]==s.Rational(275,18)
assert moments[1]==s.Rational(23765,54)
rows=[]
for K in range(1,21):
    probs=[Q(1,24*4**k) for k in range(K) for _ in range(16)]+[Q(1,48*4**k) for k in range(K) for _ in range(4)]
    probs.sort(reverse=True);mass=sum(probs)
    assert mass==1-Q(1,4**K)
    energy=sum(n*p for n,p in enumerate(probs))/mass
    expected=Q(275,18)-Q(20*K,4**K-1)
    assert energy==expected
    normalized=[p/mass for p in probs]
    entropy=-sum(float(p)*math.log2(float(p)) for p in normalized)
    expected_entropy=math.log2(24)+7/9-2*K/(4**K-1)+math.log2(1-4.0**(-K))
    assert abs(entropy-expected_entropy)<1e-13
    rows.append({'levels':K,'rank':20*K,'energy':str(energy),'entropy_bits':entropy})
H=math.log2(24)+7/9
assert abs(H-5.362740278498934)<1e-14
probs=sorted([Q(1,24*4**k) for k in range(100) for _ in range(16)]+[Q(1,48*4**k) for k in range(100) for _ in range(4)],reverse=True)
numeric_H=-sum(float(p)*math.log2(float(p)) for p in probs)
assert abs(numeric_H-H)<1e-13
# A decreasing spectrum obeys p_i <= 1/i. This checks the stated tail
# inequality on finite distributions; tightness/compactness are analytic.
for n in (20,100,1000):
    weights=[Q(1,i*i) for i in range(1,n+1)];total=sum(weights)
    weights=[p/total for p in weights];entropy=-sum(float(p)*math.log2(float(p)) for p in weights)
    for cutoff in (2,4,8,16):
        assert float(sum(weights[cutoff:]))*math.log2(cutoff)<=entropy+1e-14
out={'status':'PASS','number_pgf':str(pgf),'number_moments':[str(x) for x in moments],
     'entropy_bits':H,'numeric_entropy_bits':numeric_H,'compressions':rows,
     'scope':'Exact resource identities and finite tests, not certification of infinite-dimensional analytic proofs or global optima.'}
folder=Path(os.environ.get('GAME_CHECK_OUTPUT','.')).resolve();folder.mkdir(parents=True,exist_ok=True)
(folder/'energy-checks.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
