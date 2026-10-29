"""Build the manuscript with plain LaTeX and BibTeX outside the source tree."""
from pathlib import Path
import os,subprocess,shutil,re
R=Path(__file__).resolve().parent;E=R.with_name(R.name+'-evidence')/'tex';E.mkdir(parents=True,exist_ok=True)
assert not E.resolve().is_relative_to(R.resolve())
env={**os.environ,'SOURCE_DATE_EPOCH':'1793232000','FORCE_SOURCE_DATE':'1'}
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
latex=['pdflatex','-no-shell-escape','-recorder','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(E),'main.tex']
def run(args,cwd=R,extra=None):
 x=subprocess.run(args,cwd=cwd,env={**env,**(extra or {})},capture_output=True,timeout=600)
 if x.returncode:raise RuntimeError(x.stdout.decode(errors='replace')[-6000:])
run(latex);run(['bibtex','main'],E,{'BIBINPUTS':str(R)+os.pathsep})
shutil.copy2(E/'main.bbl',R/'main.bbl')
for _ in range(3):run(latex)
bad=re.compile(r'pdfTeX warning|(?:LaTeX(?: Font)?|Package .+|Class .+) Warning:|undefined|Overfull|Underfull|multiply defined|Rerun to get|Float too large|ignored error|Missing character|^!',re.I)
issues=[s for s in (E/'main.log').read_text(errors='replace').splitlines() if bad.search(s)]
issues += [s for s in (E/'main.blg').read_text(errors='replace').splitlines() if 'Warning--' in s or 'error message' in s]
if issues:raise RuntimeError('\n'.join(issues))
shutil.copy2(E/'main.pdf',R/'paper.pdf');shutil.copy2(E/'main.bbl',R/'main.bbl')
print('Strict build passed; PDF and fresh BBL copied to the source tree.')
