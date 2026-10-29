# Perfect play in a binary-scoring nonlocal game requires infinite dimension

Seth Douglas and Nidhal Mghirbi.

Version 1.1.0, October 2026. DOI: [10.5281/zenodo.23756753](https://doi.org/10.5281/zenodo.23756753). Concept DOI (all versions): [10.5281/zenodo.23754446](https://doi.org/10.5281/zenodo.23754446). A preprint; it has not been peer reviewed.

We construct a finite rational two-player nonlocal game with ten Alice questions, eleven Bob questions, at most six answers per question and 52 equally likely supported pairs. It has a perfect spatial strategy and no perfect finite-dimensional strategy. For arbitrary finite-dimensional local POVMs and mixed states, its dimension profile is D_Gamma(epsilon) = 10 log2(1/epsilon) + O(log log(1/epsilon)). The lower bound is 10 log2(1/epsilon) - 10 log2 log2(1/epsilon) - O(1). A direct square-root magic-square extraction, robust flagged sector law and approximate gluing force twenty Schmidt coefficients per appreciable halving level. The explicit strategy compressed to 20K dimensions has loss exactly 1/[624(4^K-1)], yielding the matching leading upper coefficient 10.

## Contents

- `paper.pdf`: the compiled manuscript.
- `main.tex`, `lower-bound.tex`, `references.bib`: complete sources.
- `arxiv/`: standalone source package, including the same ancillary checks.
- `anc/`: predicate, sequential runner, exact and numerical checks, recorded
  outputs and SHA-256 hashes. The analytic proofs establish the theorems.

Build with pdflatex, bibtex, pdflatex, pdflatex on main.tex, in that order.
For a clean source tree, build a copy in a separate directory.

Version 1.0.0 is archived at [10.5281/zenodo.23754447](https://doi.org/10.5281/zenodo.23754447).
The all-versions record is [10.5281/zenodo.23754446](https://doi.org/10.5281/zenodo.23754446).
This working copy has not been published as a new version.

## License

Paper and documentation: CC BY 4.0. Scripts: MIT or Apache 2.0, at your option.
See RIGHTS.md.
