# Perfect play in a binary-scoring nonlocal game requires infinite dimension

Seth Douglas and Nidhal Mghirbi. Version 1.0.0, October 2026. DOI: [10.5281/zenodo.23754447](https://doi.org/10.5281/zenodo.23754447). Concept DOI (all versions): [10.5281/zenodo.23754446](https://doi.org/10.5281/zenodo.23754446). A preprint; it has not been peer reviewed.

We construct a finite rational two-player nonlocal game with a binary winning predicate that has a perfect strategy on $\ell^2\otimes\ell^2$ and no perfect finite-dimensional strategy. The game has ten questions for Alice, eleven for Bob, at most six answers per question, and a uniform distribution on fifty-two question pairs. We glue two copies of Man\v{c}inska's Flagged Magic-Square game along complementary labels, adding a boundary answer to the second copy. An inverse-free sector argument forces every perfect strategy to contain a Schmidt-coefficient chain closed under halving, which rules out finite rank. An explicit normalized spatial strategy has twenty coefficients at each geometric level. Its compression to $20K$ dimensions has loss exactly $1/[624(4^K-1)]$, giving $D(\eps)\leq10\log_2(1/\eps)+20$ for general local POVMs and $0<\eps<1$. We prove a weaker algebraic lower bound and state a matching logarithmic lower bound as an open problem. Thus a binary-scoring game distinguishes spatial from finite-dimensional perfect attainment even though both values are one.

## Contents
- `paper.pdf`, with its LaTeX source (`main.tex`, `references.bib`).
- `arxiv/`: a standalone source folder with the same source and checks.
- `anc/`: the complete game and its finite checks, with a runner, recorded outputs and SHA-256 hashes (see `anc/README.txt`). The theorems
  rest on the written proofs; the computations support them and do not replace them.

## Related
"The uncomputable price of precision" (Seth Douglas and Nidhal Mghirbi), version 4.0.0,
[10.5281/zenodo.23753741](https://doi.org/10.5281/zenodo.23753741).

## License
The paper and documentation are licensed CC BY 4.0, and the scripts MIT or Apache 2.0, at your option. See RIGHTS.md.
