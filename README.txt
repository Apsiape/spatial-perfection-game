Perfect play in a binary-scoring nonlocal game requires infinite dimension
Seth Douglas and Nidhal Mghirbi.
Version 1.1.0, October 2026. A preprint; it has not been peer reviewed.

main.tex, lower-bound.tex and references.bib are the manuscript sources.
paper.pdf is the compiled manuscript (in the full working copy).
anc/ contains the rational game and reproducible finite checks.
arxiv/ is a standalone source package (in the full working copy).

Build in a clean copy:
  pdflatex main
  bibtex main
  pdflatex main
  pdflatex main

The dimension profile for arbitrary POVMs and mixed states is
D_Gamma(epsilon) = 10 log2(1/epsilon) + O(log log(1/epsilon)).
The lower bound is 10 log2(1/epsilon) - 10 log2 log2(1/epsilon) - O(1).
