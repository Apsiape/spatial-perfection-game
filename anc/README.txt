Finite checks for the game in the accompanying paper

Files:
  game.py       Standard-library generator for the complete rational game.
  game.json     Question alphabets, supported pairs and all forbidden events.
  checks.py     Exact symbolic and supplemental finite numerical checks.
  results.json  Recorded results, including every supported pair at each size.
  checks.log    Recorded console output.
  run.py        Sequential runner with checksum verification.
  SHA256SUMS    SHA-256 hashes of all other files in this directory.

Requirements: Python 3.10 or later, NumPy and SymPy.
Run from any directory:
  python -B run.py --output <new-results-directory>

All numerical thread-count environment variables are set to one before imports.
Numerical matrices have dimension at most 120. The runner verifies the shipped
records before running; fresh outputs go to the directory supplied by the user.
The game generator itself needs no external package.

The checks reconstruct the ideal magic square and flagged measurements, verify
every forbidden base product symbolically, and check the incomplete-block loss.
Finite compression checks cover all 52 pairs at dimensions 20,40,...,120.
They verify completeness, numerical positivity, normalization and exact-loss
agreement to the tolerance recorded in checks.py. The JSON predicate is also
compared exhaustively with the independently reconstructed semantic predicate.

The infinite strategy, the sector identity, absence of finite perfect strategies,
and the algebraic profile lower bound require the proofs in the paper. These
programs are not a proof-assistant formalization or a global optimization oracle.
