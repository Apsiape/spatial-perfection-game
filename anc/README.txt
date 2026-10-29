Version 2.0.0, October 2026. A preprint; it has not been peer reviewed.
Finite checks for the game in the accompanying paper

Files:
  game.py       Standard-library generator for the complete rational game.
  game.json     Question alphabets, supported pairs and all forbidden events.
  checks.py     Exact symbolic and supplemental finite numerical checks.
  results.json  Recorded results, including every supported pair at each size.
  checks.log    Recorded console output.
  run.py        Sequential runner with checksum verification.
  SHA256SUMS    SHA-256 hashes of all other files in this directory.
  fms_core.py, check_bridge.py, check_rsl.py, gamma_chain.py
                Numerical gadget, bridge, sector and glued-strategy checks.
  exact_checks.py
                Independent exact bridge words, Pauli group, trace-norm
                counterexample and supplemental finite algebraic checks.
  check_extraction.py
                Regular-representation extraction on complex perturbations.
  check_descent.py
                Positive descent levels, multiplicities and local dilation.
  *.log         Recorded output of each corresponding program.
  resources.json
                Measured runtime and sampled combined memory for each program.

Requirements: Python 3.10 or later, NumPy, SymPy, CVXPY with Clarabel,
psutil, SciPy and threadpoolctl.
Run from any directory:
  python -B run.py --output <new-results-directory>

Numerical thread counts are set to one before imports, and every Clarabel call
explicitly uses max_threads=1. The runner stops a child if sampled combined
resident memory reaches 850 MB. The largest extraction matrix is 384 by 384.
The runner verifies the shipped records before running; fresh outputs go to
the directory supplied by the user.
The game generator itself needs no external package.

The checks reconstruct the ideal magic square and flagged measurements, verify
every forbidden base product symbolically, and check the incomplete-block loss.
Finite compression checks cover all 52 pairs at dimensions 20,40,...,120.
They verify completeness, numerical positivity, normalization and exact-loss
agreement to the tolerance recorded in checks.py. The JSON predicate is also
compared exhaustively with the independently reconstructed semantic predicate.

The extra numerical ratios use mean FMS loss, whereas the sector theorem uses
the largest per-pair loss bound. The optimized examples are local searches,
with finite solver accuracy; their ratios need not equal ten at tiny errors.
The exact wrong-ratio identity supplies an independent square-root scaling
check. Entropy, product-simplex and formal equal-ratio multiset calculations
in exact_checks.py are supplemental examples, not additional game theorems.
The optional GPU-discovery warning in two numerical logs is preserved with
its module-relative filename; no machine-specific path is distributed.
All computations here use CPU arrays and the CPU Clarabel solver.

The infinite strategy, the sector law, gluing and arbitrary-strategy logarithmic
dimension lower bound require the analytic proofs in the paper. These programs
are not a proof-assistant formalization or a global optimization oracle.

Resource and operator-algebra checks:
  energy_checks.py: exact generating-function moments, passive compression
    energy, entropy and finite entropy-tail examples.
  field_checks.py: rational game reconstruction, forbidden infinite-matrix
    entries, cubic storage enumeration, smooth-mode energy inequalities
    and finite symplectic completion.
  faithfulness_checks.py: exact classical value, testing inequalities,
    finite-algebra compression and negative mirror examples.
  semifinite_checks.py and semifinite_model_checks.py: independent exact
    gadget products, finite matrix models, trace/entropy sector identities,
    scalar-boundary equality and comparison with the original compression.
  fingerprint_checks.py: exact scalar convolution, moments, released-state
    boundary law and 400 finite boundary-mixture examples.
The associated JSON and log files are recorded outputs. The runner invokes
all sixteen scripts sequentially. Every new script uses GAME_CHECK_OUTPUT
when set, otherwise the current directory, for its output directory.
See ADAPTATIONS.txt for the portable source and scope adjustments.
No finite calculation certifies the continuum or universal algebraic claims.

The ladder_checks.py program uses exact rational infinite-tail normalization,
112485 Ky Fan comparisons on 329 seeded cases, full-tail passive energies,
positive-order entropy comparisons, and exact capacity gaps through level 12.
It also checks general finite-factor ladder constants, backward-shift identities
and the critical identity ladder where a zero boundary has no normalized law.
Its output is ladder-results.json and ladder_checks.log. The universal results
and one-sided limiting-price compactness require the written proofs.
