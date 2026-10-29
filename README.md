# Perfect play in a binary-scoring nonlocal game requires infinite dimension but only finite energy

Seth Douglas and Nidhal Mghirbi.

Version 2.0.0, October 2026. DOI: [10.5281/zenodo.23772890](https://doi.org/10.5281/zenodo.23772890). Concept DOI (all versions): [10.5281/zenodo.23754446](https://doi.org/10.5281/zenodo.23754446). A preprint; it has not been peer reviewed.

## Abstract

A finite rational binary-scoring game can require infinite dimension for perfect play while admitting finite energy and finite entanglement entropy. Our game has ten Alice questions, eleven Bob questions, at most six answers, and 52 uniform question pairs. It is spatially perfect and never finitely perfect, with sharp dimension profile $D_\Gamma(\varepsilon)=10\log_2(1/\varepsilon)+O(\log\log(1/\varepsilon))$. Among pure perfect strategies, the displayed strategy attains the least entanglement entropy, $5.362740278498934$ ebits, and least mean occupation, $275/18$ per oscillator. We classify all perfect normal type-I marginal spectra and prove that this spectrum is the unique majorization maximum. These constants are also the small-loss limits of the entropy and one-sided oscillator energy prices. In the massive free scalar field we construct a pure perfect state in two strictly separated bounded regions, equal to the vacuum outside a bounded containing region and with every polynomial energy moment finite. In contrast, every fixed faithful split state has a positive loss floor; the complementary-wedge vacuum has supremum one without attainment. Energy-analytic states, bounded-region invertible vacuum preparations and product-Kraus preparations cannot be perfect. Every perfect pure spatial strategy has infinite Schmidt rank; finite entropy is attained but not forced. In every type-II$_1$ factor the normalized trace has a loss floor, while bounded faithful prepared densities win perfectly with sharp trace-entropy deficit $\Delta_* = \ln(7/6)-\ln(2)/7$. Type-II$_\infty$ permits every finite trace entropy. A graded modular ladder gives a normal semifinite hosting criterion. Its scalar fingerprint has sharp capacity and entropy-gap bounds; the capacity floor is approached but not attained in type I, and attained in type II.

## Sources and checks

The main manuscript and its lower-bound source retain the two separation and sharp-profile theorems. The additional sources cover oscillators, the localized field state, faithfulness and preparation, entropy versus dimension, and semifinite densities and their spectral law. `paper.pdf` is the compiled manuscript. `arxiv/` builds independently. `anc/` contains the rational game, finite checks, recorded outputs, a sequential runner and LF SHA-256 manifests. The mathematical proofs establish the universal statements.

Build in a clean copy with pdflatex, bibtex, pdflatex and pdflatex. The provided build helper keeps intermediate products outside this source tree. Arbitrary bounded local operations are part of the mathematical model; no Gaussian-only implementation or microscopic control synthesis is claimed.

Earlier versions: 1.1.0, [10.5281/zenodo.23756753](https://doi.org/10.5281/zenodo.23756753); 1.0.0, [10.5281/zenodo.23754447](https://doi.org/10.5281/zenodo.23754447). All versions: [10.5281/zenodo.23754446](https://doi.org/10.5281/zenodo.23754446).

The exact type-I spectral classification and majorization theorem prove the global pure-state entropy and oscillator optima. The small-loss one-sided energy proof permits an unrestricted Bob system. The general ladder law retains its critical exception, moment conditions and support-transport convention. The capacity floor is an unattained type-I infimum and a type-II minimum.

Text and documentation: CC BY 4.0. Scripts: MIT or Apache 2.0; see RIGHTS.md.
