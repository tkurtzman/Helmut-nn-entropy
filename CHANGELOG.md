# Changelog

All notable changes to molentropy are recorded here. Versions follow
[semantic versioning](https://semver.org/); before 1.0, a minor version may
change the API or the default estimates.

## [Unreleased]

- Constrained-coordinate detection (`--drop-constrained`,
  `drop_constrained=True` in `load_system`,
  `CoordinateOptions::drop_constrained`): drops every bond and angle that the
  trajectory shows to be constrained, keeps free bonds to hydrogen, and keeps
  the dropped coordinates' terms in the BAT Jacobian, so one run gives both
  the entropy and ln J. `System.constrained_atoms` and
  `InternalCoordinates::constrained_coordinates` list what was dropped.
- Coordinate sets (`molentropy::coordinate_set::CoordinateSet`,
  `--write-coordinates PREFIX` and `--from-coordinates`,
  `System.write_coordinates` and `load_coordinates`): the internal coordinates
  and their values can be written to a definition table and a value file and
  read back without the topology, so they can be generated elsewhere. The
  round trip reproduces the entropy exactly, the Jacobian included.
- `python/bat_phase.py`, a generator for those files: the canonical tree and
  the phase angles built with MDAnalysis, reproducing the crate's own
  coordinates bit for bit.
- A Bayesian network soft estimator (`estimate_network_entropy`,
  `--soft-estimator network`, `soft_estimator="network"`): MIST with each
  coordinate conditioned on up to several earlier ones (5 by default) instead
  of one. On peptides it halves MIST's error; see the validation page.
- Truncated Gaussian hard estimators for the split estimator
  (`--hard-estimator qh-mie1|qh-mie2`, `hard_estimator="qh-mie1"|"qh-mie2"`,
  `HardEstimator::QuasiHarmonicMie1|QuasiHarmonicMie2`): QH with its
  correlation term dropped (first order) or truncated at second order in the
  mutual information expansion.

## [0.1.0] - 2026-10-26

First release, under the name molentropy (previously nn-entropy).

- Reads Amber topologies (`.parm7`) and NetCDF trajectories (`.nc`) and converts
  every frame to bond-angle-torsion coordinates, with phase angles and an
  optional Jacobian term.
- Estimators:
  - nearest-neighbor mutual information expansion (NN MIE) to fourth order, and
    pairwise mutual information;
  - mutual information spanning tree (MIST);
  - joint nearest-neighbor entropy up to eight dimensions;
  - quasi-harmonic (QH) and Boltzmann quasi-harmonic (BQH);
  - a split estimator: soft torsions by MIE, MIST or the joint estimate, stiff
    coordinates by QH or BQH, and a Gaussian (Fourier-harmonic) or per-state
    coupling term.
- Per-coordinate entropies for every estimator except the joint estimate.
- Theiler windows for time-correlated frames, and exact finite-sample constants.
- Results are bit-for-bit reproducible across runs and thread counts.
- A command-line program, a Rust library and Python bindings.
- Validation against HREX/MBAR reference entropies of small molecules and
  peptides at 300 K.

[0.1.0]: https://github.com/helmutcarter/molentropy/releases/tag/v0.1.0
