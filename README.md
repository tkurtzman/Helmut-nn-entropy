# molentropy

[![CI](https://github.com/helmutcarter/molentropy/actions/workflows/ci.yml/badge.svg)](https://github.com/helmutcarter/molentropy/actions/workflows/ci.yml)

Estimate the configurational entropy of a molecular system, with a Rust library, CLI, and Python bindings.

The crate converts Amber trajectories to internal coordinates (bond lengths, bond angles, and torsions) and estimates their entropy. File-based calculations use linear metrics for bonds and angles and a 2π-periodic metric for torsions.

## Features
- Estimators: the nearest-neighbor mutual information expansion (NN MIE) up to fourth order, with pairwise mutual information; the mutual information spanning tree (MIST) and its extension to a Bayesian network; quasi-harmonic (QH) and Boltzmann quasi-harmonic (BQH) entropy; and a split estimator that treats soft torsions non-parametrically and stiff coordinates with a Gaussian. Every estimator but the joint nearest-neighbor estimate can also report each coordinate's share of the entropy.
- Validated against thermodynamic reference entropies; see [Validation](https://helmutcarter.github.io/molentropy/validation.html).
- A CLI that reads `.parm7` + `.nc`, a Rust library, and Python bindings via `pyo3` for in-memory arrays or file-based calculation.
- A [user guide](https://helmutcarter.github.io/molentropy/) covering the CLI, Python and Rust interfaces and the methods.
- Calculations parallelized with `rayon`.

## Requirements
- Rust 1.94 or newer.
- For Python bindings: a Python environment with build tooling for `pyo3` (see below).

## Install

Prebuilt Python wheels are attached to each [release](https://github.com/helmutcarter/molentropy/releases); see [Installation](https://helmutcarter.github.io/molentropy/installation.html). To build from source:

```bash
cargo build --release
```

For the Python module, use [maturin](https://www.maturin.rs/): `maturin develop --release`.

## Quick start

From the command line:

```bash
cargo run --release <path_to_parm7> <path_to_nc> [--torsions-only] [--no-periodic] [--exact-constant] [--start N] [--stop N] [--stride N] [--mie-order 1|2|3|4] [--theiler-window W] [--tree-bonds] [--bonds heavy|tree|tree-unconstrained] [--no-phase-angles] [--jacobian] [--estimator nn|qh|bqh|split] [--histogram-bins N] [--periodic-correlation complex|centered] [--per-coordinate] [--mutual-information] [--soft topology|detect|LIST] [--soft-estimator mie|mist|network|joint] [--network-parents N] [--hard-estimator qh|bqh|qh-mie1|qh-mie2] [--coupling gaussian|states|none] [--soft-features complex|cossin|harmonics] [--harmonics N] [--state-torsions auto|LIST] [--min-state-frames N]
```

Example:

```bash
cargo run --release <path_to_parm7> <path_to_nc>
```

For example, Boltzmann quasi-harmonic entropy, the most accurate estimator here:

```bash
cargo run --release system.parm7 trajectory.nc --estimator bqh
```

From Python:

```python
import molentropy

system = molentropy.load_system("system.parm7", "trajectory.nc")
entropy = system.estimate_quasi_harmonic_entropy()["boltzmann_quasi_harmonic"]
```

The [user guide](https://helmutcarter.github.io/molentropy/) documents every [option](https://helmutcarter.github.io/molentropy/cli.html), the [Python](https://helmutcarter.github.io/molentropy/python.html) and [Rust](https://helmutcarter.github.io/molentropy/rust.html) interfaces, and the [methods](https://helmutcarter.github.io/molentropy/coordinates.html).

Results are differential entropies of the internal coordinates in units of k_B, meaningful as differences between ensembles of the same molecule, not absolute thermodynamic entropies; see [Interpretation and limitations](https://helmutcarter.github.io/molentropy/limitations.html).

## Validation

On the butane end states of an alchemical transformation, BQH with phase angles agrees with thermodynamic integration within the reference's uncertainty. The [validation page](https://helmutcarter.github.io/molentropy/validation.html) has the results for every estimator and an example program that reproduces them.

## Documentation

The user guide is an mdBook in `docs/`, published at <https://helmutcarter.github.io/molentropy/> from the main branch. To read it locally, run `mdbook serve docs`. The Rust API documentation is built with `cargo doc --no-deps --open`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the test suite, CI, and the layout of the code.

## Citing

If you use molentropy in published work, please cite the release you used. Each [release](https://github.com/helmutcarter/molentropy/releases) will eventually be archived on Zenodo with its own DOI; [`CITATION.cff`](CITATION.cff) has the citation metadata, and GitHub's "Cite this repository" button formats it. Changes between releases are listed in [CHANGELOG.md](CHANGELOG.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.