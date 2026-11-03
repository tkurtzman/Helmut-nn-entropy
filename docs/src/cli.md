# Command line

```bash
cargo run --release <path_to_parm7> <path_to_nc> [--torsions-only] [--no-periodic] [--exact-constant] [--start N] [--stop N] [--stride N] [--mie-order 1|2|3|4] [--theiler-window W] [--tree-bonds] [--bonds heavy|tree|tree-unconstrained] [--tree mass|canonical] [--no-phase-angles] [--jacobian] [--estimator nn|qh|bqh|split] [--histogram-bins N] [--periodic-correlation complex|centered] [--per-coordinate] [--mutual-information] [--soft topology|detect|LIST] [--soft-estimator mie|mist|network|joint] [--network-parents N] [--hard-estimator qh|bqh|qh-mie1|qh-mie2] [--coupling gaussian|states|nn|none] [--soft-features complex|cossin|harmonics] [--harmonics N] [--state-torsions auto|LIST] [--min-state-frames N] [--write-coordinates PREFIX] [--coordinate-format npy|tsv]
```

Example:

```bash
cargo run --release <path_to_parm7> <path_to_nc>
```

## Options

- For most molecules `--estimator bqh` is the most accurate choice; see [Validation](validation.md). The default estimator is the NN MIE.
- `--torsions-only` drops the bonds and angles and estimates the entropy of the torsions alone.
- `--stop` limits the number of frames read to N.
- `--start` skips the first N frames.
- `--stride` retains every Nth frame after `--start` (default: 1). Choose a stride based on a separate correlation-time analysis; the crate does not estimate an effective sample size automatically.
- `--mie-order` selects the expansion order of the NN MIE. The default is 2.
- `--no-periodic` forces every coordinate to use an ordinary linear metric. This produces a significant speedup, but the torsion entropy contribution loses some correctness.
- `--theiler-window W` excludes temporally adjacent frames from the nearest-neighbor search: frame i's neighbor must satisfy |i - j| > W, counted in frames after `--start` and `--stride`. The default, 0, is the plain estimator. See [Correlated frames](correlated-frames.md).
- `--exact-constant` uses the exact finite-sample term `psi(N) - psi(1) = H_(N-1)`. The default is the common approximation `ln(N) + EulerGamma`.
- `--tree-bonds` uses the bonds of the BAT spanning tree, including bonds to hydrogen and excluding ring closures, in place of every heavy-atom bond. Use it only for trajectories without constrained bonds to hydrogen; see [Choosing the bond set](coordinates.md#choosing-the-bond-set).
- `--bonds` selects the bond set outright: `heavy` (the default), `tree` (the same as `--tree-bonds`), or `tree-unconstrained`, the spanning tree without the bonds to hydrogen that SHAKE holds fixed. `tree-unconstrained` keeps the same angles and torsions as `tree` and is the one to use with constrained X-H bonds when the Jacobian is wanted. Pass `--bonds` or `--tree-bonds`, not both.
- `--tree` selects the spanning tree the coordinates are built on: `mass` (the default, rooted at the heaviest terminal atom and grown lightest-first, as MDAnalysis does) or `canonical`, rooted at the N-terminal backbone nitrogen and grown along the chain so that the primary torsion about each central bond is the one it is named for (phi, psi, omega, chi1, ...). A molecule with no backbone is rooted at its highest-degree atom. See [Choosing the tree](coordinates.md#choosing-the-tree).
- `--no-phase-angles` uses the raw torsions. By default each torsion that shares its central bond with an earlier torsion is replaced by its offset from that torsion. See [Coordinates](coordinates.md).
- `--jacobian` also prints `Mean ln J = ...`, the mean of the log BAT Jacobian over the frames used, on the line before the entropy. It requires `--tree-bonds` or `--bonds tree-unconstrained`, and cannot be combined with `--torsions-only`.
- With `--estimator qh` or `bqh`, two lines before the total give the terms it is the sum of: `First-order entropy = ...` and `Correlation term = ...`.
- `--estimator` chooses the estimator: `nn` (the default) is the nearest-neighbor MIE, `qh` is quasi-harmonic, `bqh` is Boltzmann quasi-harmonic, and `split` treats soft torsions non-parametrically and the rest with a Gaussian (see [Split estimator](split.md)). See [Quasi-harmonic estimators](quasi-harmonic.md). `--mie-order` applies only to `nn`.
- `--histogram-bins N` makes `bqh` estimate each coordinate's first-order entropy from an N-bin histogram instead of nearest neighbors.
- `--per-coordinate` also prints a tab-separated table of each coordinate's share of the entropy before the total: its number, type (`bond`, `angle`, `torsion` or `phase angle`), atoms (numbered from 1, as in Amber and cpptraj), and entropy. The shares sum to the total. It works with `--estimator nn` at every `--mie-order`; with `qh` and `bqh`, for which the table has two more columns, `first_order` and `correlation`, the two terms each coordinate's entropy is the sum of; and with `split`, for which it has `set` (`soft` or `hard`), `own` and `coupling`. It cannot be combined with `--soft-estimator joint`. See [Per-coordinate entropies](per-coordinate.md).
- `--mutual-information` also prints a tab-separated table of the mutual information of every pair of coordinates, numbered as in `--per-coordinate`. It applies only to `--estimator nn`.
- With `--estimator split`: `--soft topology|detect|LIST` chooses the soft coordinates (default `topology`; LIST counts from 1), `--soft-estimator mie|mist|network|joint` (with `--network-parents N`, default 5, for `network`) and `--hard-estimator qh|bqh|qh-mie1|qh-mie2` the estimators for each set, `--coupling gaussian|states|nn|none` the coupling term (`nn` estimates the hard-soft cross terms of the second-order expansion with the nearest-neighbor estimator; `none` drops them), `--soft-features complex|cossin|harmonics` and `--harmonics N` how soft torsions enter the Gaussian coupling (default: 3 harmonics), and `--state-torsions auto|LIST` and `--min-state-frames N` the states of the per-state coupling. `--mie-order`, `--histogram-bins` and `--periodic-correlation` apply to the matching part. The output lists the soft coordinates, the states, and `Soft entropy`, `Hard entropy` and `Coupling term` before the total. See [Split estimator](split.md).
- `--write-coordinates PREFIX` also writes the coordinates it built to `PREFIX.bat` (the definitions) and `PREFIX.npy` (their values), which `--from-coordinates` reads back. `--coordinate-format tsv` writes the values as text instead. See [Coordinate sets](coordinate-sets.md).
- `--from-coordinates` makes the two positional arguments a definition file and a values file rather than a topology and a trajectory, so the coordinates can come from another program. The flags that build coordinates (`--tree`, `--bonds`, `--tree-bonds`, `--no-phase-angles`, `--torsions-only`, `--write-coordinates`) are refused; the estimator flags, `--start`, `--stop` and `--stride` all work as usual. See [Coordinate sets](coordinate-sets.md).
- `--periodic-correlation complex|centered` sets how torsions enter the quasi-harmonic correlation term; `complex`, the default for `qh` and `bqh`, is Harpole and Sharp's `e^{iφ}`; `centered` is the default for the hard set of `split`. See [Quasi-harmonic estimators](quasi-harmonic.md).
