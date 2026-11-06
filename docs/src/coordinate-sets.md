# Coordinate sets

molentropy normally builds its own internal coordinates from a topology and a
Cartesian trajectory, as [Coordinates](coordinates.md) describes. That ties the
definition of the coordinates to this implementation, which is not always what
you want. A collaborator may already have a coordinate list from another
program; a trajectory may be in a format the reader here does not speak; or the
coordinates may themselves be the object of study, worth freezing in a file
that outlives any one run.

A *coordinate set* is that file pair. It holds everything the estimators need
to know about a set of coordinates and their values, and nothing else — in
particular, not the topology. molentropy can write one out and read one back,
and the round trip changes nothing: the entropy from a set written and read
again is the same number, to the last digit, as the entropy from the
trajectory it came from.

## The two files

`PREFIX.bat` is the definitions: a tab-separated table with a `#` comment
header, one row per coordinate. Coordinates and atoms are numbered from 1, as
in Amber and cpptraj.

```text
# molentropy coordinate set 1
# bonds: tree-unconstrained
coordinate	kind	atoms	reference	ring	symmetric	soft
1	bond	2-5	-	0	0	0
...
30	torsion	5-7-9-15	-	0	0	1
31	phase	11-9-7-5	30	0	0	0
```

| column | meaning |
| --- | --- |
| `coordinate` | the coordinate's number, counting from 1 and in order |
| `kind` | `bond`, `angle`, `torsion` or `phase` |
| `atoms` | the atoms behind it, numbered from 1: two for a bond, three for an angle with the vertex in the middle, four for a torsion about the middle two |
| `reference` | for a phase angle, the torsion it is measured from; `-` otherwise |
| `ring` | 1 when the central bond lies on a ring |
| `symmetric` | 1 when the central bond is a symmetric rotor, such as a methyl |
| `soft` | 1 when the coordinate is in the default soft set of the [split estimator](split.md) |

The `ring`, `symmetric` and `soft` flags describe the *bond*, so every torsion
about a methyl carries `symmetric`, while only the primary one is also `soft`.
That is what makes the table self-contained: without them, working out which
coordinates are soft would need the topology back.

The only header line that is read is `bonds:`, which records whether the set
came from a spanning tree (`tree`, `tree-unconstrained`) or from the redundant
heavy-atom bonds (`heavy`). The BAT Jacobian is defined only for the first two,
so `--jacobian` needs that line to be right. Every other `#` line is
provenance.

`PREFIX.npy` (or `PREFIX.tsv`) is the values: one row per frame, one column per
coordinate, in the order of the table. Bond lengths are in the trajectory's
length unit, angles and torsions in radians, and phase angles already wrapped
to (-pi, pi]. The NumPy form is a plain two-dimensional C-order `float64`
array — `np.load` reads it, `np.save` writes it. The text form is
whitespace-separated numbers, `#` comments allowed.

One thing to know before comparing a column against another program's output:
a torsion here carries the *negative* of the IUPAC sign, so molentropy's value
is `-phi` where cpptraj and PyMOL would print `phi`. Nothing in the entropy
depends on it — a reflection changes no density — but a coordinate set read in
from elsewhere should use the same convention throughout, or a phase angle will
be the sum of two torsions rather than their difference.

## Writing a set

Add `--write-coordinates PREFIX` to any run. The coordinates are built as
usual, written out, and then used for the estimate, so one command gives both
the number and the files behind it:

```bash
molentropy system.parm7 traj.nc --tree canonical --bonds tree-unconstrained \
    --estimator split --jacobian --write-coordinates run0
```

`--coordinate-format tsv` writes the values as text instead of `.npy`. From
Python, `System.write_coordinates(prefix, format='npy')` does the same for the
frames the `System` holds.

## Reading a set

`--from-coordinates` makes the two positional arguments a definition file and a
values file instead of a topology and a trajectory. Every estimator flag works
as usual:

```bash
molentropy run0.bat run0.npy --from-coordinates --estimator split --jacobian
```

The flags that *build* coordinates — `--tree`, `--bonds`, `--tree-bonds`,
`--no-phase-angles`, `--torsions-only`, `--write-coordinates` — are refused,
because the coordinates already exist. `--start`, `--stop` and `--stride`
select frames as they always do.

From Python, `load_coordinates(definitions, values)` returns the same `System`
object `load_system` does, with the same methods and the same per-coordinate
getters.

## Generating a set elsewhere

`python/bat_phase.py` in this repository builds the canonical tree and the
phase angles with MDAnalysis and writes a coordinate set, so the coordinates
can be made outside this crate and still read back into it. It does what
MDAnalysis's `BAT.py` does, but on the [canonical tree](coordinates.md#choosing-the-tree)
and keeping the redundant torsions as phase angles rather than deleting them:

```bash
python bat_phase.py system.prmtop traj.nc --bonds tree-unconstrained --out run0
molentropy run0.bat run0.npy --from-coordinates --estimator split --jacobian
```

Because it takes an MDAnalysis `Universe`, it reads topologies and trajectories
this crate cannot, and `--select` can override the solute selection. It
reproduces molentropy's own coordinates bit for bit, which `--check` verifies
against a definition file written by either side:

```bash
python bat_phase.py system.prmtop --check run0.bat
```

`python/test_bat_phase.py` covers the parts with no file to compare against:
the solute selection, the hydrogen and element rules, the neighbour ordering,
the torsion sign and the ring and rotor searches. Run it with `python
test_bat_phase.py` or under pytest.

Bit-exactness needs a little care, because NumPy's vectorized `arctan2` differs
from the platform's `atan2` by one ulp on roughly one value in fourteen. The
script goes through `math.atan2` by default; `--fast-atan2` takes the
vectorized path, which is worth having on a large protein and moves no angle by
more than an ulp.

Nothing obliges a generator to be this crate's twin, though. Any program that
can write the two files can feed molentropy: the table says which coordinates
are torsions, which are phase angles of which, and which make up the soft set,
and the reader checks that the answers are consistent before using them.
