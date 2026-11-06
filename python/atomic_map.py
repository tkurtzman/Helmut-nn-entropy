#!/usr/bin/env python3
"""Atomic entropy maps from ``molentropy --per-coordinate`` output.

The estimators report an entropy share per internal coordinate. This turns
those shares into a share per *atom*, so that an entropy map can be laid
beside an energy map computed the same way.

Assignment rules
----------------

A coordinate's share is divided among the atoms that define it: a bond in
halves, a valence angle in thirds, a lone torsion in quarters. The only
question is what to do with a group of torsions about one central bond, where
the generator keeps the first as the primary torsion and the rest as phase
angles measured from it.

``pooled`` (the default)
    Treat the group as one object. Which child was designated primary is a
    convention -- the group spans the same subspace either way -- so the
    assignment should not depend on it. The group's entropy goes one quarter
    to each of the two central-bond atoms and the remaining half split equally
    among the peripheral atoms the group involves: the children, plus the
    reference atom of the primary torsion. For a group of one this is exactly
    the quarter rule, so lone torsions are unaffected.

``quarter``
    The per-coordinate rule: a phase angle psi_i is a four-atom coordinate
    {c_i, c_1, p, g} and is divided in quarters among them. Faithful to each
    coordinate taken alone, but it credits the primary's child c_1 in every
    phase angle of the group, which splits atoms that are chemically
    equivalent. Kept for comparison.

Both rules hand out exactly 100% of every coordinate's share, so the per-atom
values sum to the total entropy whichever is used. The choice of rule cannot
change a total, only how it is distributed.

The Jacobian
------------

With ``--values`` the volume-element term is mapped too: a bond contributes
``2 ln b`` split between its two atoms, a valence angle ``ln sin(theta)``
among its three, and torsions contribute nothing. The summed terms reproduce
the ``Mean ln J`` that ``molentropy --jacobian`` reports.

Usage
-----

One state, shares in nats::

    molentropy top.parm7 traj.nc --tree canonical --estimator split \\
        --soft 50,53,58 --soft-estimator mie --mie-order 2 \\
        --hard-estimator qh-mie2 --coupling none \\
        --per-coordinate --jacobian > state0.txt

    atomic_map.py state0.txt -o map0.csv

Two states, ``-T dS`` per atom in kcal/mol (state 0 minus state 1)::

    atomic_map.py state0.txt state1.txt --temperature 300 \\
        --values0 state0.npy --values1 state1.npy \\
        --names top.parm7 -o map.csv

``--values`` files are the ``PREFIX.npy`` (or ``.tsv``) written by
``--write-coordinates``; their column order is the coordinate order of the
table, which is how the two are matched.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

import numpy as np

KB_KCAL = 0.001987204  # kcal/mol/K

BOND, ANGLE, TORSION, PHASE = "bond", "angle", "torsion", "phase angle"


# --------------------------------------------------------------------------
# reading molentropy output
# --------------------------------------------------------------------------


class Coordinate:
    """One row of the ``--per-coordinate`` table."""

    __slots__ = ("index", "kind", "atoms", "share", "block")

    def __init__(self, index, kind, atoms, share, block):
        self.index = index  # 1-based, as printed
        self.kind = kind  # bond | angle | torsion | phase angle
        self.atoms = atoms  # 0-based atom indices
        self.share = share  # entropy share, nats
        self.block = block  # soft | hard | None

    def __repr__(self):  # pragma: no cover - debugging aid
        return "Coordinate(%d, %r, %r, %g)" % (
            self.index,
            self.kind,
            [a + 1 for a in self.atoms],
            self.share,
        )


def read_per_coordinate(path):
    """Parse a ``--per-coordinate`` table out of molentropy's stdout.

    Tolerates the summary lines printed around the table, and the differing
    trailing columns of the nn, qh/bqh and split estimators: the share is
    always the fourth column.
    """
    rows = []
    in_table = False
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("coordinate\ttype"):
                in_table = True
                header = line.split("\t")
                continue
            if not in_table:
                continue
            if "\t" not in line:
                break  # the table ended; summary lines follow
            field = line.split("\t")
            if len(field) < 4:
                break
            block = None
            if "set" in header:
                at = header.index("set")
                if at < len(field):
                    block = field[at]
            rows.append(
                Coordinate(
                    index=int(field[0]),
                    kind=field[1],
                    atoms=[int(a) - 1 for a in field[2].split("-")],
                    share=float(field[3]),
                    block=block,
                )
            )
    if not rows:
        raise SystemExit("no --per-coordinate table found in %s" % path)
    expected = set(range(1, len(rows) + 1))
    if {r.index for r in rows} != expected:
        raise SystemExit("coordinate indices in %s are not 1..n" % path)
    return rows


def read_values(path, ncoord):
    """Coordinate values as ``values[frame][coordinate]``."""
    if path.endswith(".npy"):
        values = np.load(path)
    else:
        values = np.loadtxt(path)
    values = np.atleast_2d(values)
    if values.shape[1] != ncoord:
        raise SystemExit(
            "%s has %d columns but the table has %d coordinates"
            % (path, values.shape[1], ncoord)
        )
    return values


def read_atom_names(path, natoms):
    """(name, residue) per atom from an Amber prmtop; blanks if unavailable."""
    names = [""] * natoms
    residues = [""] * natoms
    try:
        text = open(path).read()
    except OSError:
        return names, residues
    sections = {}
    for block in text.split("%FLAG ")[1:]:
        lines = block.splitlines()
        flag = lines[0].split()[0]
        sections[flag] = [ln for ln in lines[1:] if not ln.startswith("%")]

    def fixed(flag, width):
        out = []
        for line in sections.get(flag, []):
            for start in range(0, len(line), width):
                token = line[start : start + width].strip()
                if token:
                    out.append(token)
        return out

    got = fixed("ATOM_NAME", 4)
    for i in range(min(natoms, len(got))):
        names[i] = got[i]
    labels = fixed("RESIDUE_LABEL", 4)
    pointers = [int(x) for x in fixed("RESIDUE_POINTER", 8)]
    for r, start in enumerate(pointers):
        end = pointers[r + 1] - 1 if r + 1 < len(pointers) else natoms
        for a in range(start - 1, min(end, natoms)):
            if r < len(labels):
                residues[a] = labels[r]
    return names, residues


# --------------------------------------------------------------------------
# assignment rules
# --------------------------------------------------------------------------


def central_bond(coord):
    """The unordered central bond of a torsion or phase angle."""
    return frozenset((coord.atoms[1], coord.atoms[2]))


def torsion_groups(coords):
    """Map each central bond to the indices of the coordinates about it."""
    groups = {}
    for i, c in enumerate(coords):
        if c.kind in (TORSION, PHASE):
            groups.setdefault(central_bond(c), []).append(i)
    return groups


def weights_pooled(coords):
    """Per-coordinate {atom: weight} under the pooled rule."""
    weights = [None] * len(coords)
    for bond, members in torsion_groups(coords).items():
        if len(bond) != 2:
            raise SystemExit(
                "coordinate %d has a degenerate central bond" % (members[0] + 1)
            )
        p, g = sorted(bond)
        peripheral = {coords[i].atoms[0] for i in members}
        for i in members:
            if coords[i].kind == TORSION and len(coords[i].atoms) == 4:
                peripheral.add(coords[i].atoms[3])
        shared = {p: 0.25, g: 0.25}
        for atom in peripheral:
            shared[atom] = shared.get(atom, 0.0) + 0.5 / len(peripheral)
        for i in members:
            weights[i] = dict(shared)
    for i, c in enumerate(coords):
        if weights[i] is None:
            weights[i] = _even(c.atoms)
    return weights


def weights_quarter(coords):
    """Per-coordinate {atom: weight} under the per-coordinate quarter rule."""
    primary = {}
    for bond, members in torsion_groups(coords).items():
        for i in members:
            if coords[i].kind == TORSION:
                primary[bond] = coords[i].atoms[0]
                break
    weights = []
    for c in coords:
        if c.kind == PHASE:
            c1 = primary.get(central_bond(c))
            if c1 is None:
                raise SystemExit(
                    "phase angle %d has no primary torsion about its central bond"
                    % c.index
                )
            weights.append(_even([c.atoms[0], c1, c.atoms[1], c.atoms[2]]))
        else:
            weights.append(_even(c.atoms))
    return weights


def _even(atoms):
    out = {}
    for a in atoms:
        out[a] = out.get(a, 0.0) + 1.0 / len(atoms)
    return out


RULES = {"pooled": weights_pooled, "quarter": weights_quarter}


# --------------------------------------------------------------------------
# mapping
# --------------------------------------------------------------------------


def map_shares(coords, natoms, rule="pooled"):
    """Entropy share per atom, in nats."""
    out = np.zeros(natoms)
    for c, w in zip(coords, RULES[rule](coords)):
        for atom, weight in w.items():
            out[atom] += c.share * weight
    return out


def map_jacobian(coords, values, natoms):
    """(<ln J> per atom, total <ln J>). Bonds give 2 ln b, angles ln sin t."""
    out = np.zeros(natoms)
    total = 0.0
    for c in coords:
        column = values[:, c.index - 1]
        if c.kind == BOND:
            term = float(np.mean(2.0 * np.log(column)))
        elif c.kind == ANGLE:
            term = float(np.mean(np.log(np.sin(column))))
        else:
            continue
        total += term
        for atom in c.atoms:
            out[atom] += term / len(c.atoms)
    return out, total


def natoms_of(*coordinate_sets):
    highest = 0
    for coords in coordinate_sets:
        for c in coords:
            highest = max(highest, max(c.atoms))
    return highest + 1


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Map molentropy per-coordinate entropies onto atoms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("state0", help="--per-coordinate output for the first state")
    ap.add_argument(
        "state1",
        nargs="?",
        help="second state; with it the map is -T dS = RT (state0 - state1)",
    )
    ap.add_argument("--values0", help="coordinate values for state0 (.npy or .tsv)")
    ap.add_argument("--values1", help="coordinate values for state1")
    ap.add_argument(
        "--rule", choices=sorted(RULES), default="pooled", help="assignment rule"
    )
    ap.add_argument(
        "--temperature", type=float, help="K; required with two states, for RT"
    )
    ap.add_argument("--natoms", type=int, help="override the inferred atom count")
    ap.add_argument("--names", help="Amber prmtop, for atom and residue labels")
    ap.add_argument("-o", "--output", help="write CSV here instead of stdout")
    args = ap.parse_args(argv)

    s0 = read_per_coordinate(args.state0)
    s1 = read_per_coordinate(args.state1) if args.state1 else None
    if s1 is not None:
        if args.temperature is None:
            ap.error("--temperature is required when two states are given")
        if len(s0) != len(s1):
            ap.error(
                "the two states have %d and %d coordinates; they must be described "
                "in the same coordinate set" % (len(s0), len(s1))
            )
        for a, b in zip(s0, s1):
            if a.kind != b.kind or a.atoms != b.atoms:
                ap.error(
                    "coordinate %d differs between the two states (%s %s vs %s %s); "
                    "both states must use the same tree and bond set"
                    % (
                        a.index,
                        a.kind,
                        "-".join(str(x + 1) for x in a.atoms),
                        b.kind,
                        "-".join(str(x + 1) for x in b.atoms),
                    )
                )

    natoms = args.natoms or natoms_of(*[c for c in (s0, s1) if c is not None])
    names, residues = (
        read_atom_names(args.names, natoms) if args.names else ([""] * natoms,) * 2
    )

    share0 = map_shares(s0, natoms, args.rule)
    jac0 = np.zeros(natoms)
    if args.values0:
        jac0, _ = map_jacobian(s0, read_values(args.values0, len(s0)), natoms)

    if s1 is None:
        columns = ["atom_index_1based", "atom_name", "residue", "S_nats"]
        if args.values0:
            columns.append("lnJ_nats")
        table = [share0] + ([jac0] if args.values0 else [])
    else:
        share1 = map_shares(s1, natoms, args.rule)
        jac1 = np.zeros(natoms)
        if args.values1:
            jac1, _ = map_jacobian(s1, read_values(args.values1, len(s1)), natoms)
        rt = KB_KCAL * args.temperature
        value = rt * ((share0 + jac0) - (share1 + jac1))
        columns = [
            "atom_index_1based",
            "atom_name",
            "residue",
            "mTdS_kcal_per_mol",
            "S_state0_nats",
            "S_state1_nats",
            "lnJ_state0_nats",
            "lnJ_state1_nats",
        ]
        table = [value, share0, share1, jac0, jac1]

    handle = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for a in range(natoms):
            writer.writerow(
                [a + 1, names[a], residues[a]] + ["%.10g" % col[a] for col in table]
            )
    finally:
        if args.output:
            handle.close()

    if args.output:
        total = table[0].sum()
        unit = "kcal/mol" if s1 is not None else "nats"
        print(
            "%s rule, %d atoms, total %.6f %s -> %s"
            % (args.rule, natoms, total, unit, args.output),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
