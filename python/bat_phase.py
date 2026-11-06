#!/usr/bin/env python3
"""Bond-angle-torsion internal coordinates on the canonical spanning tree,
with the redundant torsions kept as phase angles.

This is the generator side of the coordinate set that ``molentropy`` reads:
it does what MDAnalysis's ``BAT.py`` does, but builds the tree by the
canonical rule rather than by mass order, and keeps every torsion about an
already-used central bond as its wrapped offset from the first one instead of
deleting it.

It writes two files:

``PREFIX.bat``
    The coordinate definitions: one row per coordinate with its atoms, its
    kind, the primary torsion a phase angle is measured from, and flags for
    ring membership, symmetric rotors and the default soft set. The file is
    self-contained, so ``molentropy`` needs no topology to read it back.

``PREFIX.npy`` or ``PREFIX.tsv``
    The coordinate values, one row per frame and one column per coordinate.
    Bond lengths in angstroms, angles and torsions in radians.

The tree, the coordinate order and the arithmetic match
``molentropy``'s own generator, so a set written here and read back gives the
same entropy as running ``molentropy`` on the topology and trajectory
directly. ``--check`` compares the two coordinate lists directly.

Usage
-----

    python bat_phase.py system.prmtop traj.nc --out run0
    python bat_phase.py system.prmtop traj1.nc traj2.nc --bonds tree-unconstrained
    python bat_phase.py system.prmtop --definitions-only --out run0

Requires MDAnalysis and NumPy.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

#: Amber atom types that belong only to water models, including the extra
#: point of four-point models such as TIP4P-Ew and OPC.
WATER_ATOM_TYPES = frozenset({"OW", "HW", "EP", "EPW"})

#: Residue names used for water by Amber, CHARMM and GROMACS topologies.
WATER_RESIDUES = frozenset(
    {
        "WAT", "HOH", "H2O", "SOL", "TIP3", "TP3", "T3P", "TIP4", "TP4",
        "T4P", "T4E", "TIP5", "TP5", "SPC", "SPCE", "OPC",
    }
)

SYMBOL_TO_Z = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15,
    "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20, "SC": 21, "TI": 22,
    "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29,
    "ZN": 30, "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36,
    "RB": 37, "SR": 38, "Y": 39, "ZR": 40, "MO": 42, "AG": 47, "CD": 48,
    "IN": 49, "SN": 50, "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55,
    "BA": 56, "PT": 78, "AU": 79, "HG": 80, "TL": 81, "PB": 82, "BI": 83,
}


class Topology:
    """The per-atom facts the tree rules need, read once from a Universe."""

    def __init__(self, universe, select=None):
        atoms = universe.atoms
        self.natom = atoms.n_atoms
        self.names = [str(n).strip() for n in atoms.names]
        self.types = [str(t).strip() for t in _attribute(atoms, "types", self.names)]
        self.masses = np.asarray(atoms.masses, dtype=np.float64)
        self.resnames = [
            str(r).strip() for r in _attribute(atoms, "resnames", [""] * self.natom)
        ]
        self.declared_z = _declared_atomic_numbers(atoms)

        self.adjacency = [[] for _ in range(self.natom)]
        for a, b in _bond_indices(universe):
            if a >= self.natom or b >= self.natom:
                raise ValueError(f"bond ({a}, {b}) references an atom beyond NATOM")
            self.adjacency[a].append(b)
            self.adjacency[b].append(a)

        if select is None:
            self.solute = [not self._is_excluded(i) for i in range(self.natom)]
        else:
            chosen = set(universe.select_atoms(select).indices.tolist())
            self.solute = [i in chosen for i in range(self.natom)]

        # Hydrogens are identified by atom type, as the simulation's constraint
        # algorithm does, not by element: a type is what SHAKE keys off.
        self.hydrogens = {
            i
            for i, t in enumerate(self.types)
            if t[:1] in ("H", "h") and t != "HW"
        }

    # -- element helpers ---------------------------------------------------

    def element(self, atom):
        """Atomic number, from the topology when it declares one and from the
        mass otherwise. Mirrors the Rust ``Naming::element``."""
        z = self.declared_z[atom]
        if z > 0:
            return z
        return max(1, _round_half_away(self.masses[atom] / 1.9))

    def is_hydrogen(self, atom):
        return self.element(atom) == 1

    def numeral(self, atom):
        """The trailing digits of a PDB atom name (``CG1`` -> 1, ``CG`` -> 0),
        which order the branches of a symmetric centre the way IUPAC does."""
        digits = ""
        for char in reversed(self.names[atom]):
            if not char.isdigit():
                break
            digits = char + digits
        return int(digits) if digits else 0

    def _is_excluded(self, index):
        if self.types[index] in WATER_ATOM_TYPES:
            return True
        if self.resnames[index].upper() in WATER_RESIDUES:
            return True
        if self.masses[index] <= 0.0:
            return True
        return self.declared_z[index] == 0 and self.has_declared_z

    @property
    def has_declared_z(self):
        return bool(np.any(self.declared_z > 0))


def _attribute(atoms, name, fallback):
    try:
        return list(getattr(atoms, name))
    except Exception:
        return list(fallback)


def _declared_atomic_numbers(atoms):
    """Atomic numbers as the topology declares them, 0 where it does not.

    An Amber prmtop's ``ATOMIC_NUMBER`` section reaches us through
    MDAnalysis's ``elements``; formats that carry no element leave this all
    zero and the mass rule takes over, exactly as in the Rust reader.
    """
    try:
        symbols = list(atoms.elements)
    except Exception:
        return np.zeros(atoms.n_atoms, dtype=np.int64)
    return np.array(
        [SYMBOL_TO_Z.get(str(s).strip().upper(), 0) for s in symbols],
        dtype=np.int64,
    )


def _bond_indices(universe):
    try:
        return [tuple(int(x) for x in bond) for bond in universe.bonds.indices]
    except Exception as err:  # pragma: no cover - depends on the input file
        raise SystemExit(
            "the topology carries no bond information, which the BAT tree "
            f"needs: {err}"
        )


def _round_half_away(x):
    """Rust's ``f64::round``: halfway cases go away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


# ---------------------------------------------------------------------------
# The canonical spanning tree
# ---------------------------------------------------------------------------


def backbone_successor(name):
    """The backbone atom that continues the chain after ``name``, so that
    walking N -> CA -> C -> N makes the primary torsion about each backbone
    bond the one that torsion is named for."""
    return {"N": "CA", "CA": "C", "C": "N"}.get(name)


class CanonicalTree:
    """The canonical tree of one molecule: rooted at the N-terminal backbone
    nitrogen, grown along the chain, with the chemically named torsion primary
    about every central bond.

    ``root`` is the three-atom root triple; ``torsions`` is one
    ``(atom, parent, grandparent, reference)`` entry per remaining atom,
    grouped by central bond with the named torsion first in each group.
    """

    def __init__(self, fragment, topology):
        if len(fragment) < 3:
            raise ValueError("fragment must have at least 3 atoms")
        self.top = topology
        self.in_fragment = set(fragment)
        self.fragment = fragment

        root = self._root()
        parent, children, visit_order = self._grow(root)
        if len(visit_order) != len(fragment):
            raise ValueError("fragment not connected")

        bat_root, first = self._root_triple(root, parent, children)
        placement = list(bat_root) + [a for a in visit_order if a not in bat_root]
        placed_at = {a: i for i, a in enumerate(placement)}
        subtree = self._subtree_sizes(parent, visit_order)

        self.root = bat_root
        self.torsions = self._torsions(
            placement, placed_at, parent, first, subtree, len(bat_root)
        )
        self.angles = [t[:3] for t in self.torsions]

    # -- construction ------------------------------------------------------

    def neighbors(self, a):
        return sorted({n for n in self.top.adjacency[a] if n in self.in_fragment})

    def ranked(self, a):
        return sorted(self.neighbors(a), key=lambda n: self.priority(a, n))

    def priority(self, parent, child):
        """How good a continuation ``child`` is from ``parent``, smallest
        first: the backbone successor, then any other backbone atom, then the
        heavy atoms by decreasing element and increasing IUPAC numeral, then
        the hydrogens."""
        top = self.top
        if top.is_hydrogen(child):
            cls = 3
        elif backbone_successor(top.names[parent]) == top.names[child]:
            cls = 0
        elif top.names[child] in ("N", "CA", "C"):
            cls = 1
        else:
            cls = 2
        return (cls, -top.element(child), top.numeral(child), child)

    def _root(self):
        """The N-terminal backbone nitrogen: an N bonded to a CA that is
        bonded to a C. The lowest index among them is the first residue, since
        Amber writes residues in order. A molecule with no backbone is rooted
        at the highest-degree atom, lowest index breaking ties."""
        top = self.top
        candidates = [
            a
            for a in self.fragment
            if top.names[a] == "N"
            and any(
                top.names[ca] == "CA"
                and any(top.names[c] == "C" for c in self.neighbors(ca))
                for ca in self.neighbors(a)
            )
        ]
        if candidates:
            return min(candidates)
        return max(self.fragment, key=lambda a: (len(self.neighbors(a)), -a))

    def _grow(self, root):
        """Breadth-first, each atom's neighbours in canonical order so that
        ring closures are broken the same way every time."""
        parent = {}
        children = {}
        visit_order = [root]
        seen = {root}
        queue = deque([root])
        while queue:
            a = queue.popleft()
            for child in self.ranked(a):
                if child in seen:
                    continue
                seen.add(child)
                parent[child] = a
                children.setdefault(a, []).append(child)
                visit_order.append(child)
                queue.append(child)
        return parent, children, visit_order

    def _root_triple(self, root, parent, children):
        """The root triple is the path root -> first -> second, the start of
        the chain; or second -> root -> first when the third atom hangs off
        the root instead. Both are paths of bonded atoms."""
        root_children = list(children.get(root, []))
        first = next((c for c in root_children if c in children), None)
        if first is None:
            if not root_children:
                raise ValueError("fragment has no second atom")
            first = root_children[0]
        second = next(iter(children.get(first, [])), None)
        if second is None:
            second = next((c for c in root_children if c != first), None)
        if second is None:
            raise ValueError("fragment has no third atom for the BAT root")
        if parent.get(second) == first:
            return (root, first, second), first
        return (second, root, first), first

    @staticmethod
    def _subtree_sizes(parent, visit_order):
        """Subtree sizes break ties in favour of the branch that carries the
        rest of the molecule, when the names say nothing."""
        subtree = {a: 1 for a in visit_order}
        for a in reversed(visit_order):
            p = parent.get(a)
            if p is not None:
                subtree[p] += subtree[a]
        return subtree

    def _torsions(self, placement, placed_at, parent, first, subtree, root_len):
        """Every atom outside the root triple contributes one torsion
        ``(child, parent, grandparent, reference)``, grouped by the central
        bond so that the named torsion of each group becomes the primary and
        the rest become phase angles measured from it."""
        grouped = []
        slot_of = {}
        for a in placement[root_len:]:
            p = parent[a]
            # The root's other children have no grandparent of their own, so
            # they use `first`, which the root triple has already placed.
            g = parent.get(p, first)
            reference = self._reference(a, p, g, placement, placed_at, parent)
            key = (min(p, g), max(p, g))
            slot = slot_of.get(key)
            if slot is None:
                slot = len(grouped)
                slot_of[key] = slot
                grouped.append([])
            grouped[slot].append((a, p, g, reference))

        torsions = []
        for group in grouped:
            group.sort(key=lambda t: self._group_key(t, subtree))
            torsions.extend(group)
        return torsions

    def _reference(self, a, p, g, placement, placed_at, parent):
        """The fourth atom of the torsion: the grandparent's own parent when
        it is usable, otherwise the best-ranked neighbour of the grandparent
        already placed, otherwise any atom placed earlier."""

        def usable(h):
            return h is not None and h != p and h != a and placed_at[h] < placed_at[a]

        great = parent.get(g)
        if usable(great):
            return great
        for h in self.ranked(g):
            if usable(h):
                return h
        for h in placement[: placed_at[a]]:
            if h != p and h != g and h != a:
                return h
        raise ValueError("no reference atom for a BAT torsion")

    def _group_key(self, torsion, subtree):
        a, p = torsion[0], torsion[1]
        cls, negated_element, numeral, atom = self.priority(p, a)
        return (cls, -subtree[a], negated_element, numeral, atom)


# ---------------------------------------------------------------------------
# The coordinate list
# ---------------------------------------------------------------------------

BOND, ANGLE, TORSION, PHASE = "bond", "angle", "torsion", "phase"


def build_coordinates(top, bonds="heavy", tree="canonical", phase_angles=True):
    """The coordinate list of the whole solute, in the order the estimators
    see it: per molecule, the bonds, then the angles, then the torsions.

    Returns a list of ``(kind, atoms)`` with topology atom indices, and the
    list of solute atoms.
    """
    entries = []
    atoms = []
    visited = [False] * top.natom
    for seed in range(top.natom):
        if not top.solute[seed] or visited[seed]:
            continue
        fragment = []
        queue = deque([seed])
        visited[seed] = True
        while queue:
            a = queue.popleft()
            fragment.append(a)
            for b in top.adjacency[a]:
                if top.solute[b] and not visited[b]:
                    visited[b] = True
                    queue.append(b)

        if len(fragment) == 1:
            # A lone atom (a monatomic ion, say) has no internal coordinates.
            continue
        if len(fragment) == 2:
            a, b = min(fragment), max(fragment)
            if bonds == "tree" or not (a in top.hydrogens or b in top.hydrogens):
                entries.append([a, b])
            atoms.extend(fragment)
            continue
        entries.extend(_molecule_entries(fragment, top, bonds, tree))
        atoms.extend(fragment)

    atoms.sort()
    return entries, atoms


def _molecule_entries(fragment, top, bonds, tree):
    if tree == "canonical":
        bat = CanonicalTree(fragment, top)
    else:
        raise SystemExit(
            "only the canonical tree is implemented here; use molentropy "
            "itself for --tree mass"
        )

    entries = []
    if bonds in ("tree", "tree-unconstrained"):
        # The two root bonds, then the bond that places each later atom, in
        # the same order as the angles and torsions below.
        def keep(a, b):
            return bonds == "tree" or not (
                a in top.hydrogens or b in top.hydrogens
            )

        if keep(bat.root[0], bat.root[1]):
            entries.append([bat.root[0], bat.root[1]])
        if keep(bat.root[1], bat.root[2]):
            entries.append([bat.root[1], bat.root[2]])
        for t in bat.torsions:
            if keep(t[0], t[1]):
                entries.append([t[0], t[1]])
    else:
        fragment_set = set(fragment)
        heavy = []
        for a in fragment:
            for b in top.adjacency[a]:
                if (
                    a < b
                    and b in fragment_set
                    and a not in top.hydrogens
                    and b not in top.hydrogens
                ):
                    heavy.append([a, b])
        heavy.sort()
        entries.extend(heavy)

    entries.append(list(bat.root))
    entries.extend(list(angle) for angle in bat.angles)
    entries.extend(list(torsion) for torsion in bat.torsions)
    return entries


def phase_references(entries, phase_angles=True):
    """For each coordinate, the index of the torsion it is a phase angle of,
    or ``None``.

    The first torsion about each central bond, in coordinate order, keeps its
    value; every later torsion about the same bond becomes its offset from
    that first one, wrapped to (-pi, pi]. The map is a shear on the torus with
    unit Jacobian, so the joint entropy of the full set is unchanged.
    """
    references = [None] * len(entries)
    if not phase_angles:
        return references
    first_about = {}
    for index, entry in enumerate(entries):
        if len(entry) != 4:
            continue
        bond = (min(entry[1], entry[2]), max(entry[1], entry[2]))
        if bond in first_about:
            references[index] = first_about[bond]
        else:
            first_about[bond] = index
    return references


def bond_topology(top):
    """Ring bonds and symmetric-rotor bonds of the solute, as sorted pairs of
    topology indices. A bond lies on a ring exactly when it is not a bridge of
    the bond graph."""
    n = top.natom
    include = top.solute

    def neighbors(a):
        return [b for b in top.adjacency[a] if include[b]]

    # Iterative Tarjan bridge search.
    discovered = [-1] * n
    low = [0] * n
    timer = 0
    bridges = set()
    for root in range(n):
        if not include[root] or discovered[root] != -1:
            continue
        discovered[root] = timer
        low[root] = timer
        timer += 1
        stack = [[root, -1, 0]]
        while stack:
            atom, par, nxt = stack[-1]
            if nxt < len(top.adjacency[atom]):
                neighbor = top.adjacency[atom][nxt]
                stack[-1][2] += 1
                if not include[neighbor] or neighbor == par:
                    continue
                if discovered[neighbor] == -1:
                    discovered[neighbor] = timer
                    low[neighbor] = timer
                    timer += 1
                    stack.append([neighbor, atom, 0])
                else:
                    low[atom] = min(low[atom], discovered[neighbor])
            else:
                stack.pop()
                if stack:
                    up = stack[-1][0]
                    low[up] = min(low[up], low[atom])
                    if low[atom] > discovered[up]:
                        bridges.add((min(up, atom), max(up, atom)))

    ring_bonds = set()
    symmetric_rotor_bonds = set()
    for a in range(n):
        if not include[a]:
            continue
        for b in neighbors(a):
            if b <= a:
                continue
            if (a, b) not in bridges:
                ring_bonds.add((a, b))
                continue

            def symmetric_end(end, other):
                subs = [x for x in neighbors(end) if x != other]
                return (
                    len(subs) >= 2
                    and all(len(neighbors(x)) == 1 for x in subs)
                    and all(top.element(x) == top.element(subs[0]) for x in subs)
                )

            if symmetric_end(a, b) or symmetric_end(b, a):
                symmetric_rotor_bonds.add((a, b))
    return ring_bonds, symmetric_rotor_bonds


def classify(entries, references, ring_bonds, symmetric_bonds):
    """The kind, ring flag, symmetric-rotor flag and soft flag of each
    coordinate.

    The soft set is the first torsion about every non-ring central bond, which
    is what the split estimator takes by default. It is defined by the tree
    rather than by the phase angles, so it is the same set whether or not the
    redundant torsions were turned into offsets.
    """
    primary = set()
    seen = set()
    for index, entry in enumerate(entries):
        if len(entry) != 4:
            continue
        bond = (min(entry[1], entry[2]), max(entry[1], entry[2]))
        if bond not in seen:
            seen.add(bond)
            primary.add(index)

    kinds, rings, symmetric, soft = [], [], [], []
    for index, (entry, reference) in enumerate(zip(entries, references)):
        if len(entry) == 2:
            kinds.append(BOND)
        elif len(entry) == 3:
            kinds.append(ANGLE)
        elif reference is not None:
            kinds.append(PHASE)
        else:
            kinds.append(TORSION)
        if len(entry) == 4:
            bond = (min(entry[1], entry[2]), max(entry[1], entry[2]))
            on_ring = bond in ring_bonds
            rings.append(on_ring)
            symmetric.append(bond in symmetric_bonds)
            soft.append(index in primary and not on_ring)
        else:
            rings.append(False)
            symmetric.append(False)
            soft.append(False)
    return kinds, rings, symmetric, soft


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _atan2(y, x, exact=True):
    """``atan2`` element by element.

    NumPy's vectorized ``arctan2`` is not the platform's ``atan2``: its SIMD
    loop differs from libm by one ulp on roughly one value in fourteen, which
    is harmless physically but would stop a coordinate set written here from
    reproducing ``molentropy``'s own numbers exactly. Going through
    ``math.atan2`` keeps them bit for bit identical; ``exact=False`` takes the
    vectorized path instead, which is worth having on a large protein.
    """
    if not exact:
        return np.arctan2(y, x)
    values = np.fromiter(
        map(math.atan2, y.ravel().tolist(), x.ravel().tolist()),
        dtype=np.float64,
        count=y.size,
    )
    return values.reshape(y.shape)


def _dot(a, b):
    """``a . b`` summed left to right, the order the Rust reader uses.

    NumPy's ``einsum`` and ``dot`` are free to reassociate the sum, which
    moves the last bit of an angle; spelling the three products out keeps the
    two implementations in step to the last ulp.
    """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _cross(a, b):
    """``a x b``, component by component."""
    return np.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


def bond_lengths(positions, index):
    """|a1 - a2| with the difference taken in the trajectory's own precision,
    as the Rust reader does for 32-bit NetCDF coordinates."""
    delta = positions[:, index[:, 0], :] - positions[:, index[:, 1], :]
    delta = delta.astype(np.float64)
    return np.sqrt(_dot(delta, delta))


def bond_angles(positions, index, exact=True):
    """atan2(|v1 x v2|, v1 . v2): accurate over the whole range and never NaN
    for a straight angle. Promoted to f64 first, as the Rust reader does."""
    p = positions.astype(np.float64)
    v1 = p[:, index[:, 0], :] - p[:, index[:, 1], :]
    v2 = p[:, index[:, 2], :] - p[:, index[:, 1], :]
    cross = _cross(v1, v2)
    angle = _atan2(np.sqrt(_dot(cross, cross)), _dot(v1, v2), exact)
    degenerate = (_dot(v1, v1) == 0.0) | (_dot(v2, v2) == 0.0)
    return np.where(degenerate, np.nan, angle)


def torsion_angles(positions, index, exact=True):
    """The torsion a1-a2-a3-a4 in (-pi, pi].

    The sign is the negative of the IUPAC dihedral convention, which is what
    the Rust reader produces; the differences and cross products are taken in
    the trajectory's own precision and only the final combination in f64, so
    the values agree bit for bit.
    """
    b1 = positions[:, index[:, 0], :] - positions[:, index[:, 1], :]
    b2 = positions[:, index[:, 1], :] - positions[:, index[:, 2], :]
    b3 = positions[:, index[:, 2], :] - positions[:, index[:, 3], :]
    c1 = _cross(b2, b3)
    c2 = _cross(b1, b2)
    p1 = _dot(b1, c1).astype(np.float64)
    b2_sum = _dot(b2, b2).astype(np.float64)
    p2 = _dot(c1, c2).astype(np.float64)
    return _atan2(p1 * np.sqrt(b2_sum), p2, exact)


def wrap_angle(angle):
    """``angle`` wrapped to (-pi, pi], the range of the torsions themselves."""
    return np.pi - np.mod(np.pi - angle, 2.0 * np.pi)


def read_positions(universe, trajectory, atoms, start=0, stop=None, stride=1):
    """The selected frames' positions, as ``positions[frame][atom][3]``.

    MDAnalysis reads the trajectory when it can. Its AMBER NetCDF reader
    insists on the 64-bit-offset form of the format, which Amber itself does
    not always write, so a classic NetCDF file falls back to SciPy; the values
    are the same 32-bit numbers either way.
    """
    gather = np.asarray(atoms, dtype=np.intp)
    try:
        universe.load_new(list(trajectory))
    except Exception as err:
        positions = _netcdf_positions(trajectory, universe.atoms.n_atoms, err)
        return positions[start:stop:stride][:, gather, :]

    frames = [
        universe.atoms.positions[gather]
        for _ in universe.trajectory[start:stop:stride]
    ]
    if not frames:
        raise SystemExit("no frames were read from the trajectory")
    return np.stack(frames)


def _netcdf_positions(trajectory, natom, reason):
    """Every frame of one or more Amber NetCDF files, read with SciPy."""
    try:
        from scipy.io import netcdf_file
    except ImportError:  # pragma: no cover - depends on the environment
        raise SystemExit(f"cannot read the trajectory ({reason}), and SciPy is not installed")
    blocks = []
    for path in trajectory:
        with netcdf_file(str(path), "r", mmap=False) as handle:
            if "coordinates" not in handle.variables:
                raise SystemExit(f"{path} has no coordinates variable ({reason})")
            block = np.asarray(handle.variables["coordinates"][:])
            if block.shape[1] != natom:
                raise SystemExit(
                    f"{path} has {block.shape[1]} atoms, the topology has {natom}"
                )
            blocks.append(block.astype(np.float32, copy=False))
    if not blocks:
        raise SystemExit("no frames were read from the trajectory")
    return np.concatenate(blocks) if len(blocks) > 1 else blocks[0]


def evaluate(positions, entries, atoms, references, exact=True):
    """Every coordinate on every frame, as ``values[frame][coord]``."""
    position_of = {atom: i for i, atom in enumerate(atoms)}
    local = [[position_of[a] for a in entry] for entry in entries]
    groups = {
        2: np.array(
            [e for e in local if len(e) == 2], dtype=np.intp
        ).reshape(-1, 2),
        3: np.array(
            [e for e in local if len(e) == 3], dtype=np.intp
        ).reshape(-1, 3),
        4: np.array(
            [e for e in local if len(e) == 4], dtype=np.intp
        ).reshape(-1, 4),
    }
    slots = {size: [i for i, e in enumerate(local) if len(e) == size] for size in (2, 3, 4)}

    values = np.empty((positions.shape[0], len(entries)), dtype=np.float64)
    if len(groups[2]):
        values[:, slots[2]] = bond_lengths(positions, groups[2])
    if len(groups[3]):
        values[:, slots[3]] = bond_angles(positions, groups[3], exact)
    if len(groups[4]):
        values[:, slots[4]] = torsion_angles(positions, groups[4], exact)

    for index, reference in enumerate(references):
        if reference is not None:
            values[:, index] = wrap_angle(values[:, index] - values[:, reference])
    return values


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

DEFINITION_HEADER = "coordinate\tkind\tatoms\treference\tring\tsymmetric\tsoft"


def write_definitions(path, entries, kinds, references, rings, symmetric, soft, meta):
    """The coordinate definitions, one row per coordinate.

    Coordinates and atoms are numbered from 1, as in Amber and cpptraj.
    """
    with open(path, "w", encoding="utf-8") as out:
        out.write("# molentropy coordinate set 1\n")
        for key, value in meta.items():
            out.write(f"# {key}: {value}\n")
        out.write(DEFINITION_HEADER + "\n")
        for index, entry in enumerate(entries):
            atoms = "-".join(str(a + 1) for a in entry)
            reference = "-" if references[index] is None else references[index] + 1
            out.write(
                f"{index + 1}\t{kinds[index]}\t{atoms}\t{reference}"
                f"\t{int(rings[index])}\t{int(symmetric[index])}"
                f"\t{int(soft[index])}\n"
            )


def write_values(path, values):
    """The coordinate values, as ``.npy`` or as tab-separated text."""
    if path.suffix == ".npy":
        np.save(path, np.ascontiguousarray(values, dtype=np.float64))
        return
    with open(path, "w", encoding="utf-8") as out:
        for row in values:
            out.write("\t".join(f"{x:.17g}" for x in row) + "\n")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("topology", help="topology file (prmtop, psf, tpr, ...)")
    parser.add_argument("trajectory", nargs="*", help="trajectory file(s)")
    parser.add_argument(
        "--out",
        default="coordinates",
        help="output prefix; writes PREFIX.bat and PREFIX.npy (default: coordinates)",
    )
    parser.add_argument(
        "--bonds",
        choices=("heavy", "tree", "tree-unconstrained"),
        default="heavy",
        help="which bonds are coordinates (default: heavy)",
    )
    parser.add_argument(
        "--no-phase-angles",
        action="store_true",
        help="keep the raw torsions instead of the phase angles",
    )
    parser.add_argument(
        "--format",
        choices=("npy", "tsv"),
        default="npy",
        help="value file format (default: npy)",
    )
    parser.add_argument("--start", type=int, default=0, help="first frame (from 0)")
    parser.add_argument("--stop", type=int, default=None, help="stop before this frame")
    parser.add_argument("--stride", type=int, default=1, help="frame stride")
    parser.add_argument(
        "--select",
        default=None,
        help="MDAnalysis selection for the solute; the default drops water, "
        "virtual sites and massless atoms",
    )
    parser.add_argument(
        "--fast-atan2",
        action="store_true",
        help="use NumPy's vectorized arctan2, which is up to one ulp from "
        "libm on about one angle in fourteen; faster on a large system, but "
        "the values no longer match molentropy's own bit for bit",
    )
    parser.add_argument(
        "--definitions-only",
        action="store_true",
        help="write the coordinate list without reading a trajectory",
    )
    parser.add_argument(
        "--check",
        metavar="BAT",
        default=None,
        help="compare the generated coordinate list against an existing "
        "definition file and report the first difference",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        import MDAnalysis as mda
    except ImportError:  # pragma: no cover - depends on the environment
        raise SystemExit("this script needs MDAnalysis: pip install MDAnalysis")

    try:
        universe = mda.Universe(args.topology)
    except Exception as err:
        raise SystemExit(f"MDAnalysis cannot read {args.topology}: {err}")
    top = Topology(universe, args.select)
    entries, atoms = build_coordinates(top, bonds=args.bonds)
    references = phase_references(entries, phase_angles=not args.no_phase_angles)
    ring_bonds, symmetric_bonds = bond_topology(top)
    kinds, rings, symmetric, soft = classify(
        entries, references, ring_bonds, symmetric_bonds
    )

    if args.check:
        return compare(args.check, entries, kinds, references)

    prefix = Path(args.out)
    definitions = prefix.with_suffix(".bat")
    values = None
    if not args.definitions_only:
        if not args.trajectory:
            raise SystemExit("a trajectory is needed unless --definitions-only")
        positions = read_positions(
            universe, args.trajectory, atoms, args.start, args.stop, args.stride
        )
        values = evaluate(positions, entries, atoms, references, not args.fast_atan2)

    meta = {
        "topology": Path(args.topology).name,
        "atoms": len(atoms),
        "coordinates": len(entries),
        "bonds": args.bonds,
        "tree": "canonical",
        "phase_angles": str(not args.no_phase_angles).lower(),
        "frames": 0 if values is None else values.shape[0],
    }
    write_definitions(
        definitions, entries, kinds, references, rings, symmetric, soft, meta
    )
    print(f"{definitions}: {len(entries)} coordinates over {len(atoms)} atoms")
    if values is not None:
        value_path = prefix.with_suffix("." + args.format)
        write_values(value_path, values)
        print(f"{value_path}: {values.shape[0]} frames")
    return 0


def compare(path, entries, kinds, references):
    """Report the first disagreement with an existing definition file."""
    rows = read_definitions(path)
    if len(rows) != len(entries):
        print(
            f"coordinate count differs: {path} has {len(rows)}, "
            f"this run built {len(entries)}"
        )
        return 1
    for index, (row, entry) in enumerate(zip(rows, entries)):
        mine = (
            kinds[index],
            tuple(a + 1 for a in entry),
            None if references[index] is None else references[index] + 1,
        )
        theirs = (row["kind"], row["atoms"], row["reference"])
        if mine != theirs:
            print(f"coordinate {index + 1} differs: {path} has {theirs}, this run built {mine}")
            return 1
    print(f"{path}: identical, {len(entries)} coordinates")
    return 0


def read_definitions(path):
    """Parse a definition file written by this script or by molentropy."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("coordinate\t"):
                continue
            cells = line.split("\t")
            rows.append(
                {
                    "index": int(cells[0]),
                    "kind": cells[1],
                    "atoms": tuple(int(a) for a in cells[2].split("-")),
                    "reference": None if cells[3] == "-" else int(cells[3]),
                    "ring": bool(int(cells[4])),
                    "symmetric": bool(int(cells[5])),
                    "soft": bool(int(cells[6])),
                }
            )
    return rows


if __name__ == "__main__":
    sys.exit(main())
