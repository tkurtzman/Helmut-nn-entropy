#!/usr/bin/env python3
"""Checks for ``atomic_map.py``.

The arithmetic the module does is small, so what matters is the structure:
that no rule can invent or lose entropy, that the pooled rule really is a
generalisation of the quarter rule rather than a different thing, and that
the parser survives the summary lines molentropy prints around its table.

Run with ``python -m pytest test_atomic_map.py`` or just ``python
test_atomic_map.py``.
"""

import math
import os
import tempfile

import numpy as np

import atomic_map as am


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def coord(index, kind, atoms, share, block="hard"):
    """A Coordinate from 1-based atom numbers, as the table prints them."""
    return am.Coordinate(index, kind, [a - 1 for a in atoms], share, block)


def methyl_set():
    """A methyl rotor: C-C bond with three hydrogens, one primary + two phases.

    Atoms: 1,2 the central bond; 3 the reference atom of the primary; 4,5,6
    the three hydrogens, which are chemically equivalent and so are given
    equal phase-angle entropies.
    """
    return [
        coord(1, am.TORSION, [4, 2, 1, 3], 1.5, "soft"),
        coord(2, am.PHASE, [5, 2, 1, 3], 0.3),
        coord(3, am.PHASE, [6, 2, 1, 3], 0.3),
    ]


def write(text):
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


TABLE = """Soft coordinates = 1
Soft entropy = 1.5
Hard entropy = 0.6
Coupling term = 0
coordinate\ttype\tatoms\tentropy\tset\town\tcoupling
1\ttorsion\t4-2-1-3\t1.5\tsoft\t1.5\t0
2\tphase angle\t5-2-1-3\t0.25\thard\t0.25\t0
3\tphase angle\t6-2-1-3\t0.35\thard\t0.35\t0
Mean ln J = 0.0
Total entropy = 2.1
"""


# --------------------------------------------------------------------------
# every rule conserves the total
# --------------------------------------------------------------------------


def test_weights_sum_to_one_per_coordinate():
    coords = methyl_set() + [
        coord(4, am.BOND, [1, 2], -1.9),
        coord(5, am.ANGLE, [4, 2, 1], -0.7),
    ]
    for rule in am.RULES:
        for w in am.RULES[rule](coords):
            assert abs(sum(w.values()) - 1.0) < 1e-12, rule


def test_total_is_rule_independent():
    coords = methyl_set() + [
        coord(4, am.BOND, [1, 2], -1.9),
        coord(5, am.ANGLE, [4, 2, 1], -0.7),
    ]
    want = sum(c.share for c in coords)
    for rule in am.RULES:
        got = am.map_shares(coords, 6, rule).sum()
        assert abs(got - want) < 1e-12, (rule, got, want)


# --------------------------------------------------------------------------
# the pooled rule
# --------------------------------------------------------------------------


def test_pooled_reduces_to_quarters_for_a_lone_torsion():
    """With no phase angles about the bond, pooling must change nothing."""
    coords = [coord(1, am.TORSION, [4, 2, 1, 3], 1.0)]
    pooled = am.map_shares(coords, 4, "pooled")
    quarter = am.map_shares(coords, 4, "quarter")
    assert np.allclose(pooled, quarter)
    assert np.allclose(pooled, 0.25)


def test_pooled_gives_equivalent_children_equal_shares():
    """The three hydrogens of a methyl are equivalent and must come out so."""
    coords = methyl_set()
    pooled = am.map_shares(coords, 6, "pooled")
    assert abs(pooled[3] - pooled[4]) < 1e-12
    assert abs(pooled[4] - pooled[5]) < 1e-12


def test_quarter_rule_splits_equivalent_children():
    """The behaviour the pooled rule exists to remove, pinned so it is visible."""
    coords = methyl_set()
    quarter = am.map_shares(coords, 6, "quarter")
    # atom 4 is the primary's child and is credited in both phase angles too
    assert quarter[3] > quarter[4] + 0.1
    assert abs(quarter[4] - quarter[5]) < 1e-12


def test_pooled_keeps_the_central_bond_weight_at_a_quarter_each():
    coords = methyl_set()
    total = sum(c.share for c in coords)
    pooled = am.map_shares(coords, 6, "pooled")
    assert abs(pooled[0] - total / 4) < 1e-12
    assert abs(pooled[1] - total / 4) < 1e-12


def test_pooled_direction_of_the_central_bond_does_not_matter():
    """A phase angle traversing the bond the other way joins the same group."""
    forward = methyl_set()
    reversed_ = methyl_set()
    c = reversed_[2]
    c.atoms = [c.atoms[0], c.atoms[2], c.atoms[1], c.atoms[3]]
    assert np.allclose(
        am.map_shares(forward, 6, "pooled"), am.map_shares(reversed_, 6, "pooled")
    )


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_reads_table_around_summary_lines():
    path = write(TABLE)
    try:
        coords = am.read_per_coordinate(path)
    finally:
        os.unlink(path)
    assert len(coords) == 3
    assert coords[0].kind == am.TORSION
    assert coords[0].atoms == [3, 1, 0, 2]
    assert coords[0].block == "soft"
    assert abs(coords[2].share - 0.35) < 1e-12


def test_reads_a_table_without_the_set_column():
    """qh/bqh print entropy, first_order, correlation and no set column."""
    path = write(
        "coordinate\ttype\tatoms\tentropy\tfirst_order\tcorrelation\n"
        "1\tbond\t1-2\t-1.9\t-1.8\t-0.1\n"
        "Total entropy = -1.9\n"
    )
    try:
        coords = am.read_per_coordinate(path)
    finally:
        os.unlink(path)
    assert len(coords) == 1
    assert coords[0].block is None
    assert abs(coords[0].share + 1.9) < 1e-12


def test_missing_table_is_an_error():
    path = write("Total entropy = 3.0\n")
    try:
        try:
            am.read_per_coordinate(path)
        except SystemExit:
            return
    finally:
        os.unlink(path)
    raise AssertionError("expected SystemExit for a file with no table")


# --------------------------------------------------------------------------
# the Jacobian
# --------------------------------------------------------------------------


def test_jacobian_terms_match_the_closed_form():
    coords = [
        coord(1, am.BOND, [1, 2], 0.0),
        coord(2, am.ANGLE, [3, 2, 1], 0.0),
        coord(3, am.TORSION, [4, 2, 1, 3], 0.0),
    ]
    values = np.array([[1.5, 2.0, 0.3], [1.5, 2.0, 0.9]])
    per_atom, total = am.map_jacobian(coords, values, 4)
    want = 2.0 * math.log(1.5) + math.log(math.sin(2.0))
    assert abs(total - want) < 1e-12
    assert abs(per_atom.sum() - want) < 1e-12
    assert abs(per_atom[3]) < 1e-12  # the torsion contributes nothing


# --------------------------------------------------------------------------
# atom count
# --------------------------------------------------------------------------


def test_natoms_is_inferred_from_the_highest_index():
    assert am.natoms_of(methyl_set()) == 6


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   %s" % name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL %s: %s" % (name, exc))
    raise SystemExit(1 if failures else 0)
