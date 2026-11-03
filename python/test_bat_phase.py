#!/usr/bin/env python3
"""Checks for the parts of ``bat_phase.py`` that have no counterpart to
compare against on a real file.

The tree itself is checked by running the script against ``molentropy`` and
diffing the two coordinate lists, which is what ``--check`` does. What is left
over is the bookkeeping around it: which atoms count as solute, which count as
hydrogen, and how the canonical rules order a molecule's neighbours. Those are
mirrors of the Rust implementation, which is exactly where the two can drift
apart without anyone noticing, so they are pinned here.

Run with ``python -m pytest test_bat_phase.py`` or just ``python
test_bat_phase.py``.
"""

import math

import numpy as np

import bat_phase as bp


class FakeAtoms:
    """The handful of MDAnalysis attributes `Topology` reads."""

    def __init__(self, names, types, masses, resnames, elements):
        self.n_atoms = len(names)
        self.names = names
        self.types = types
        self.masses = np.asarray(masses, dtype=np.float64)
        self.resnames = resnames
        if elements is not None:
            self.elements = elements


class FakeUniverse:
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = type("Bonds", (), {"indices": bonds})()


def topology(names, types, masses, resnames=None, elements=None, bonds=()):
    resnames = resnames if resnames is not None else ["MOL"] * len(names)
    return bp.Topology(FakeUniverse(FakeAtoms(names, types, masses, resnames, elements), bonds))


# ---------------------------------------------------------------------------
# Which atoms are solute
# ---------------------------------------------------------------------------


def test_amber_water_types_are_excluded():
    top = topology(
        names=["C1", "C2", "O", "H1", "H2"],
        types=["c3", "c3", "OW", "HW", "HW"],
        masses=[12.01, 12.01, 15.999, 1.008, 1.008],
        bonds=[(0, 1), (2, 3), (2, 4)],
    )
    assert top.solute == [True, True, False, False, False]


def test_water_residue_names_are_excluded():
    # The types say nothing here, so the residue name has to carry it.
    top = topology(
        names=["C1", "C2", "O", "H1", "H2"],
        types=["c3", "c3", "o", "h1", "h1"],
        masses=[12.01, 12.01, 15.999, 1.008, 1.008],
        resnames=["MOL", "MOL", "SOL", "SOL", "SOL"],
        bonds=[(0, 1), (2, 3), (2, 4)],
    )
    assert top.solute == [True, True, False, False, False]
    for name in ("HOH", "TIP3", "SPCE", "OPC", "hoh"):
        top = topology(
            names=["C1", "C2", "O"],
            types=["c3", "c3", "o"],
            masses=[12.01, 12.01, 15.999],
            resnames=["MOL", "MOL", name],
            bonds=[(0, 1)],
        )
        assert top.solute[2] is False, name


def test_massless_sites_are_excluded():
    top = topology(
        names=["C1", "C2", "EP"],
        types=["c3", "c3", "ep"],
        masses=[12.01, 12.01, 0.0],
        bonds=[(0, 1), (1, 2)],
    )
    assert top.solute == [True, True, False]


def test_a_declared_atomic_number_of_zero_is_excluded():
    # A virtual site with a mass but no element.
    top = topology(
        names=["C1", "C2", "LP"],
        types=["c3", "c3", "lp"],
        masses=[12.01, 12.01, 1.0],
        elements=["C", "C", ""],
        bonds=[(0, 1), (1, 2)],
    )
    assert top.solute == [True, True, False]


def test_hydrogens_are_found_by_type_not_element():
    top = topology(
        names=["C1", "HA", "H2", "O"],
        types=["c3", "hc", "H1", "o"],
        masses=[12.01, 1.008, 1.008, 15.999],
        bonds=[(0, 1), (0, 2), (0, 3)],
    )
    assert top.hydrogens == {1, 2}
    # HW is water's hydrogen and is handled by the solute mask instead.
    top = topology(
        names=["O", "H1"],
        types=["OW", "HW"],
        masses=[15.999, 1.008],
        bonds=[(0, 1)],
    )
    assert top.hydrogens == set()


# ---------------------------------------------------------------------------
# Elements and names
# ---------------------------------------------------------------------------


def test_the_element_comes_from_the_topology_then_from_the_mass():
    top = topology(
        names=["C1", "N", "S"],
        types=["c3", "n", "s"],
        masses=[12.01, 14.007, 32.06],
        elements=["C", "N", "S"],
        bonds=[(0, 1), (1, 2)],
    )
    assert [top.element(i) for i in range(3)] == [6, 7, 16]

    # Without elements the mass rule takes over: round(mass / 1.9), at least 1,
    # with halfway cases going away from zero as Rust's f64::round does.
    bare = topology(
        names=["C1", "N", "H"],
        types=["c3", "n", "hc"],
        masses=[12.01, 14.007, 1.008],
        bonds=[(0, 1), (1, 2)],
    )
    assert [bare.element(i) for i in range(3)] == [6, 7, 1]
    assert bp._round_half_away(2.5) == 3
    assert bp._round_half_away(-2.5) == -3
    assert bp._round_half_away(0.53) == 1


def test_the_numeral_is_the_trailing_digits_of_the_name():
    top = topology(
        names=["CG", "CG1", "CG12", "C", "HB3"],
        types=["c3"] * 5,
        masses=[12.01] * 5,
        bonds=[(0, 1), (1, 2), (2, 3), (3, 4)],
    )
    assert [top.numeral(i) for i in range(5)] == [0, 1, 12, 0, 3]


def test_the_backbone_successor_walks_the_chain():
    assert bp.backbone_successor("N") == "CA"
    assert bp.backbone_successor("CA") == "C"
    assert bp.backbone_successor("C") == "N"
    assert bp.backbone_successor("CB") is None


# ---------------------------------------------------------------------------
# The canonical rules
# ---------------------------------------------------------------------------


def alanine():
    """A single alanine with its neighbours' backbone atoms: enough for phi,
    psi and the CA-CB bond. The flanking atoms are named ``C`` and ``N`` as a
    real topology names them, which is what the backbone rules key off."""
    names = ["N", "H", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O", "C", "N"]
    types = ["n", "hn", "c3", "h1", "c3", "hc", "hc", "hc", "c", "o", "c", "n"]
    masses = [14.007, 1.008, 12.01, 1.008, 12.01, 1.008, 1.008, 1.008, 12.01, 15.999, 12.01, 14.007]
    bonds = [
        (0, 1), (0, 2), (0, 10),
        (2, 3), (2, 4), (2, 8),
        (4, 5), (4, 6), (4, 7),
        (8, 9), (8, 11),
    ]
    return topology(names, types, masses, bonds=bonds)


def test_the_tree_is_rooted_at_the_backbone_nitrogen():
    top = alanine()
    tree = bp.CanonicalTree(sorted(range(top.natom)), top)
    assert tree.root[0] == 0, "the root triple starts at N"
    # N -> CA -> C is the backbone walk.
    assert tree.root == (0, 2, 8)


def test_the_named_torsion_is_primary_about_each_bond():
    top = alanine()
    entries, atoms = bp.build_coordinates(top, bonds="tree")
    references = bp.phase_references(entries)
    torsions = [
        (index, entry)
        for index, entry in enumerate(entries)
        if len(entry) == 4
    ]
    primary = {
        (min(e[1], e[2]), max(e[1], e[2])): e
        for index, e in reversed(torsions)
        if references[index] is None
    }
    atoms = lambda entry: tuple(entry)

    # Each primary is the named torsion, read from the atom it places back to
    # its reference: psi is N(i+1)-C-CA-N, about the CA-C bond.
    assert atoms(primary[(2, 8)]) == (11, 8, 2, 0)
    # phi is C(i-1)-N-CA-C, about the N-CA bond.
    assert atoms(primary[(0, 2)]) == (10, 0, 2, 8)
    # In place of chi1 this alanine has only HB, but the frame is the chi1
    # one: the atom it places, then CB, CA and N.
    assert atoms(primary[(2, 4)])[1:] == (4, 2, 0)


def test_heavy_atoms_outrank_hydrogens_and_numerals_order_branches():
    top = topology(
        names=["CB", "CG2", "CG1", "HB", "OG"],
        types=["c3", "c3", "c3", "hc", "oh"],
        masses=[12.01, 12.01, 12.01, 1.008, 15.999],
        elements=["C", "C", "C", "H", "O"],
        bonds=[(0, 1), (0, 2), (0, 3), (0, 4)],
    )
    tree = bp.CanonicalTree(list(range(5)), top)
    # Oxygen first (higher atomic number), then CG1 before CG2 by numeral,
    # and the hydrogen last.
    assert tree.ranked(0) == [4, 2, 1, 3]


def test_a_molecule_without_a_backbone_is_rooted_at_the_highest_degree_atom():
    # Neopentane-like: the central carbon has four neighbours.
    names = ["C1", "C2", "C3", "C4", "C0"]
    bonds = [(4, 0), (4, 1), (4, 2), (4, 3)]
    top = topology(names, ["c3"] * 5, [12.01] * 5, bonds=bonds)
    tree = bp.CanonicalTree(list(range(5)), top)
    assert tree.root[0] == 4 or tree.root[1] == 4


# ---------------------------------------------------------------------------
# Geometry and files
# ---------------------------------------------------------------------------


def iupac_dihedral(p0, p1, p2, p3):
    """The textbook dihedral, written out independently of `bat_phase`, in the
    usual IUPAC sign convention."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.atan2(float(np.dot(np.cross(b1, v), w)), float(np.dot(v, w)))


def test_the_torsion_sign_is_the_negative_of_the_iupac_convention():
    rng = np.random.default_rng(7)
    points = rng.normal(size=(20, 4, 3))
    positions = points.astype(np.float32)
    index = np.array([[0, 1, 2, 3]], dtype=np.intp)
    for frame in range(points.shape[0]):
        ours = bp.torsion_angles(positions[frame : frame + 1], index)[0, 0]
        theirs = iupac_dihedral(*points[frame])
        assert math.isclose(ours, -theirs, abs_tol=1e-5), (frame, ours, theirs)


def test_exact_and_vectorized_atan2_agree_to_an_ulp():
    rng = np.random.default_rng(0)
    y, x = rng.normal(size=5000), rng.normal(size=5000)
    exact = bp._atan2(y, x)
    fast = bp._atan2(y, x, exact=False)
    assert np.allclose(exact, fast, rtol=0, atol=4 * np.spacing(np.abs(exact)).max())
    # And the exact path really is the platform's atan2.
    assert exact[0] == math.atan2(y[0], x[0])


def test_wrapping_puts_an_angle_in_the_half_open_turn():
    values = np.array([0.0, math.pi, -math.pi, 3 * math.pi, -3 * math.pi, 7.0])
    wrapped = bp.wrap_angle(values)
    assert np.all(wrapped > -math.pi - 1e-12)
    assert np.all(wrapped <= math.pi + 1e-12)
    assert math.isclose(bp.wrap_angle(np.array([math.pi]))[0], math.pi)


def test_a_definition_file_reads_back_as_it_was_written(tmp_path=None):
    import tempfile, pathlib

    directory = pathlib.Path(tmp_path or tempfile.mkdtemp())
    top = alanine()
    entries, atoms = bp.build_coordinates(top, bonds="tree-unconstrained")
    references = bp.phase_references(entries)
    ring_bonds, symmetric_bonds = bp.bond_topology(top)
    kinds, rings, symmetric, soft = bp.classify(
        entries, references, ring_bonds, symmetric_bonds
    )
    path = directory / "test.bat"
    bp.write_definitions(
        path, entries, kinds, references, rings, symmetric, soft, {"test": "1"}
    )
    rows = bp.read_definitions(path)
    assert len(rows) == len(entries)
    for row, entry, kind, reference in zip(rows, entries, kinds, references):
        assert row["atoms"] == tuple(a + 1 for a in entry)
        assert row["kind"] == kind
        assert row["reference"] == (None if reference is None else reference + 1)
    # The methyl is a symmetric rotor, and only its primary torsion is soft.
    rotor = [i for i, flag in enumerate(symmetric) if flag]
    assert rotor, "the alanine methyl should be found"
    assert sum(1 for i in rotor if soft[i]) == 1


def test_the_soft_set_does_not_depend_on_the_phase_angles():
    top = alanine()
    entries, _ = bp.build_coordinates(top, bonds="tree")
    ring_bonds, symmetric_bonds = bp.bond_topology(top)
    with_phase = bp.classify(
        entries, bp.phase_references(entries, True), ring_bonds, symmetric_bonds
    )[3]
    without = bp.classify(
        entries, bp.phase_references(entries, False), ring_bonds, symmetric_bonds
    )[3]
    assert with_phase == without
    # Three bonds have an atom placed beyond them: N-CA, CA-C and CA-CB. The
    # flanking C and N are leaves, so their own bonds carry no torsion.
    assert sum(with_phase) == 3


def test_rings_are_the_bonds_that_are_not_bridges():
    # Cyclopropane's three carbons, each with a hydrogen hanging off.
    names = ["C1", "C2", "C3", "H1", "H2", "H3"]
    bonds = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 4), (2, 5)]
    top = topology(names, ["c3"] * 3 + ["hc"] * 3, [12.01] * 3 + [1.008] * 3, bonds=bonds)
    ring_bonds, _ = bp.bond_topology(top)
    assert ring_bonds == {(0, 1), (1, 2), (0, 2)}


def test_a_symmetric_rotor_needs_two_or_more_identical_terminal_atoms():
    methyl = topology(
        ["C1", "C2", "H1", "H2", "H3"],
        ["c3", "c3", "hc", "hc", "hc"],
        [12.01, 12.01, 1.008, 1.008, 1.008],
        elements=["C", "C", "H", "H", "H"],
        bonds=[(0, 1), (1, 2), (1, 3), (1, 4)],
    )
    _, rotors = bp.bond_topology(methyl)
    assert (0, 1) in rotors

    # One hydrogen and one fluorine is not a rotor.
    mixed = topology(
        ["C1", "C2", "H1", "F1"],
        ["c3", "c3", "hc", "f"],
        [12.01, 12.01, 1.008, 18.998],
        elements=["C", "C", "H", "F"],
        bonds=[(0, 1), (1, 2), (1, 3)],
    )
    _, rotors = bp.bond_topology(mixed)
    assert rotors == set()


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
        except AssertionError as err:
            failures += 1
            print(f"FAIL {name}: {err}")
        except Exception as err:  # noqa: BLE001 - report and keep going
            failures += 1
            print(f"ERROR {name}: {type(err).__name__}: {err}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
