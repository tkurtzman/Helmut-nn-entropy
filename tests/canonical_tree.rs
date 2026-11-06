//! `TreeRule::Canonical`: the backbone-rooted tree whose primary torsion about
//! every central bond is the one the torsion is named for.
//!
//! The assertions are on the coordinate definitions themselves — which four
//! atoms each primary torsion is built from — because that is the whole point
//! of the rule. A entropy value would only test that the coordinates are
//! self-consistent, which the mass-ordered tree already is.

mod common;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::Rng;
use common::amber::{
    NetcdfOptions, Topology, concat_frames, scratch_path, write_netcdf, write_prmtop,
};
use molentropy::bat_library::{BondSet, CoordinateOptions, InternalCoordinates, TreeRule};

const FRAMES: usize = 40;

/// Two residues of backbone with an alanine-like side chain, named as a PDB
/// file names them. Atom indices are referred to by name below.
const PEPTIDE: [(&str, &str, f64, i64); 16] = [
    ("N", "N", 14.01, 7),    // 0
    ("H", "H", 1.008, 1),    // 1
    ("CA", "CX", 12.01, 6),  // 2
    ("HA", "H1", 1.008, 1),  // 3
    ("CB", "CT", 12.01, 6),  // 4
    ("HB1", "HC", 1.008, 1), // 5
    ("HB2", "HC", 1.008, 1), // 6
    ("HB3", "HC", 1.008, 1), // 7
    ("C", "C", 12.01, 6),    // 8
    ("O", "O", 16.00, 8),    // 9
    ("N", "N", 14.01, 7),    // 10
    ("H", "H", 1.008, 1),    // 11
    ("CA", "CX", 12.01, 6),  // 12
    ("HA", "H1", 1.008, 1),  // 13
    ("C", "C", 12.01, 6),    // 14
    ("O", "O", 16.00, 8),    // 15
];
const PEPTIDE_BONDS: [(usize, usize); 15] = [
    (0, 1),
    (0, 2),
    (2, 3),
    (2, 4),
    (4, 5),
    (4, 6),
    (4, 7),
    (2, 8),
    (8, 9),
    (8, 10),
    (10, 11),
    (10, 12),
    (12, 13),
    (12, 14),
    (14, 15),
];

/// A threonine-like centre: CB carries OG1, CG2 and one hydrogen. IUPAC
/// numbers chi1 through the oxygen, not through the heavier branch.
const THREONINE: [(&str, &str, f64, i64); 12] = [
    ("N", "N", 14.01, 7),     // 0
    ("H", "H", 1.008, 1),     // 1
    ("CA", "CX", 12.01, 6),   // 2
    ("HA", "H1", 1.008, 1),   // 3
    ("CB", "CT", 12.01, 6),   // 4
    ("HB", "H1", 1.008, 1),   // 5
    ("OG1", "OH", 16.00, 8),  // 6
    ("HG1", "HO", 1.008, 1),  // 7
    ("CG2", "CT", 12.01, 6),  // 8
    ("HG21", "HC", 1.008, 1), // 9
    ("C", "C", 12.01, 6),     // 10
    ("O", "O", 16.00, 8),     // 11
];
const THREONINE_BONDS: [(usize, usize); 11] = [
    (0, 1),
    (0, 2),
    (2, 3),
    (2, 4),
    (4, 5),
    (4, 6),
    (6, 7),
    (4, 8),
    (8, 9),
    (2, 10),
    (10, 11),
];

const BUTANE: [(&str, &str, f64, i64); 8] = [
    ("C1", "c3", 12.01, 6),
    ("C2", "c3", 12.01, 6),
    ("H11", "hc", 1.008, 1),
    ("H12", "hc", 1.008, 1),
    ("H13", "hc", 1.008, 1),
    ("H21", "hc", 1.008, 1),
    ("H22", "hc", 1.008, 1),
    ("H23", "hc", 1.008, 1),
];
const BUTANE_BONDS: [(usize, usize); 7] = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)];

fn jittered(rng: &mut Rng, ideal: &[[f64; 3]]) -> Vec<Vec<[f64; 3]>> {
    (0..FRAMES)
        .map(|_| {
            ideal
                .iter()
                .map(|p| {
                    [
                        p[0] + 0.05 * rng.normal(),
                        p[1] + 0.05 * rng.normal(),
                        p[2] + 0.05 * rng.normal(),
                    ]
                })
                .collect()
        })
        .collect()
}

/// Positions do not matter for the tree, only for being able to read a
/// trajectory at all, so a loose spiral with jitter is enough.
fn spiral(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.9;
            [1.5 * t.cos() + 0.4 * i as f64, 1.5 * t.sin(), 0.7 * t]
        })
        .collect()
}

fn system(
    name: &str,
    atoms: &[(&'static str, &'static str, f64, i64)],
    bonds: &[(usize, usize)],
) -> (PathBuf, PathBuf) {
    let mut topology = Topology::default();
    topology.push_molecule("MOL", atoms, bonds);
    let top = scratch_path(&format!("canonical_{name}.parm7"));
    let traj = scratch_path(&format!("canonical_{name}.nc"));
    write_prmtop(&top, &topology);
    let mut rng = Rng::new(0xca_70_11);
    let frames = jittered(&mut rng, &spiral(atoms.len()));
    write_netcdf(&traj, &concat_frames(&[&frames]), NetcdfOptions::default());
    (top, traj)
}

fn load(top: &Path, traj: &Path, tree: TreeRule) -> InternalCoordinates {
    let options = CoordinateOptions {
        bonds: BondSet::Tree,
        tree,
        phase_angles: true,
        ..Default::default()
    };
    let mut internal = InternalCoordinates::with_options(top, options).unwrap();
    internal
        .calculate_internal_coords(traj, usize::MAX, false)
        .unwrap();
    internal
}

/// The atom quadruple of every primary torsion (the ones that stay torsions
/// rather than becoming phase angles).
fn primary_torsions(internal: &InternalCoordinates) -> Vec<Vec<usize>> {
    let references = internal.phase_references();
    internal
        .coordinate_atoms()
        .into_iter()
        .enumerate()
        .filter(|(index, atoms)| atoms.len() == 4 && references[*index].is_none())
        .map(|(_, atoms)| atoms)
        .collect()
}

fn has(torsions: &[Vec<usize>], wanted: [usize; 4]) -> bool {
    let reversed: Vec<usize> = wanted.iter().rev().copied().collect();
    torsions
        .iter()
        .any(|t| t[..] == wanted[..] || t[..] == reversed[..])
}

// ---------------------------------------------------------------------------
// The named torsions
// ---------------------------------------------------------------------------

#[test]
fn the_primary_torsions_are_the_named_ones() {
    let (top, traj) = system("peptide", &PEPTIDE, &PEPTIDE_BONDS);
    let torsions = primary_torsions(&load(&top, &traj, TreeRule::Canonical));

    // psi   : N-CA-C-N'
    assert!(has(&torsions, [10, 8, 2, 0]), "psi missing: {torsions:?}");
    // omega : CA-C-N'-CA'
    assert!(
        has(&torsions, [12, 10, 8, 2]),
        "omega missing: {torsions:?}"
    );
    // phi of the second residue: C-N'-CA'-C'
    assert!(has(&torsions, [14, 12, 10, 8]), "phi missing: {torsions:?}");
    // chi1 of an alanine-like side chain: N-CA-CB-HB1
    assert!(has(&torsions, [5, 4, 2, 0]), "chi1 missing: {torsions:?}");
}

#[test]
fn the_mass_ordered_tree_does_not_produce_them() {
    // Not a defect of that tree, but the reason this rule exists.
    let (top, traj) = system("peptide_mass", &PEPTIDE, &PEPTIDE_BONDS);
    let torsions = primary_torsions(&load(&top, &traj, TreeRule::MassOrdered));
    let named = [[10, 8, 2, 0], [12, 10, 8, 2], [14, 12, 10, 8]];
    let found = named.iter().filter(|&&t| has(&torsions, t)).count();
    assert!(
        found < named.len(),
        "mass-ordered matched all of them by chance: {torsions:?}"
    );
}

#[test]
fn chi1_follows_iupac_through_the_oxygen() {
    // CB carries OG1 and CG2; IUPAC numbers chi1 through OG1 even though the
    // methyl is the larger branch.
    let (top, traj) = system("threonine", &THREONINE, &THREONINE_BONDS);
    let torsions = primary_torsions(&load(&top, &traj, TreeRule::Canonical));
    assert!(
        has(&torsions, [6, 4, 2, 0]),
        "N-CA-CB-OG1 missing: {torsions:?}"
    );
    assert!(
        !has(&torsions, [8, 4, 2, 0]),
        "chi1 went through CG2 instead: {torsions:?}"
    );
}

// ---------------------------------------------------------------------------
// The construction is still a valid, deterministic BAT set
// ---------------------------------------------------------------------------

#[test]
fn the_coordinate_set_is_still_three_n_minus_six() {
    for (name, atoms, bonds) in [
        ("count_peptide", &PEPTIDE[..], &PEPTIDE_BONDS[..]),
        ("count_threonine", &THREONINE[..], &THREONINE_BONDS[..]),
        ("count_butane", &BUTANE[..], &BUTANE_BONDS[..]),
    ] {
        let (top, traj) = system(name, atoms, bonds);
        let internal = load(&top, &traj, TreeRule::Canonical);
        let n = atoms.len();
        assert_eq!(internal.dim, 3 * n - 6, "{name}");
        let kinds = internal.coordinate_atoms();
        let count = |k: usize| kinds.iter().filter(|a| a.len() == k).count();
        assert_eq!(count(2), n - 1, "{name}: bonds");
        assert_eq!(count(3), n - 2, "{name}: angles");
        assert_eq!(count(4), n - 3, "{name}: torsions");
    }
}

#[test]
fn the_tree_bonds_still_form_a_spanning_tree() {
    let (top, traj) = system("spanning", &PEPTIDE, &PEPTIDE_BONDS);
    let internal = load(&top, &traj, TreeRule::Canonical);
    let topology: HashSet<(usize, usize)> = PEPTIDE_BONDS
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    let mut component: Vec<usize> = (0..PEPTIDE.len()).collect();
    fn root(component: &mut [usize], mut a: usize) -> usize {
        while component[a] != a {
            a = component[a];
        }
        a
    }
    let mut edges = 0;
    for atoms in internal
        .coordinate_atoms()
        .into_iter()
        .filter(|a| a.len() == 2)
    {
        let bond = (atoms[0].min(atoms[1]), atoms[0].max(atoms[1]));
        assert!(topology.contains(&bond), "{bond:?} is not a bond");
        let (ra, rb) = (root(&mut component, bond.0), root(&mut component, bond.1));
        assert_ne!(ra, rb, "tree bond {bond:?} closes a cycle");
        component[ra] = rb;
        edges += 1;
    }
    assert_eq!(edges, PEPTIDE.len() - 1);
}

#[test]
fn a_molecule_without_a_backbone_still_builds() {
    // No atom named N/CA/C: the rule falls back to the highest-degree atom and
    // everything else is unchanged.
    let (top, traj) = system("no_backbone", &BUTANE, &BUTANE_BONDS);
    let canonical = load(&top, &traj, TreeRule::Canonical);
    assert_eq!(canonical.dim, 3 * BUTANE.len() - 6);
    assert_eq!(primary_torsions(&canonical).len(), 1, "one C-C rotor");
}

#[test]
fn the_default_is_the_mass_ordered_tree() {
    let (top, traj) = system("default", &PEPTIDE, &PEPTIDE_BONDS);
    let explicit = load(&top, &traj, TreeRule::MassOrdered);
    let implicit = {
        // Everything as before except `tree`, which is left at its default.
        let options = CoordinateOptions {
            bonds: BondSet::Tree,
            ..Default::default()
        };
        let mut internal = InternalCoordinates::with_options(&top, options).unwrap();
        internal
            .calculate_internal_coords(&traj, usize::MAX, false)
            .unwrap();
        internal
    };
    assert_eq!(TreeRule::default(), TreeRule::MassOrdered);
    assert_eq!(explicit.coordinate_atoms(), implicit.coordinate_atoms());
}

#[test]
fn the_tree_flag_reaches_the_program() {
    let (top, traj) = system("cli", &PEPTIDE, &PEPTIDE_BONDS);
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_molentropy"))
            .args([top.to_str().unwrap(), traj.to_str().unwrap()])
            .args(args)
            .output()
            .expect("failed to run CLI")
    };
    let total = |args: &[&str]| -> f64 {
        let output = run(args);
        assert!(
            output.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .lines()
            .last()
            .and_then(|line| line.strip_prefix("Total entropy = "))
            .expect("missing entropy output")
            .parse()
            .unwrap()
    };
    assert_eq!(total(&["--tree", "mass"]), total(&[]));
    assert_ne!(total(&["--tree", "canonical"]), total(&["--tree", "mass"]));

    let output = run(&["--tree", "backbone"]);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("must be one of"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
