//! `BondSet::TreeUnconstrained`: the spanning tree without the bonds a
//! constraint algorithm holds fixed.
//!
//! The point of the option is that it changes the bonds and nothing else, so
//! most of these tests compare it against `BondSet::Tree` coordinate by
//! coordinate rather than against recorded numbers. The Jacobian test checks
//! the one quantity that does change, by the amount it must.

mod common;

use std::collections::HashSet;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::Rng;
use common::amber::{
    NetcdfOptions, Topology, concat_frames, scratch_path, write_netcdf, write_prmtop,
};
use molentropy::bat_library::{BondSet, CoordinateOptions, InternalCoordinates};

const FRAMES: usize = 40;

const ETHANE_ATOMS: [(&str, &str, f64, i64); 8] = [
    ("C1", "c3", 12.01, 6),
    ("C2", "c3", 12.01, 6),
    ("H11", "hc", 1.008, 1),
    ("H12", "hc", 1.008, 1),
    ("H13", "hc", 1.008, 1),
    ("H21", "hc", 1.008, 1),
    ("H22", "hc", 1.008, 1),
    ("H23", "hc", 1.008, 1),
];
const ETHANE_BONDS: [(usize, usize); 7] = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)];

const CYCLOPROPANE_ATOMS: [(&str, &str, f64, i64); 9] = [
    ("C1", "cx", 12.01, 6),
    ("C2", "cx", 12.01, 6),
    ("C3", "cx", 12.01, 6),
    ("H11", "hc", 1.008, 1),
    ("H12", "hc", 1.008, 1),
    ("H21", "hc", 1.008, 1),
    ("H22", "hc", 1.008, 1),
    ("H31", "hc", 1.008, 1),
    ("H32", "hc", 1.008, 1),
];
const CYCLOPROPANE_BONDS: [(usize, usize); 9] = [
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (0, 4),
    (1, 5),
    (1, 6),
    (2, 7),
    (2, 8),
];

/// Ethane along x with tetrahedral methyls, the second one rotated about the
/// C-C axis each frame.
fn ethane_frames(rng: &mut Rng) -> Vec<Vec<[f64; 3]>> {
    let radial = (8.0_f64 / 9.0).sqrt() * 1.09;
    let axial = 1.09 / 3.0;
    let jitter = |rng: &mut Rng, p: [f64; 3]| {
        [
            p[0] + 0.01 * rng.normal(),
            p[1] + 0.01 * rng.normal(),
            p[2] + 0.01 * rng.normal(),
        ]
    };
    (0..FRAMES)
        .map(|_| {
            let rotation = rng.uniform_range(0.0, 2.0 * PI);
            let mut atoms = vec![jitter(rng, [0.0, 0.0, 0.0]), jitter(rng, [1.53, 0.0, 0.0])];
            for k in 0..3 {
                let phi = 2.0 * PI * k as f64 / 3.0;
                atoms.push(jitter(
                    rng,
                    [-axial, radial * phi.cos(), radial * phi.sin()],
                ));
            }
            for k in 0..3 {
                let phi = 2.0 * PI * k as f64 / 3.0 + rotation;
                atoms.push(jitter(
                    rng,
                    [1.53 + axial, radial * phi.cos(), radial * phi.sin()],
                ));
            }
            atoms
        })
        .collect()
}

fn cyclopropane_frames(rng: &mut Rng) -> Vec<Vec<[f64; 3]>> {
    let ideal: [[f64; 3]; 9] = [
        [0.0, 0.87, 0.0],
        [-0.75, -0.43, 0.0],
        [0.75, -0.43, 0.0],
        [0.0, 1.45, 0.9],
        [0.0, 1.45, -0.9],
        [-1.25, -0.72, 0.9],
        [-1.25, -0.72, -0.9],
        [1.25, -0.72, 0.9],
        [1.25, -0.72, -0.9],
    ];
    (0..FRAMES)
        .map(|_| {
            ideal
                .iter()
                .map(|p| {
                    [
                        p[0] + 0.03 * rng.normal(),
                        p[1] + 0.03 * rng.normal(),
                        p[2] + 0.03 * rng.normal(),
                    ]
                })
                .collect()
        })
        .collect()
}

fn write_system(
    name: &str,
    atoms: &[(&'static str, &'static str, f64, i64)],
    bonds: &[(usize, usize)],
    frames: &Vec<Vec<[f64; 3]>>,
) -> (PathBuf, PathBuf) {
    let mut topology = Topology::default();
    topology.push_molecule("MOL", atoms, bonds);
    let top = scratch_path(&format!("constrained_{name}.parm7"));
    let traj = scratch_path(&format!("constrained_{name}.nc"));
    write_prmtop(&top, &topology);
    write_netcdf(&traj, &concat_frames(&[frames]), NetcdfOptions::default());
    (top, traj)
}

fn ethane(name: &str) -> (PathBuf, PathBuf) {
    let mut rng = Rng::new(0x5eed_e7a2);
    write_system(name, &ETHANE_ATOMS, &ETHANE_BONDS, &ethane_frames(&mut rng))
}

fn cyclopropane(name: &str) -> (PathBuf, PathBuf) {
    let mut rng = Rng::new(0xc3c3_c3c3);
    write_system(
        name,
        &CYCLOPROPANE_ATOMS,
        &CYCLOPROPANE_BONDS,
        &cyclopropane_frames(&mut rng),
    )
}

fn load(top: &Path, traj: &Path, bonds: BondSet) -> InternalCoordinates {
    let options = CoordinateOptions {
        bonds,
        phase_angles: false,
        ..Default::default()
    };
    let mut internal = InternalCoordinates::with_options(top, options).unwrap();
    internal
        .calculate_internal_coords(traj, usize::MAX, false)
        .unwrap();
    internal
}

/// The coordinates of one kind, as (atoms, values over frames).
fn of_size(internal: &InternalCoordinates, size: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let atoms = internal.coordinate_atoms();
    let keep: Vec<usize> = (0..atoms.len())
        .filter(|&i| atoms[i].len() == size)
        .collect();
    (
        keep.iter().map(|&i| atoms[i].clone()).collect(),
        internal
            .int_coords
            .iter()
            .map(|frame| keep.iter().map(|&i| frame[i]).collect())
            .collect(),
    )
}

fn hydrogens(atoms: &[(&'static str, &'static str, f64, i64)]) -> HashSet<usize> {
    atoms
        .iter()
        .enumerate()
        .filter(|(_, (_, kind, _, _))| kind.starts_with('h'))
        .map(|(index, _)| index)
        .collect()
}

// ---------------------------------------------------------------------------
// Which bonds survive
// ---------------------------------------------------------------------------

#[test]
fn only_the_bonds_to_hydrogen_are_dropped() {
    let (top, traj) = ethane("ethane_counts");
    let tree = load(&top, &traj, BondSet::Tree);
    let unconstrained = load(&top, &traj, BondSet::TreeUnconstrained);
    let (tree_bonds, _) = of_size(&tree, 2);
    let (kept, _) = of_size(&unconstrained, 2);

    // Ethane's tree holds all seven bonds; six of them reach a hydrogen.
    assert_eq!(tree_bonds.len(), 7);
    assert_eq!(kept.len(), 1, "only C-C is unconstrained");
    assert_eq!(
        kept[0].iter().copied().collect::<HashSet<usize>>(),
        HashSet::from([0, 1]),
        "the kept bond is C-C, whichever way round the tree stores it"
    );
    // 3N - 6 less the six constrained bonds.
    assert_eq!(unconstrained.dim, 3 * 8 - 6 - 6);
    assert_eq!(unconstrained.int_coords[0].len(), unconstrained.dim);
}

#[test]
fn the_kept_bonds_are_exactly_the_trees_heavy_bonds() {
    let (top, traj) = cyclopropane("cyclopropane_bonds");
    let tree = load(&top, &traj, BondSet::Tree);
    let unconstrained = load(&top, &traj, BondSet::TreeUnconstrained);
    let hydrogen = hydrogens(&CYCLOPROPANE_ATOMS);
    let expected: Vec<Vec<usize>> = of_size(&tree, 2)
        .0
        .into_iter()
        .filter(|bond| !bond.iter().any(|atom| hydrogen.contains(atom)))
        .collect();
    assert_eq!(of_size(&unconstrained, 2).0, expected);
    // A ring closure is still left out, as in the tree set.
    assert_eq!(expected.len(), 2, "C-C, C-C, no ring closure");
}

#[test]
fn the_heavy_atom_set_is_not_the_same_thing() {
    // Both keep only heavy bonds, but the heavy-atom set is every heavy bond,
    // ring closures included, while this one follows the tree.
    let (top, traj) = cyclopropane("cyclopropane_vs_heavy");
    let heavy = load(&top, &traj, BondSet::HeavyAtom);
    let unconstrained = load(&top, &traj, BondSet::TreeUnconstrained);
    assert_eq!(of_size(&heavy, 2).0.len(), 3, "all three ring bonds");
    assert_eq!(of_size(&unconstrained, 2).0.len(), 2);
}

#[test]
fn angles_and_torsions_are_untouched() {
    // The spanning tree is built the same way, so dropping constrained bonds
    // must not move a single angle or torsion.
    for (top, traj) in [ethane("ethane_same"), cyclopropane("cyclopropane_same")] {
        let tree = load(&top, &traj, BondSet::Tree);
        let unconstrained = load(&top, &traj, BondSet::TreeUnconstrained);
        for size in [3, 4] {
            assert_eq!(
                of_size(&tree, size),
                of_size(&unconstrained, size),
                "size {size}"
            );
        }
    }
}

#[test]
fn phase_angles_still_apply() {
    let (top, traj) = ethane("ethane_phase");
    let options = |phase_angles| CoordinateOptions {
        bonds: BondSet::TreeUnconstrained,
        phase_angles,
        ..Default::default()
    };
    let load = |phase_angles| {
        let mut internal = InternalCoordinates::with_options(&top, options(phase_angles)).unwrap();
        internal
            .calculate_internal_coords(&traj, usize::MAX, false)
            .unwrap();
        internal
    };
    let raw = load(false);
    let phase = load(true);
    assert_eq!(raw.dim, phase.dim);
    assert!(
        phase.phase_references().iter().any(Option::is_some),
        "a methyl's torsions should become phase angles"
    );
}

// ---------------------------------------------------------------------------
// Jacobian
// ---------------------------------------------------------------------------

#[test]
fn the_jacobian_loses_exactly_the_constrained_bond_terms() {
    let (top, traj) = ethane("ethane_jacobian");
    let tree = load(&top, &traj, BondSet::Tree);
    let unconstrained = load(&top, &traj, BondSet::TreeUnconstrained);
    let hydrogen = hydrogens(&ETHANE_ATOMS);
    let (tree_bonds, tree_lengths) = of_size(&tree, 2);
    let dropped: Vec<usize> = (0..tree_bonds.len())
        .filter(|&i| tree_bonds[i].iter().any(|atom| hydrogen.contains(atom)))
        .collect();
    assert_eq!(dropped.len(), 6);

    let full = tree.log_jacobian().unwrap();
    let reduced = unconstrained.log_jacobian().unwrap();
    assert_eq!(full.len(), reduced.len());
    for (frame, (full, reduced)) in full.iter().zip(&reduced).enumerate() {
        let constrained: f64 = dropped
            .iter()
            .map(|&i| 2.0 * tree_lengths[frame][i].ln())
            .sum();
        assert!(
            (full - reduced - constrained).abs() < 1e-12,
            "frame {frame}: {full} - {reduced} vs {constrained}"
        );
    }
}

#[test]
fn the_jacobian_is_still_refused_for_the_heavy_atom_set() {
    let (top, traj) = ethane("ethane_jacobian_refused");
    let err = load(&top, &traj, BondSet::HeavyAtom)
        .log_jacobian()
        .unwrap_err()
        .to_string();
    assert!(err.contains("tree bond set"), "{err}");
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[test]
fn the_bonds_flag_selects_the_set() {
    let (top, traj) = ethane("cli");
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
    assert_eq!(total(&["--bonds", "tree"]), total(&["--tree-bonds"]));
    assert_eq!(total(&["--bonds", "heavy"]), total(&[]));
    // Six fewer coordinates, so a different total, and the Jacobian is allowed.
    let unconstrained = total(&["--bonds", "tree-unconstrained"]);
    assert_ne!(unconstrained, total(&["--bonds", "tree"]));
    assert_eq!(
        total(&["--bonds", "tree-unconstrained", "--jacobian"]),
        unconstrained
    );

    for (args, message) in [
        (
            &["--bonds", "tree", "--tree-bonds"][..],
            "pass one or the other",
        ),
        (&["--bonds", "none"][..], "must be one of"),
        (
            &["--bonds", "heavy", "--jacobian"][..],
            "--jacobian requires --tree-bonds",
        ),
    ] {
        let output = run(args);
        assert!(!output.status.success(), "{args:?} should fail");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(message), "{args:?}: {stderr}");
    }
}
