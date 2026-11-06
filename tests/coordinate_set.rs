//! [`CoordinateSet`]: writing the generated coordinates out and reading a set
//! back in.
//!
//! The point of the round trip is that it changes nothing, so most of these
//! are exact identities against the generator rather than numbers only this
//! code can produce.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::Rng;
use common::amber::{
    CHAIN_ATOMS, CHAIN_BONDS, NetcdfOptions, Topology, chain_frames, concat_frames, scratch_path,
    write_netcdf, write_prmtop,
};
use molentropy::bat_library::{
    BondSet, CoordinateKind, CoordinateOptions, InternalCoordinates, TreeRule,
};
use molentropy::coordinate_set::CoordinateSet;

/// A chain molecule with more frames than coordinates.
fn chain(name: &str) -> (PathBuf, PathBuf) {
    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    let top = scratch_path(&format!("set_{name}.parm7"));
    let traj = scratch_path(&format!("set_{name}.nc"));
    write_prmtop(&top, &topology);
    let mut rng = Rng::new(0x5e7);
    let frames = chain_frames(&mut rng, 120, [0.0, 0.0, 0.0]);
    write_netcdf(&traj, &concat_frames(&[&frames]), NetcdfOptions::default());
    (top, traj)
}

fn generated(top: &Path, traj: &Path, bonds: BondSet) -> CoordinateSet {
    let options = CoordinateOptions {
        bonds,
        tree: TreeRule::Canonical,
        ..Default::default()
    };
    let mut internal = InternalCoordinates::with_options(top, options).unwrap();
    internal
        .calculate_internal_coords(traj, usize::MAX, false)
        .unwrap();
    CoordinateSet::from_internal(internal, options)
}

/// Writes `set` to a scratch pair of files and reads it back.
fn round_trip(set: &CoordinateSet, name: &str, extension: &str) -> CoordinateSet {
    let definitions = scratch_path(&format!("{name}.bat"));
    let values = scratch_path(&format!("{name}.{extension}"));
    set.write_definitions(&definitions, &[("test", name.to_string())])
        .unwrap();
    set.write_values(&values).unwrap();
    CoordinateSet::read(&definitions, Some(&values)).unwrap()
}

// ---------------------------------------------------------------------------
// Exact identities
// ---------------------------------------------------------------------------

#[test]
fn a_written_set_reads_back_as_the_same_set() {
    let (top, traj) = chain("round");
    let set = generated(&top, &traj, BondSet::TreeUnconstrained);
    let read = round_trip(&set, "round", "npy");
    assert_eq!(read, set, "the round trip changed the set");
}

#[test]
fn the_text_and_numpy_forms_carry_the_same_values() {
    let (top, traj) = chain("formats");
    let set = generated(&top, &traj, BondSet::HeavyAtom);
    let npy = round_trip(&set, "formats_npy", "npy");
    let tsv = round_trip(&set, "formats_tsv", "tsv");
    assert_eq!(npy.values, set.values, "the .npy values differ");
    assert_eq!(tsv.values, set.values, "the text values differ");
}

#[test]
fn the_read_set_describes_the_coordinates_as_the_generator_does() {
    let (top, traj) = chain("describe");
    let options = CoordinateOptions {
        bonds: BondSet::TreeUnconstrained,
        tree: TreeRule::Canonical,
        ..Default::default()
    };
    let mut internal = InternalCoordinates::with_options(&top, options).unwrap();
    internal
        .calculate_internal_coords(&traj, usize::MAX, false)
        .unwrap();
    let expected_atoms = internal.coordinate_atoms();
    let expected_kinds = internal.coordinate_kinds();
    let expected_soft = internal.soft_torsions();
    let expected_symmetric = internal.symmetric_rotor_torsions();
    let expected_rings = internal.ring_torsions();
    let expected_metrics = internal.coordinate_metrics();
    let expected_jacobian = internal.log_jacobian().unwrap();

    let read = round_trip(
        &CoordinateSet::from_internal(internal, options),
        "describe",
        "npy",
    );
    assert_eq!(read.coordinate_atoms(), expected_atoms);
    assert_eq!(read.coordinate_kinds(), expected_kinds);
    assert_eq!(read.soft_torsions(), expected_soft);
    assert_eq!(read.symmetric_rotor_torsions(), expected_symmetric);
    assert_eq!(read.ring_torsions(), expected_rings);
    assert_eq!(read.coordinate_metrics(), expected_metrics);
    assert_eq!(read.log_jacobian().unwrap(), expected_jacobian);
    // The chain has three rotatable bonds, so the soft set is not empty and
    // the comparison above has something to compare.
    assert_eq!(expected_soft.len(), 3);
}

#[test]
fn only_the_soft_torsions_about_a_symmetric_rotor_are_reported() {
    // The flags describe the central bond, so every torsion about a methyl
    // carries the symmetric flag, but only the primary one defines states.
    let table = GOOD.replace(
        "3\ttorsion\t1-2-3-4\t-\t0\t0\t1\n4\tphase\t5-2-3-4\t3\t0\t0\t0",
        "3\ttorsion\t1-2-3-4\t-\t0\t1\t1\n4\tphase\t5-2-3-4\t3\t0\t1\t0",
    );
    let path = scratch_path("symmetric.bat");
    std::fs::write(&path, &table).unwrap();
    let set = CoordinateSet::read(&path, None).unwrap();
    assert_eq!(set.soft_torsions(), vec![2]);
    assert_eq!(set.symmetric_rotor_torsions(), vec![2]);
}

#[test]
fn the_bond_set_travels_with_the_definitions() {
    let (top, traj) = chain("jacobian");
    // Heavy-atom bonds are redundant, so the Jacobian is not defined for
    // them, and the read set has to know that without the topology.
    let heavy = round_trip(&generated(&top, &traj, BondSet::HeavyAtom), "heavy", "npy");
    let err = heavy.log_jacobian().unwrap_err().to_string();
    assert!(err.contains("tree bond set"), "{err}");

    let tree = round_trip(&generated(&top, &traj, BondSet::Tree), "tree", "npy");
    assert_eq!(tree.log_jacobian().unwrap().len(), tree.values.len());
}

#[test]
fn selecting_frames_works_the_same_on_a_read_set() {
    let (top, traj) = chain("frames");
    let set = generated(&top, &traj, BondSet::HeavyAtom);
    let read = round_trip(&set, "frames", "npy");
    assert_eq!(read.columns(7, 3).unwrap(), set.columns(7, 3).unwrap());
    assert!(read.columns(0, 0).is_err(), "a zero stride is refused");
    assert!(
        read.columns(read.values.len(), 1).is_err(),
        "a start past the last frame is refused"
    );
}

// ---------------------------------------------------------------------------
// Hand-built sets
// ---------------------------------------------------------------------------

/// Two torsions about the same bond and the bond itself.
fn hand_built() -> (Vec<Vec<usize>>, Vec<CoordinateKind>) {
    (
        vec![
            vec![0, 1],
            vec![0, 1, 2],
            vec![0, 1, 2, 3],
            vec![4, 1, 2, 3],
        ],
        vec![
            CoordinateKind::Bond,
            CoordinateKind::Angle,
            CoordinateKind::Torsion,
            CoordinateKind::PhaseAngle { reference: 2 },
        ],
    )
}

#[test]
fn a_hand_built_set_checks_its_own_pieces() {
    let (atoms, kinds) = hand_built();
    let values = vec![vec![1.0, 2.0, 3.0, 4.0]; 5];
    let set = CoordinateSet::new(
        atoms.clone(),
        kinds.clone(),
        vec![],
        vec![],
        vec![false, false, true, false],
        BondSet::Tree,
        values.clone(),
    )
    .unwrap();
    assert_eq!(set.dim, 4);
    assert_eq!(set.soft_torsions(), vec![2]);
    // Empty flag lists mean none, so nothing is on a ring here.
    assert!(set.ring_torsions().is_empty());

    let refuses = |atoms: Vec<Vec<usize>>, kinds: Vec<CoordinateKind>, values: Vec<Vec<f64>>| {
        CoordinateSet::new(atoms, kinds, vec![], vec![], vec![], BondSet::Tree, values)
            .unwrap_err()
            .to_string()
    };
    let short = vec![vec![0, 1], vec![0, 1, 2], vec![0, 1, 2], vec![4, 1, 2, 3]];
    assert!(refuses(short, kinds.clone(), values.clone()).contains("needs 4 atoms"));

    let forward = vec![
        CoordinateKind::Bond,
        CoordinateKind::Angle,
        CoordinateKind::PhaseAngle { reference: 3 },
        CoordinateKind::Torsion,
    ];
    assert!(refuses(atoms.clone(), forward, values.clone()).contains("not earlier"));

    let of_an_angle = vec![
        CoordinateKind::Bond,
        CoordinateKind::Angle,
        CoordinateKind::Torsion,
        CoordinateKind::PhaseAngle { reference: 1 },
    ];
    assert!(refuses(atoms.clone(), of_an_angle, values.clone()).contains("not a torsion"));

    let ragged = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0]];
    assert!(refuses(atoms, kinds, ragged).contains("has 2 values"));
}

// ---------------------------------------------------------------------------
// Malformed files
// ---------------------------------------------------------------------------

/// Writes `table` to a scratch file and returns the error it provokes.
fn rejected(name: &str, table: &str) -> String {
    let path = scratch_path(&format!("bad_{name}.bat"));
    std::fs::write(&path, table).unwrap();
    CoordinateSet::read(&path, None).unwrap_err().to_string()
}

const GOOD: &str = "\
# molentropy coordinate set 1
# bonds: tree
coordinate\tkind\tatoms\treference\tring\tsymmetric\tsoft
1\tbond\t1-2\t-\t0\t0\t0
2\tangle\t1-2-3\t-\t0\t0\t0
3\ttorsion\t1-2-3-4\t-\t0\t0\t1
4\tphase\t5-2-3-4\t3\t0\t0\t0
";

#[test]
fn a_well_formed_table_is_accepted() {
    let path = scratch_path("bad_good.bat");
    std::fs::write(&path, GOOD).unwrap();
    let set = CoordinateSet::read(&path, None).unwrap();
    assert_eq!(set.dim, 4);
    assert_eq!(
        set.coordinate_kinds()[3],
        CoordinateKind::PhaseAngle { reference: 2 }
    );
    // Atoms are numbered from 1 in the file and from 0 in the set.
    assert_eq!(set.coordinate_atoms()[3], vec![4, 1, 2, 3]);
    assert_eq!(set.soft_torsions(), vec![2]);
    assert!(set.values.is_empty(), "no values file was given");
}

#[test]
fn a_malformed_table_says_what_is_wrong() {
    let cases = [
        (
            "kind",
            "torsion\t1-2-3-4\t-",
            "quadruple\t1-2-3-4\t-",
            "unknown coordinate kind",
        ),
        ("atoms", "angle\t1-2-3\t-", "angle\t1-2\t-", "needs 3 atoms"),
        ("zero", "bond\t1-2\t-", "bond\t0-2\t-", "numbered from 1"),
        (
            "dangling",
            "phase\t5-2-3-4\t3",
            "phase\t5-2-3-4\t-",
            "needs a reference",
        ),
        (
            "spurious",
            "torsion\t1-2-3-4\t-",
            "torsion\t1-2-3-4\t2",
            "cannot have a reference",
        ),
        (
            "forward",
            "phase\t5-2-3-4\t3",
            "phase\t5-2-3-4\t4",
            "not an earlier coordinate",
        ),
        (
            "of_angle",
            "phase\t5-2-3-4\t3",
            "phase\t5-2-3-4\t2",
            "not a torsion",
        ),
        (
            "flag",
            "1-2-3-4\t-\t0\t0\t1",
            "1-2-3-4\t-\t0\t0\t2",
            "must be 0 or 1",
        ),
        (
            "bonds",
            "# bonds: tree",
            "# bonds: some",
            "unknown bond set",
        ),
    ];
    for (name, from, to, message) in cases {
        let table = GOOD.replace(from, to);
        assert_ne!(table, GOOD, "{name}: the table was not changed");
        let err = rejected(name, &table);
        assert!(err.contains(message), "{name}: {err}");
    }

    // Numbering has to run 1, 2, 3, ... so a dropped row cannot pass silently.
    let gap = GOOD.replace("3\ttorsion", "4\ttorsion");
    assert!(rejected("gap", &gap).contains("numbered from 1 in order"));
    assert!(rejected("empty", "# nothing here\n").contains("no coordinates"));
}

#[test]
fn the_values_must_match_the_definitions() {
    let definitions = scratch_path("width.bat");
    std::fs::write(&definitions, GOOD).unwrap();
    let values = scratch_path("width.tsv");
    std::fs::write(&values, "1 2 3 4\n1 2 3\n").unwrap();
    let err = CoordinateSet::read(&definitions, Some(&values))
        .unwrap_err()
        .to_string();
    assert!(err.contains("frame 2 has 3 values"), "{err}");

    std::fs::write(&values, "# only a comment\n").unwrap();
    let err = CoordinateSet::read(&definitions, Some(&values))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no frames"), "{err}");
}

// ---------------------------------------------------------------------------
// The flags
// ---------------------------------------------------------------------------

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_molentropy"))
        .args(args)
        .output()
        .expect("failed to run CLI")
}

fn total(output: &std::process::Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("Total entropy = ").map(str::to_string))
        .expect("missing total")
}

#[test]
fn the_cli_round_trips_a_coordinate_set() {
    let (top, traj) = chain("cli");
    let prefix = scratch_path("cli_set");
    let prefix = prefix.to_str().unwrap();
    let (top, traj) = (top.to_str().unwrap(), traj.to_str().unwrap());

    let native = run(&[
        top,
        traj,
        "--tree",
        "canonical",
        "--bonds",
        "tree-unconstrained",
        "--estimator",
        "split",
        "--jacobian",
        "--write-coordinates",
        prefix,
    ]);
    let definitions = format!("{prefix}.bat");
    let values = format!("{prefix}.npy");
    assert!(Path::new(&definitions).exists(), "no definitions written");

    let external = run(&[
        &definitions,
        &values,
        "--from-coordinates",
        "--estimator",
        "split",
        "--jacobian",
    ]);
    assert_eq!(
        total(&external),
        total(&native),
        "the external path gave a different total"
    );
    // The Jacobian needs the bond set, which only the definition file carries.
    let jacobian = |output: &std::process::Output| {
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .find_map(|line| line.strip_prefix("Mean ln J = ").map(str::to_string))
            .expect("missing Jacobian")
    };
    assert_eq!(jacobian(&external), jacobian(&native));

    // The text form is the same set, read by the same flag.
    let text = run(&[
        top,
        traj,
        "--tree",
        "canonical",
        "--estimator",
        "nn",
        "--write-coordinates",
        prefix,
        "--coordinate-format",
        "tsv",
    ]);
    let from_text = run(&[
        &definitions,
        &format!("{prefix}.tsv"),
        "--from-coordinates",
        "--estimator",
        "nn",
    ]);
    assert_eq!(total(&from_text), total(&text));
}

#[test]
fn the_coordinate_flags_are_checked() {
    let (top, traj) = chain("cli_rejected");
    let (top, traj) = (top.to_str().unwrap(), traj.to_str().unwrap());
    let prefix = scratch_path("cli_rejected_set");
    let prefix = prefix.to_str().unwrap();
    run(&[
        top,
        traj,
        "--estimator",
        "nn",
        "--write-coordinates",
        prefix,
    ]);
    let definitions = format!("{prefix}.bat");
    let values = format!("{prefix}.npy");

    for (flag, value) in [
        ("--tree", Some("canonical")),
        ("--bonds", Some("tree")),
        ("--tree-bonds", None),
        ("--no-phase-angles", None),
        ("--torsions-only", None),
        ("--write-coordinates", Some("unused")),
        ("--coordinate-format", Some("tsv")),
    ] {
        let mut args = vec![
            definitions.as_str(),
            values.as_str(),
            "--from-coordinates",
            flag,
        ];
        if let Some(value) = value {
            args.push(value);
        }
        let output = run(&args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{flag}: {stderr}");
        assert!(
            stderr.contains("--from-coordinates reads from a file"),
            "{flag}: {stderr}"
        );
    }

    // A set that is not on a tree has no Jacobian, and says so rather than
    // panicking: only the definition file knows its bond set.
    let output = run(&[&definitions, &values, "--from-coordinates", "--jacobian"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("tree bond set"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");

    // And the writing flags only go together.
    let output = run(&[top, traj, "--coordinate-format", "tsv"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("--coordinate-format applies only to --write-coordinates"),
        "{stderr}"
    );
}
