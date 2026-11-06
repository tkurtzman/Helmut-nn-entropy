//! `Coupling::NearestNeighborMie`: the hard-soft cross terms of the
//! second-order expansion, estimated with the nearest-neighbor estimator.
//!
//! These are the terms `Coupling::None` drops. The assertions are exact
//! identities against the crate's own pairwise mutual information, so they
//! pin down *which* pairs are summed and with what sign, rather than checking
//! a number that only this code can produce.

mod common;

use std::f64::consts::TAU;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::Rng;
use common::amber::{
    CHAIN_ATOMS, CHAIN_BONDS, NetcdfOptions, Topology, chain_frames, concat_frames, scratch_path,
    write_netcdf, write_prmtop,
};
use molentropy::split::{
    Coupling, SoftSelection, SplitOptions, estimate_coordinate_split_entropy,
    estimate_split_entropy,
};
use molentropy::{CoordinateMetric, EntropyOptions, estimate_coordinate_mutual_information};

const TORSION: CoordinateMetric = CoordinateMetric::Periodic { period: TAU };
const LINEAR: CoordinateMetric = CoordinateMetric::Linear;

fn options(metrics: &[CoordinateMetric]) -> EntropyOptions<'_> {
    EntropyOptions {
        metrics: Some(metrics),
        ..Default::default()
    }
}

/// Two torsions and two linear coordinates. The first linear coordinate
/// follows the first torsion, so one hard-soft pair carries real dependence
/// and the rest carry none.
fn coupled_system(n: usize) -> (Vec<Vec<f64>>, Vec<CoordinateMetric>) {
    let mut rng = Rng::new(0xc0_55);
    let (mut phi, mut psi, mut x, mut y) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let p = rng.uniform() * TAU;
        phi.push(p);
        psi.push(rng.uniform() * TAU);
        x.push(p.cos() + 0.5 * rng.normal());
        y.push(rng.normal());
    }
    (vec![phi, psi, x, y], vec![TORSION, TORSION, LINEAR, LINEAR])
}

/// Position of the pair `(i, j)`, `i < j`, in the lexicographic order
/// [`estimate_coordinate_mutual_information`] returns.
fn pair_index(dimensions: usize, i: usize, j: usize) -> usize {
    i * (2 * dimensions - i - 1) / 2 + (j - i - 1)
}

// ---------------------------------------------------------------------------
// Exact identities
// ---------------------------------------------------------------------------

#[test]
fn the_cross_terms_are_the_hard_soft_pairs_and_nothing_else() {
    let (data, metrics) = coupled_system(4000);
    let soft = [0usize, 1];
    let hard = [2usize, 3];

    let mut settings = SplitOptions::new(SoftSelection::Indices(&soft));
    settings.coupling = Coupling::NearestNeighborMie;
    let result =
        estimate_split_entropy(&data, data[0].len(), &settings, &options(&metrics)).unwrap();

    let every_pair =
        estimate_coordinate_mutual_information(&data, data[0].len(), &options(&metrics)).unwrap();
    let expected: f64 = hard
        .iter()
        .flat_map(|&h| soft.iter().map(move |&s| (s.min(h), s.max(h))))
        .map(|(i, j)| every_pair[pair_index(data.len(), i, j)])
        .sum();

    // Subtracted, as every pair term in the expansion is.
    assert!(
        (result.coupling + expected).abs() < 1e-9,
        "coupling {} against -{expected}",
        result.coupling
    );
}

#[test]
fn none_drops_exactly_what_nn_estimates() {
    let (data, metrics) = coupled_system(4000);
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0, 1]));

    settings.coupling = Coupling::None;
    let dropped =
        estimate_split_entropy(&data, data[0].len(), &settings, &options(&metrics)).unwrap();
    assert_eq!(dropped.coupling, 0.0);

    settings.coupling = Coupling::NearestNeighborMie;
    let kept = estimate_split_entropy(&data, data[0].len(), &settings, &options(&metrics)).unwrap();
    assert!(kept.coupling < 0.0, "{}", kept.coupling);
    assert!(
        (kept.total - dropped.total - kept.coupling).abs() < 1e-9,
        "the two totals differ by the coupling term alone"
    );
    assert_eq!(kept.soft, dropped.soft);
    assert_eq!(kept.hard, dropped.hard);
}

#[test]
fn the_shares_sum_to_the_coupling_term() {
    let (data, metrics) = coupled_system(3000);
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0, 1]));
    settings.coupling = Coupling::NearestNeighborMie;
    let shares =
        estimate_coordinate_split_entropy(&data, data[0].len(), &settings, &options(&metrics))
            .unwrap();
    let total: f64 = shares.coupling.iter().sum();
    assert!(
        (total - shares.split.coupling).abs() < 1e-9,
        "{total} against {}",
        shares.split.coupling
    );
    // The coupled pair (torsion 0, linear 2) carries most of it, and each of
    // the two takes half of its own pair term.
    assert!(
        shares.coupling[0] < shares.coupling[1],
        "{:?}",
        shares.coupling
    );
    assert!(
        shares.coupling[2] < shares.coupling[3],
        "{:?}",
        shares.coupling
    );
}

// ---------------------------------------------------------------------------
// Behaviour
// ---------------------------------------------------------------------------

#[test]
fn independent_sets_give_almost_no_cross_terms() {
    // Only the independent coordinates, so every cross pair is noise.
    let (data, metrics) = coupled_system(4000);
    let independent = vec![data[1].clone(), data[3].clone()];
    let metrics = vec![metrics[1], metrics[3]];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    settings.coupling = Coupling::NearestNeighborMie;
    let result = estimate_split_entropy(
        &independent,
        independent[0].len(),
        &settings,
        &options(&metrics),
    )
    .unwrap();
    assert!(result.coupling.abs() < 0.05, "{}", result.coupling);
}

#[test]
fn an_empty_set_leaves_no_cross_terms() {
    let (data, metrics) = coupled_system(1000);
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[]));
    settings.coupling = Coupling::NearestNeighborMie;
    let result =
        estimate_split_entropy(&data, data[0].len(), &settings, &options(&metrics)).unwrap();
    assert_eq!(result.coupling, 0.0);
}

// ---------------------------------------------------------------------------
// The flag
// ---------------------------------------------------------------------------

fn chain(name: &str) -> (PathBuf, PathBuf) {
    let mut topology = Topology::default();
    topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
    let top = scratch_path(&format!("cross_{name}.parm7"));
    let traj = scratch_path(&format!("cross_{name}.nc"));
    write_prmtop(&top, &topology);
    let mut rng = Rng::new(0x5b17);
    let frames = chain_frames(&mut rng, 80, [0.0, 0.0, 0.0]);
    write_netcdf(&traj, &concat_frames(&[&frames]), NetcdfOptions::default());
    (top, traj)
}

fn coupling_term(top: &Path, traj: &Path, mode: &str) -> f64 {
    let output = Command::new(env!("CARGO_BIN_EXE_molentropy"))
        .args([top.to_str().unwrap(), traj.to_str().unwrap()])
        .args(["--estimator", "split", "--coupling", mode])
        .output()
        .expect("failed to run CLI");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("Coupling term = ").map(str::to_string))
        .expect("missing coupling term")
        .parse()
        .unwrap()
}

#[test]
fn the_nn_coupling_reaches_the_program() {
    let (top, traj) = chain("cli");
    assert_eq!(coupling_term(&top, &traj, "none"), 0.0);
    assert!(coupling_term(&top, &traj, "nn") != 0.0);

    let output = Command::new(env!("CARGO_BIN_EXE_molentropy"))
        .args([top.to_str().unwrap(), traj.to_str().unwrap()])
        .args(["--estimator", "split", "--coupling", "cross"])
        .output()
        .expect("failed to run CLI");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("gaussian, states, nn, none"), "{stderr}");
}
