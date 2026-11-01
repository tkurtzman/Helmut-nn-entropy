//! Reproduces the butane end-state comparison in the user guide,
//! <https://helmutcarter.github.io/molentropy/validation.html>.
//!
//! ```text
//! cargo run --release --example butane_validation -- <dir> <reference> [<dir> <reference> ...]
//! ```
//!
//! Each `<dir>` holds `0.0.top`, `0.0.nc`, `1.0.top` and `1.0.nc`, and
//! `<reference>` is that replicate's thermodynamic `S0 - S1` in nats. Prints
//! every estimate of `S0 - S1`, with the change in the mean log BAT Jacobian
//! included, and its error against the reference.

use std::path::Path;

use molentropy::bat_library::{CoordinateOptions, InternalCoordinates};
use molentropy::quasi_harmonic::{
    MarginalEstimator, PeriodicCorrelation, QuasiHarmonicOptions, estimate_quasi_harmonic_entropy,
};
use molentropy::split::{
    Coupling, HardEstimator, SoftEstimator, SoftFeatures, SoftSelection, SplitOptions,
    StateDefinition, StateOptions, estimate_split_entropy,
};
use molentropy::{CoordinateMetric, EntropyOptions, calculate_entropy};

type Error = Box<dyn std::error::Error>;

/// One end state's coordinates, `data[coordinate][frame]`.
struct State {
    data: Vec<Vec<f64>>,
    metrics: Vec<CoordinateMetric>,
    atoms: Vec<Vec<usize>>,
    /// The split estimator's topological soft set, and its methyl rotors.
    soft: Vec<usize>,
    symmetric: Vec<usize>,
}

fn load(dir: &Path, state: &str, phase_angles: bool) -> Result<State, Error> {
    let options = CoordinateOptions {
        phase_angles,
        ..Default::default()
    };
    let mut internal =
        InternalCoordinates::with_options(&dir.join(format!("{state}.top")), options)?;
    internal.calculate_internal_coords(&dir.join(format!("{state}.nc")), usize::MAX, false)?;
    let data = internal.columns(0, 1)?;
    Ok(State {
        data,
        metrics: internal.coordinate_metrics(),
        atoms: internal.coordinate_atoms(),
        soft: internal.soft_torsions(),
        symmetric: internal.symmetric_rotor_torsions(),
    })
}

/// Mean over frames of sum 2 ln b + sum ln sin(theta) over the bonds and angles
/// that are coordinates. Constrained bonds are not coordinates; they add a
/// constant that cancels between the end states.
fn mean_log_jacobian(state: &State) -> f64 {
    let frames = state.data[0].len() as f64;
    state
        .atoms
        .iter()
        .zip(&state.data)
        .map(|(atoms, values)| match atoms.len() {
            2 => values.iter().map(|b| 2.0 * b.ln()).sum::<f64>() / frames,
            3 => values.iter().map(|theta| theta.sin().ln()).sum::<f64>() / frames,
            _ => 0.0,
        })
        .sum()
}

/// Every estimate of one state's entropy, in table order.
fn estimates(state: &State) -> Result<Vec<f64>, String> {
    let frames = state.data[0].len();
    let options = EntropyOptions {
        metrics: Some(&state.metrics),
        ..Default::default()
    };
    let quasi_harmonic = |marginals, periodic| {
        let settings = QuasiHarmonicOptions {
            marginals,
            periodic,
            ..Default::default()
        };
        estimate_quasi_harmonic_entropy(&state.data, frames, &settings, &options)
    };
    let complex = quasi_harmonic(
        MarginalEstimator::NearestNeighbor,
        PeriodicCorrelation::Complex,
    )?;
    let centered = quasi_harmonic(
        MarginalEstimator::NearestNeighbor,
        PeriodicCorrelation::Centered,
    )?;
    let histogram = quasi_harmonic(
        MarginalEstimator::Histogram { bins: 100 },
        PeriodicCorrelation::Complex,
    )?;
    Ok(vec![
        complex.boltzmann_quasi_harmonic,
        centered.boltzmann_quasi_harmonic,
        histogram.boltzmann_quasi_harmonic,
        calculate_entropy(&state.data, frames, 2, &options)?,
        complex.quasi_harmonic,
    ])
}

/// The split estimator's default and three variants, with the topological soft
/// set. Split needs phase angles, so these run only with them.
fn split_estimates(state: &State) -> Result<Vec<f64>, String> {
    let frames = state.data[0].len();
    let options = EntropyOptions {
        metrics: Some(&state.metrics),
        ..Default::default()
    };
    let default = SplitOptions::new(SoftSelection::Indices(&state.soft));
    let mut cos_sin = default;
    cos_sin.coupling = Coupling::Gaussian {
        features: SoftFeatures::CosSin,
    };
    let mut per_state = default;
    per_state.coupling = Coupling::PerState(StateOptions {
        definition: StateDefinition::Detected {
            defining: None,
            exclude: &state.symmetric,
        },
        ..Default::default()
    });
    let mut joint_bqh = default;
    joint_bqh.soft_estimator = SoftEstimator::JointNearestNeighbor;
    joint_bqh.hard_estimator = HardEstimator::BoltzmannQuasiHarmonic;
    joint_bqh.coupling = Coupling::None;
    [default, cos_sin, per_state, joint_bqh]
        .iter()
        .map(|split| estimate_split_entropy(&state.data, frames, split, &options).map(|r| r.total))
        .collect()
}

const SPLIT_ESTIMATORS: [&str; 4] = [
    "split",
    "split, cos/sin coupling",
    "split, per-state coupling",
    "split, joint soft, BQH hard",
];

const ESTIMATORS: [&str; 5] = [
    "BQH",
    "BQH, centered torsions",
    "BQH, 100-bin histogram",
    "NN MIE-2",
    "QH",
];

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err("usage: butane_validation <dir> <reference> [<dir> <reference> ...]".into());
    }
    for pair in args.chunks(2) {
        let dir = Path::new(&pair[0]);
        let reference: f64 = pair[1].parse()?;
        println!("{} (reference S0 - S1 = {reference})", dir.display());
        for phase_angles in [true, false] {
            let (s0, s1) = (
                load(dir, "0.0", phase_angles)?,
                load(dir, "1.0", phase_angles)?,
            );
            let jacobian = mean_log_jacobian(&s0) - mean_log_jacobian(&s1);
            let coordinates = if phase_angles {
                "phase angles"
            } else {
                "raw torsions"
            };
            let (mut names, mut e0, mut e1) =
                (ESTIMATORS.to_vec(), estimates(&s0)?, estimates(&s1)?);
            if phase_angles {
                names.extend(SPLIT_ESTIMATORS);
                e0.extend(split_estimates(&s0)?);
                e1.extend(split_estimates(&s1)?);
            }
            for (name, (e0, e1)) in names.iter().zip(e0.into_iter().zip(e1)) {
                let estimate = e0 - e1 + jacobian;
                println!(
                    "  {coordinates:<13} {name:<28} {estimate:8.3}  error {:+7.3}",
                    estimate - reference
                );
            }
        }
    }
    Ok(())
}
