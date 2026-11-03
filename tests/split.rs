//! The split (soft/hard) estimator.
//!
//! Expected values come from the estimators the split one is assembled from
//! (exact identities), from closed forms, or from a synthetic system whose
//! entropy is known: a torsion with well-separated wells, and a hard
//! coordinate whose Gaussian spread depends on the well.

mod common;

use std::f64::consts::{E, PI, TAU};

use common::Rng;
use molentropy::quasi_harmonic::{
    PeriodicCorrelation, QuasiHarmonicOptions, estimate_quasi_harmonic_entropy,
};
use molentropy::split::{
    Coupling, DEFAULT_HARMONICS, HardEstimator, SoftEstimator, SoftFeatures, SoftSelection,
    SplitOptions, StateDefinition, StateOptions, estimate_split_entropy,
};
use molentropy::{
    CoordinateMetric, EntropyOptions, MAX_JOINT_DIMENSION, calculate_entropy,
    estimate_joint_entropy, estimate_mist_entropy, estimate_network_entropy,
};

const TORSION: CoordinateMetric = CoordinateMetric::Periodic { period: TAU };
const LINEAR: CoordinateMetric = CoordinateMetric::Linear;

/// Wells of the soft torsion: center, weight, and the standard deviation of
/// the hard coordinate in that well.
const WELLS: [(f64, f64, f64); 3] = [
    (0.0, 0.5, 0.1),
    (2.0 * PI / 3.0, 0.3, 0.2),
    (4.0 * PI / 3.0, 0.2, 0.4),
];
const WELL_WIDTH: f64 = 0.2;

/// A torsion drawn from `WELLS`, a hard coordinate whose spread depends on the
/// well, the well of each frame, and an independent narrow torsion.
fn well_system(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<u32>) {
    let (mut phi, mut x, mut wells, mut narrow) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let mut pick = rng.uniform();
        let well = WELLS
            .iter()
            .position(|w| {
                pick -= w.1;
                pick < 0.0
            })
            .unwrap_or(WELLS.len() - 1);
        let (center, _, sd) = WELLS[well];
        phi.push((center + WELL_WIDTH * rng.normal()).rem_euclid(TAU));
        x.push(sd * rng.normal());
        wells.push(well as u32);
        narrow.push((1.0 + 0.15 * rng.normal()).rem_euclid(TAU));
    }
    (vec![phi, x, narrow], wells)
}

/// The exact entropy of `well_system`'s torsion and hard coordinate: with
/// wells this far apart, H(phi) is the wells' own entropy plus the mixing
/// entropy, and x depends on phi only through the well.
fn well_system_entropy() -> f64 {
    WELLS
        .iter()
        .map(|&(_, p, sd)| {
            p * (0.5 * (TAU * E * WELL_WIDTH.powi(2)).ln() + 0.5 * (TAU * E * sd * sd).ln()
                - p.ln())
        })
        .sum()
}

fn options(metrics: &[CoordinateMetric]) -> EntropyOptions<'_> {
    EntropyOptions {
        metrics: Some(metrics),
        ..Default::default()
    }
}

fn split(
    data: &[Vec<f64>],
    metrics: &[CoordinateMetric],
    split: &SplitOptions,
) -> molentropy::split::SplitEntropy {
    estimate_split_entropy(data, data[0].len(), split, &options(metrics)).unwrap()
}

// ---------------------------------------------------------------------------
// Exact identities
// ---------------------------------------------------------------------------

#[test]
fn without_soft_coordinates_it_is_the_quasi_harmonic_entropy() {
    let mut rng = Rng::new(71);
    let (data, _) = well_system(&mut rng, 3_000);
    let metrics = [TORSION, LINEAR, TORSION];
    // The split estimator's default centers hard periodic coordinates.
    let centered = QuasiHarmonicOptions {
        periodic: PeriodicCorrelation::Centered,
        ..Default::default()
    };
    let qh = estimate_quasi_harmonic_entropy(&data, 3_000, &centered, &options(&metrics)).unwrap();
    for (hard_estimator, expected) in [
        (HardEstimator::QuasiHarmonic, qh.quasi_harmonic),
        (
            HardEstimator::BoltzmannQuasiHarmonic,
            qh.boltzmann_quasi_harmonic,
        ),
    ] {
        let mut settings = SplitOptions::new(SoftSelection::Indices(&[]));
        settings.hard_estimator = hard_estimator;
        let result = split(&data, &metrics, &settings);
        assert_eq!(result.total, expected, "{hard_estimator:?}");
        assert_eq!((result.soft, result.coupling), (0.0, 0.0));
        assert_eq!(result.hard_coordinates, vec![0, 1, 2]);
    }
}

#[test]
fn without_coupling_it_is_the_sum_of_its_parts() {
    let mut rng = Rng::new(72);
    let (data, _) = well_system(&mut rng, 3_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0, 2]));
    settings.coupling = Coupling::None;
    let result = split(&data, &metrics, &settings);

    let soft = calculate_entropy(
        &[data[0].clone(), data[2].clone()],
        3_000,
        2,
        &options(&[TORSION, TORSION]),
    )
    .unwrap();
    let hard = estimate_quasi_harmonic_entropy(
        &[data[1].clone()],
        3_000,
        &QuasiHarmonicOptions::default(),
        &options(&[LINEAR]),
    )
    .unwrap()
    .quasi_harmonic;
    assert_eq!(result.soft, soft);
    assert_eq!(result.hard, hard);
    assert_eq!(result.total, soft + hard);
    assert_eq!(result.soft_coordinates, vec![0, 2]);
    assert_eq!(result.hard_coordinates, vec![1]);
}

#[test]
fn without_hard_coordinates_it_is_the_soft_entropy() {
    let mut rng = Rng::new(73);
    let (data, _) = well_system(&mut rng, 3_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let result = split(
        &data,
        &metrics,
        &SplitOptions::new(SoftSelection::Indices(&[0, 1, 2])),
    );
    assert_eq!(
        result.total,
        calculate_entropy(&data, 3_000, 2, &options(&metrics)).unwrap()
    );
    assert_eq!((result.hard, result.coupling), (0.0, 0.0));

    let mut joint = SplitOptions::new(SoftSelection::Indices(&[0, 1, 2]));
    joint.soft_estimator = SoftEstimator::JointNearestNeighbor;
    assert_eq!(
        split(&data, &metrics, &joint).total,
        estimate_joint_entropy(&data, 3_000, &options(&metrics)).unwrap()
    );

    let mut mist = SplitOptions::new(SoftSelection::Indices(&[0, 1, 2]));
    mist.soft_estimator = SoftEstimator::MutualInformationSpanningTree;
    assert_eq!(
        split(&data, &metrics, &mist).total,
        estimate_mist_entropy(&data, 3_000, &options(&metrics)).unwrap()
    );

    let mut network = SplitOptions::new(SoftSelection::Indices(&[0, 1, 2]));
    network.soft_estimator = SoftEstimator::BayesianNetwork { parents: 2 };
    assert_eq!(
        split(&data, &metrics, &network).total,
        estimate_network_entropy(&data, 3_000, 2, &options(&metrics)).unwrap()
    );
}

#[test]
fn ring_closure_defeats_the_pairwise_mie_but_not_the_joint_estimate() {
    // Three torsions tied by a ring-like closure, phi3 = -(phi1 + phi2) +
    // noise: each pair is independent, so the MIE-2 sees three uniform
    // torsions, but jointly they lie near a surface. As for cyclohexane with
    // detected soft torsions.
    let mut rng = Rng::new(84);
    let n = 20_000;
    let noise = 0.2;
    let phi1 = rng.uniform_vec(n, 0.0, TAU);
    let phi2 = rng.uniform_vec(n, 0.0, TAU);
    let phi3: Vec<f64> = phi1
        .iter()
        .zip(&phi2)
        .map(|(a, b)| (-(a + b) + noise * rng.normal()).rem_euclid(TAU))
        .collect();
    let data = vec![phi1, phi2, phi3];
    let metrics = [TORSION; 3];
    let truth = 2.0 * TAU.ln() + 0.5 * (TAU * E * noise * noise).ln();
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0, 1, 2]));
    let mie = split(&data, &metrics, &settings).total;
    settings.soft_estimator = SoftEstimator::JointNearestNeighbor;
    let joint = split(&data, &metrics, &settings).total;
    assert!((joint - truth).abs() < 0.1, "joint {joint} vs {truth}");
    assert!(mie - truth > 1.5, "MIE-2 {mie} vs {truth}");
}

#[test]
fn for_one_soft_coordinate_the_mie_and_joint_estimates_agree() {
    let mut rng = Rng::new(74);
    let (data, _) = well_system(&mut rng, 3_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let mut mie = SplitOptions::new(SoftSelection::Indices(&[0]));
    mie.coupling = Coupling::None;
    let mut joint = mie;
    joint.soft_estimator = SoftEstimator::JointNearestNeighbor;
    assert_eq!(
        split(&data, &metrics, &mie).soft,
        split(&data, &metrics, &joint).soft
    );
}

// ---------------------------------------------------------------------------
// Coupling
// ---------------------------------------------------------------------------

#[test]
fn gaussian_coupling_recovers_a_linear_dependence_on_cos_and_sin() {
    // x = a cos(phi) + noise with phi uniform: the Gaussian mutual information
    // between (cos phi, sin phi) and x is -1/2 ln(1 - rho^2), with
    // rho^2 = (a^2 / 2) / (a^2 / 2 + sigma^2).
    let mut rng = Rng::new(75);
    let n = 50_000;
    let (a, sigma) = (1.0, 0.5);
    let phi = rng.uniform_vec(n, 0.0, TAU);
    let x: Vec<f64> = phi
        .iter()
        .map(|p| a * p.cos() + sigma * rng.normal())
        .collect();
    let data = vec![phi, x];
    let metrics = [TORSION, LINEAR];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    settings.coupling = Coupling::Gaussian {
        features: SoftFeatures::CosSin,
    };
    let rho_squared = (a * a / 2.0) / (a * a / 2.0 + sigma * sigma);
    let coupling = split(&data, &metrics, &settings).coupling;
    assert!(
        (coupling - 0.5 * (1.0 - rho_squared).ln()).abs() < 0.01,
        "{coupling}"
    );

    settings.coupling = Coupling::Gaussian {
        features: SoftFeatures::Complex,
    };
    assert!(split(&data, &metrics, &settings).coupling < 0.0);
}

#[test]
fn cos_sin_features_catch_a_trans_gauche_dependence_that_complex_ones_miss() {
    // A three-well torsion and a hard coordinate whose mean depends only on
    // trans against gauche, that is on cos(phi), as a bond angle does in a
    // chain. The single complex entry averages cos and sin, so it misses most
    // of the dependence; this is why cos/sin is the default.
    let mut rng = Rng::new(81);
    let n = 50_000;
    let wells = [(PI, 0.6, -1.0), (PI / 3.0, 0.2, 0.5), (-PI / 3.0, 0.2, 0.5)];
    let (width, sd) = (0.2, 0.3);
    let (mut phi, mut x) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let mut pick = rng.uniform();
        let &(center, _, mean) = wells
            .iter()
            .find(|w| {
                pick -= w.1;
                pick < 0.0
            })
            .unwrap_or(&wells[2]);
        phi.push((center + width * rng.normal()).rem_euclid(TAU));
        x.push(mean + sd * rng.normal());
    }
    let data = vec![phi, x];
    let metrics = [TORSION, LINEAR];
    // H(phi) for well-separated wells, plus H(x | phi) = H(x | well).
    let truth: f64 = wells
        .iter()
        .map(|&(_, p, _)| p * (0.5 * (TAU * E * width * width).ln() - p.ln()))
        .sum::<f64>()
        + 0.5 * (TAU * E * sd * sd).ln();

    let total = |coupling| {
        let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
        settings.coupling = coupling;
        split(&data, &metrics, &settings).total
    };
    let cos_sin = total(Coupling::Gaussian {
        features: SoftFeatures::CosSin,
    });
    let complex = total(Coupling::Gaussian {
        features: SoftFeatures::Complex,
    });
    let independent = total(Coupling::None);
    assert!((cos_sin - truth).abs() < 0.15, "{cos_sin} vs {truth}");
    assert!(complex - truth > 0.4, "{complex} vs {truth}");
    assert!(independent > complex, "{independent} vs {complex}");
}

#[test]
fn the_default_coupling_takes_three_harmonics_of_each_soft_torsion() {
    let settings = SplitOptions::new(SoftSelection::Indices(&[]));
    assert_eq!(DEFAULT_HARMONICS, 3);
    assert_eq!(
        settings.coupling,
        Coupling::Gaussian {
            features: SoftFeatures::Harmonics(3)
        }
    );
}

#[test]
fn one_harmonic_is_cos_and_sin() {
    let mut rng = Rng::new(84);
    let (data, _) = well_system(&mut rng, 3_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let coupling = |features| {
        let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
        settings.coupling = Coupling::Gaussian { features };
        split(&data, &metrics, &settings).coupling
    };
    assert_eq!(
        coupling(SoftFeatures::Harmonics(1)),
        coupling(SoftFeatures::CosSin)
    );
}

#[test]
fn higher_harmonics_catch_a_threefold_dependence_that_cos_and_sin_miss() {
    // A free rotor and a hard coordinate that follows cos(3 phi), as a bond
    // angle strained by eclipsing does. cos(phi) and sin(phi) are uncorrelated
    // with it; the third harmonic recovers the Gaussian mutual information
    // -1/2 ln(1 - rho^2) exactly.
    let mut rng = Rng::new(85);
    let n = 50_000;
    let (a, sigma) = (1.0, 0.5);
    let phi = rng.uniform_vec(n, 0.0, TAU);
    let x: Vec<f64> = phi
        .iter()
        .map(|p| a * (3.0 * p).cos() + sigma * rng.normal())
        .collect();
    let data = vec![phi, x];
    let metrics = [TORSION, LINEAR];
    let coupling = |features| {
        let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
        settings.coupling = Coupling::Gaussian { features };
        split(&data, &metrics, &settings).coupling
    };
    let rho_squared = (a * a / 2.0) / (a * a / 2.0 + sigma * sigma);
    let three = coupling(SoftFeatures::Harmonics(3));
    assert!(
        (three - 0.5 * (1.0 - rho_squared).ln()).abs() < 0.01,
        "{three}"
    );
    assert!(coupling(SoftFeatures::CosSin).abs() < 0.001);
    assert!(coupling(SoftFeatures::Harmonics(2)).abs() < 0.001);
}

#[test]
fn independent_sets_have_almost_no_gaussian_coupling() {
    let mut rng = Rng::new(76);
    let n = 50_000;
    let data = vec![
        rng.uniform_vec(n, 0.0, TAU),
        rng.normal_vec(n, 0.0, 1.0),
        rng.normal_vec(n, 0.0, 1.0),
    ];
    let metrics = [TORSION, LINEAR, LINEAR];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    settings.coupling = Coupling::Gaussian {
        features: SoftFeatures::Complex,
    };
    let coupling = split(&data, &metrics, &settings).coupling;
    assert!(coupling <= 0.0 && coupling > -0.005, "{coupling}");
}

#[test]
fn per_state_coupling_recovers_a_well_dependent_spread() {
    let mut rng = Rng::new(77);
    let (data, _) = well_system(&mut rng, 60_000);
    let data = vec![data[0].clone(), data[1].clone()];
    let metrics = [TORSION, LINEAR];
    let truth = well_system_entropy();

    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    settings.coupling = Coupling::PerState(StateOptions::default());
    let per_state = split(&data, &metrics, &settings);
    assert!(
        (per_state.total - truth).abs() < 0.03,
        "{} vs {truth}",
        per_state.total
    );
    let states = per_state.states.unwrap();
    assert_eq!(states.defining, vec![0]);
    assert_eq!(states.states.len(), 3);
    assert_eq!(states.dropped_frames, 0);
    assert!(per_state.coupling < -0.1);

    // Without conditioning, the hard coordinate is a mixture of Gaussians of
    // different widths, so the entropy comes out too high.
    let mut independent = SplitOptions::new(SoftSelection::Indices(&[0]));
    independent.coupling = Coupling::None;
    assert!(split(&data, &metrics, &independent).total > truth + 0.1);
}

#[test]
fn per_state_coupling_beats_gaussian_coupling_when_the_spread_depends_on_the_state() {
    // As for biphenyl: a few well-populated states whose hard coordinate
    // differs in spread, not in mean, which no second moment of (cos, sin)
    // can see.
    let mut rng = Rng::new(82);
    let (data, _) = well_system(&mut rng, 60_000);
    let data = vec![data[0].clone(), data[1].clone()];
    let metrics = [TORSION, LINEAR];
    let truth = well_system_entropy();
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    let gaussian = split(&data, &metrics, &settings).total;
    settings.coupling = Coupling::PerState(StateOptions::default());
    let per_state = split(&data, &metrics, &settings).total;
    assert!(
        (gaussian - truth).abs() > 0.1 && (per_state - truth).abs() < 0.03,
        "Gaussian {gaussian}, per-state {per_state}, truth {truth}"
    );
}

/// `torsions` independent three-well torsions and one hard coordinate.
fn many_wells(rng: &mut Rng, torsions: usize, n: usize) -> Vec<Vec<f64>> {
    let mut data: Vec<Vec<f64>> = (0..torsions)
        .map(|_| {
            (0..n)
                .map(|_| {
                    let well = (3.0 * rng.uniform()).floor();
                    (well * TAU / 3.0 + 0.2 * rng.normal()).rem_euclid(TAU)
                })
                .collect()
        })
        .collect();
    data.push(rng.normal_vec(n, 0.0, 1.0));
    data
}

#[test]
fn per_state_coupling_refuses_too_many_states() {
    // As for decane and longer chains: three wells per torsion multiply into
    // more states than the frames can fill.
    let mut rng = Rng::new(83);
    let data = many_wells(&mut rng, 5, 6_000);
    let mut metrics = vec![TORSION; 5];
    metrics.push(LINEAR);
    let soft: Vec<usize> = (0..5).collect();
    let mut settings = SplitOptions::new(SoftSelection::Indices(&soft));
    settings.coupling = Coupling::PerState(StateOptions::default());
    let err = estimate_split_entropy(&data, 6_000, &settings, &options(&metrics)).unwrap_err();
    assert!(
        err.contains("no conformational state has enough frames"),
        "{err}"
    );

    let data = many_wells(&mut rng, 6, 40_000);
    let mut metrics = vec![TORSION; 6];
    metrics.push(LINEAR);
    let soft: Vec<usize> = (0..6).collect();
    settings.soft = SoftSelection::Indices(&soft);
    settings.coupling = Coupling::PerState(StateOptions {
        min_frames: 20,
        ..Default::default()
    });
    let err = estimate_split_entropy(&data, 40_000, &settings, &options(&metrics)).unwrap_err();
    assert!(err.contains("729 conformational states"), "{err}");
    assert!(err.contains("maximum of 256"), "{err}");

    // Gaussian coupling has no states to run out of.
    settings.coupling = Coupling::Gaussian {
        features: SoftFeatures::CosSin,
    };
    assert!(
        estimate_split_entropy(&data, 40_000, &settings, &options(&metrics))
            .unwrap()
            .total
            .is_finite()
    );
}

#[test]
fn a_sparse_state_is_dropped_and_the_rest_renormalized() {
    let mut rng = Rng::new(78);
    let (data, mut wells) = well_system(&mut rng, 20_000);
    let data = vec![data[0].clone(), data[1].clone()];
    // Make 50 frames a state of their own.
    for well in wells.iter_mut().take(50) {
        *well = 7;
    }
    let metrics = [TORSION, LINEAR];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0]));
    settings.coupling = Coupling::PerState(StateOptions {
        definition: StateDefinition::Labels(&wells),
        ..Default::default()
    });
    let result = split(&data, &metrics, &settings);
    let states = result.states.unwrap();
    assert_eq!(states.dropped_frames, 50);
    assert!((states.dropped_population - 50.0 / 20_000.0).abs() < 1e-12);

    // The conditional entropy is the population-weighted mean of the kept
    // states' QH entropies, computed here from the frames directly.
    let (mut weighted, mut used) = (0.0, 0.0);
    for state in states.states.iter().filter(|s| s.hard_entropy.is_some()) {
        let frames: Vec<f64> = data[1]
            .iter()
            .zip(&wells)
            .filter(|&(_, &w)| w == state.basins[0])
            .map(|(&x, _)| x)
            .collect();
        let entropy = estimate_quasi_harmonic_entropy(
            std::slice::from_ref(&frames),
            frames.len(),
            &QuasiHarmonicOptions::default(),
            &options(&[LINEAR]),
        )
        .unwrap()
        .quasi_harmonic;
        assert_eq!(state.hard_entropy, Some(entropy));
        weighted += state.population * entropy;
        used += state.population;
    }
    assert!((result.coupling - (weighted / used - result.hard)).abs() < 1e-12);
}

#[test]
fn symmetric_or_single_basin_torsions_define_no_states() {
    let mut rng = Rng::new(79);
    let (data, _) = well_system(&mut rng, 20_000);
    let metrics = [TORSION, LINEAR, TORSION];
    // Excluding the multi-basin torsion leaves only the narrow one, which has
    // one basin: a single state, so no coupling.
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[0, 2]));
    settings.coupling = Coupling::PerState(StateOptions {
        definition: StateDefinition::Detected {
            defining: None,
            exclude: &[0],
        },
        ..Default::default()
    });
    let result = split(&data, &metrics, &settings);
    assert_eq!(result.coupling, 0.0);
    let states = result.states.unwrap();
    assert!(states.defining.is_empty());
    assert_eq!(states.states.len(), 1);
}

// ---------------------------------------------------------------------------
// Detecting the soft set
// ---------------------------------------------------------------------------

#[test]
fn detection_picks_the_multi_basin_torsion() {
    let mut rng = Rng::new(80);
    let (data, _) = well_system(&mut rng, 20_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let result = split(
        &data,
        &metrics,
        &SplitOptions::new(SoftSelection::Detect { candidates: None }),
    );
    assert_eq!(result.soft_coordinates, vec![0]);
    assert_eq!(result.hard_coordinates, vec![1, 2]);
    assert_eq!(result.basin_counts, vec![Some(3), None, Some(1)]);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
fn invalid_settings_are_refused() {
    let mut rng = Rng::new(81);
    let (data, wells) = well_system(&mut rng, 2_000);
    let metrics = [TORSION, LINEAR, TORSION];
    let error = |settings: &SplitOptions, theiler_window: usize| {
        let options = EntropyOptions {
            metrics: Some(&metrics),
            theiler_window,
            ..Default::default()
        };
        estimate_split_entropy(&data, 2_000, settings, &options).unwrap_err()
    };

    assert!(error(&SplitOptions::new(SoftSelection::Indices(&[3])), 0).contains("out of range"));
    assert!(error(&SplitOptions::new(SoftSelection::Indices(&[0, 0])), 0).contains("repeated"));
    let linear_candidate = SplitOptions::new(SoftSelection::Detect {
        candidates: Some(&[1]),
    });
    assert!(error(&linear_candidate, 0).contains("not periodic"));

    let mut short_labels = SplitOptions::new(SoftSelection::Indices(&[0]));
    short_labels.coupling = Coupling::PerState(StateOptions {
        definition: StateDefinition::Labels(&wells[..10]),
        ..Default::default()
    });
    assert!(error(&short_labels, 0).contains("state labels"));

    let mut windowed = SplitOptions::new(SoftSelection::Indices(&[0]));
    windowed.hard_estimator = HardEstimator::BoltzmannQuasiHarmonic;
    windowed.coupling = Coupling::PerState(StateOptions::default());
    assert!(error(&windowed, 1).contains("Theiler window"));

    let mut no_harmonics = SplitOptions::new(SoftSelection::Indices(&[0]));
    no_harmonics.coupling = Coupling::Gaussian {
        features: SoftFeatures::Harmonics(0),
    };
    assert!(error(&no_harmonics, 0).contains("at least one harmonic"));

    let mut too_many = SplitOptions::new(SoftSelection::Indices(&[0]));
    too_many.coupling = Coupling::PerState(StateOptions {
        min_frames: 1,
        max_states: 2,
        ..Default::default()
    });
    assert!(error(&too_many, 0).contains("more than the maximum"));

    // The joint estimate has a dimension limit.
    let wide: Vec<Vec<f64>> = (0..=MAX_JOINT_DIMENSION)
        .map(|_| rng.uniform_vec(200, 0.0, TAU))
        .collect();
    let wide_metrics = vec![TORSION; wide.len()];
    let all: Vec<usize> = (0..wide.len()).collect();
    let mut joint = SplitOptions::new(SoftSelection::Indices(&all));
    joint.soft_estimator = SoftEstimator::JointNearestNeighbor;
    let err = estimate_split_entropy(&wide, 200, &joint, &options(&wide_metrics)).unwrap_err();
    assert!(
        err.contains("use the mutual information expansion"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// Pairwise Gaussian hard entropy
// ---------------------------------------------------------------------------

/// `n` frames of `dimensions` Gaussian coordinates with unit variance and
/// correlation `rho` between every pair.
fn equicorrelated(rng: &mut Rng, dimensions: usize, rho: f64, n: usize) -> Vec<Vec<f64>> {
    let (shared, own) = (rho.sqrt(), (1.0 - rho).sqrt());
    let mut columns = vec![Vec::with_capacity(n); dimensions];
    for _ in 0..n {
        let common = rng.normal();
        for column in &mut columns {
            column.push(shared * common + own * rng.normal());
        }
    }
    columns
}

fn pairwise_hard(data: &[Vec<f64>]) -> f64 {
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[]));
    settings.hard_estimator = HardEstimator::QuasiHarmonicMie2;
    split(data, &vec![LINEAR; data.len()], &settings).total
}

#[test]
fn pairwise_hard_entropy_is_the_gaussian_mie2_of_the_sample() {
    let mut rng = Rng::new(91);
    let data = equicorrelated(&mut rng, 4, 0.5, 5_000);
    let n = data[0].len() as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
    let moment = |a: &[f64], b: &[f64]| {
        let (ma, mb) = (mean(a), mean(b));
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>()
            / n
    };
    let mut expected = 0.0;
    for i in 0..data.len() {
        expected += 0.5 * (TAU * E * moment(&data[i], &data[i])).ln();
        for j in (i + 1)..data.len() {
            let r2 = moment(&data[i], &data[j]).powi(2)
                / (moment(&data[i], &data[i]) * moment(&data[j], &data[j]));
            expected += 0.5 * (1.0 - r2).ln();
        }
    }
    let got = pairwise_hard(&data);
    assert!((got - expected).abs() < 1e-10, "{got} vs {expected}");
}

#[test]
fn for_two_hard_coordinates_pairwise_and_full_quasi_harmonic_agree() {
    let mut rng = Rng::new(92);
    let data = equicorrelated(&mut rng, 2, 0.7, 3_000);
    let metrics = [LINEAR, LINEAR];
    let full = split(
        &data,
        &metrics,
        &SplitOptions::new(SoftSelection::Indices(&[])),
    );
    let pairwise = pairwise_hard(&data);
    assert!(
        (pairwise - full.total).abs() < 1e-12,
        "{pairwise} vs {}",
        full.total
    );
}

#[test]
fn pairwise_hard_entropy_overcounts_three_mutually_correlated_coordinates() {
    // With every pair correlated by rho, |C| = (1 - rho)^2 (1 + 2 rho), while
    // the pairwise sum subtracts -1/2 ln(1 - rho^2) three times.
    let rho: f64 = 0.6;
    let mut rng = Rng::new(93);
    let data = equicorrelated(&mut rng, 3, rho, 50_000);
    let full = split(
        &data,
        &[LINEAR; 3],
        &SplitOptions::new(SoftSelection::Indices(&[])),
    )
    .total;
    let expected =
        1.5 * (1.0 - rho * rho).ln() - 0.5 * ((1.0 - rho).powi(2) * (1.0 + 2.0 * rho)).ln();
    let difference = pairwise_hard(&data) - full;
    assert!(expected < -0.1);
    assert!(
        (difference - expected).abs() < 0.01,
        "{difference} vs {expected}"
    );
}

fn first_order_hard(data: &[Vec<f64>]) -> f64 {
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[]));
    settings.hard_estimator = HardEstimator::QuasiHarmonicMie1;
    split(data, &vec![LINEAR; data.len()], &settings).total
}

#[test]
fn first_order_hard_entropy_is_the_sum_of_gaussian_entropies() {
    let mut rng = Rng::new(95);
    let data = equicorrelated(&mut rng, 4, 0.5, 5_000);
    let n = data[0].len() as f64;
    let expected: f64 = data
        .iter()
        .map(|v| {
            let mean = v.iter().sum::<f64>() / n;
            let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            0.5 * (TAU * E * variance).ln()
        })
        .sum();
    let got = first_order_hard(&data);
    assert!((got - expected).abs() < 1e-10, "{got} vs {expected}");
}

#[test]
fn first_order_hard_entropy_exceeds_full_quasi_harmonic_by_the_correlation_term() {
    // For equicorrelated coordinates, |C| = (1 - rho)^(d - 1) (1 + (d - 1) rho).
    let (rho, dimensions): (f64, usize) = (0.5, 4);
    let mut rng = Rng::new(96);
    let data = equicorrelated(&mut rng, dimensions, rho, 50_000);
    let full = split(
        &data,
        &[LINEAR; 4],
        &SplitOptions::new(SoftSelection::Indices(&[])),
    )
    .total;
    let d = dimensions as f64;
    let expected = -0.5 * ((1.0 - rho).powf(d - 1.0) * (1.0 + (d - 1.0) * rho)).ln();
    let difference = first_order_hard(&data) - full;
    assert!(difference > 0.0);
    assert!(
        (difference - expected).abs() < 0.01,
        "{difference} vs {expected}"
    );
}

#[test]
fn pairwise_hard_entropy_refuses_a_perfectly_correlated_pair() {
    let mut rng = Rng::new(94);
    let x = rng.uniform_vec(500, 0.0, 1.0);
    let data = vec![
        x.clone(),
        x.iter().map(|v| 2.0 * v + 1.0).collect(),
        rng.uniform_vec(500, 0.0, 1.0),
    ];
    let mut settings = SplitOptions::new(SoftSelection::Indices(&[]));
    settings.hard_estimator = HardEstimator::QuasiHarmonicMie2;
    let err = estimate_split_entropy(&data, 500, &settings, &options(&[LINEAR; 3])).unwrap_err();
    assert!(
        err.contains("coordinates 0 and 1 are perfectly correlated"),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

mod cli {
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use super::common::Rng;
    use super::common::amber::{
        CHAIN_ATOMS, CHAIN_BONDS, NetcdfOptions, Topology, chain_frames, concat_frames,
        scratch_path, write_netcdf, write_prmtop,
    };
    use molentropy::EntropyOptions;
    use molentropy::bat_library::InternalCoordinates;
    use molentropy::split::{
        Coupling, HardEstimator, LARGE_SOFT_SET, SoftEstimator, SoftFeatures, SoftSelection,
        SplitOptions, StateDefinition, StateOptions, estimate_coordinate_split_entropy,
        estimate_split_entropy,
    };

    /// A chain molecule (three rotatable bonds) with more frames than
    /// coordinates.
    fn chain(name: &str) -> (PathBuf, PathBuf) {
        let mut topology = Topology::default();
        topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
        let top = scratch_path(&format!("split_{name}.parm7"));
        let traj = scratch_path(&format!("split_{name}.nc"));
        write_prmtop(&top, &topology);
        let mut rng = Rng::new(0x5b17);
        let frames = chain_frames(&mut rng, 80, [0.0, 0.0, 0.0]);
        write_netcdf(&traj, &concat_frames(&[&frames]), NetcdfOptions::default());
        (top, traj)
    }

    fn run(top: &Path, traj: &Path, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_molentropy"))
            .args([top.to_str().unwrap(), traj.to_str().unwrap()])
            .args(args)
            .output()
            .expect("failed to run CLI")
    }

    fn lines(output: &Output) -> Vec<String> {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn value(lines: &[String], name: &str) -> String {
        lines
            .iter()
            .find_map(|line| line.strip_prefix(&format!("{name} = ")))
            .unwrap_or_else(|| panic!("no {name} line in {lines:?}"))
            .to_string()
    }

    fn library(
        top: &Path,
        traj: &Path,
        configure: impl Fn(&mut SplitOptions, &InternalCoordinates),
    ) -> molentropy::split::SplitEntropy {
        let mut internal = InternalCoordinates::new(top).unwrap();
        internal
            .calculate_internal_coords(traj, usize::MAX, false)
            .unwrap();
        let data = internal.columns(0, 1).unwrap();
        let metrics = internal.coordinate_metrics();
        let soft = internal.soft_torsions();
        let mut settings = SplitOptions::new(SoftSelection::Indices(&soft));
        configure(&mut settings, &internal);
        let options = EntropyOptions {
            metrics: Some(&metrics),
            ..Default::default()
        };
        estimate_split_entropy(&data, data[0].len(), &settings, &options).unwrap()
    }

    #[test]
    fn default_split_output_matches_the_library() {
        let (top, traj) = chain("cli_default");
        let lines = lines(&run(&top, &traj, &["--estimator", "split"]));
        let expected = library(&top, &traj, |_, _| {});
        let numbers: Vec<String> = expected
            .soft_coordinates
            .iter()
            .map(|i| (i + 1).to_string())
            .collect();
        assert_eq!(value(&lines, "Soft coordinates"), numbers.join(","));
        assert_eq!(
            value(&lines, "Soft entropy").parse::<f64>().unwrap(),
            expected.soft
        );
        assert_eq!(
            value(&lines, "Hard entropy").parse::<f64>().unwrap(),
            expected.hard
        );
        assert_eq!(
            value(&lines, "Coupling term").parse::<f64>().unwrap(),
            expected.coupling
        );
        assert_eq!(
            lines.last().unwrap(),
            &format!("Total entropy = {}", expected.total)
        );
        assert_eq!(numbers.len(), 3);
    }

    #[test]
    fn first_order_hard_estimator_output_matches_the_library() {
        let (top, traj) = chain("cli_qh_mie1");
        let lines = lines(&run(
            &top,
            &traj,
            &["--estimator", "split", "--hard-estimator", "qh-mie1"],
        ));
        let expected = library(&top, &traj, |settings, _| {
            settings.hard_estimator = HardEstimator::QuasiHarmonicMie1;
        });
        let full = library(&top, &traj, |_, _| {});
        assert_eq!(
            value(&lines, "Hard entropy").parse::<f64>().unwrap(),
            expected.hard
        );
        assert_eq!(
            lines.last().unwrap(),
            &format!("Total entropy = {}", expected.total)
        );
        // Only the hard term changes, and dropping the correlation term raises it.
        assert_eq!(
            (expected.soft, expected.coupling),
            (full.soft, full.coupling)
        );
        assert!(expected.hard > full.hard);
    }

    #[test]
    fn pairwise_hard_estimator_output_matches_the_library() {
        let (top, traj) = chain("cli_qh_mie2");
        let lines = lines(&run(
            &top,
            &traj,
            &["--estimator", "split", "--hard-estimator", "qh-mie2"],
        ));
        let expected = library(&top, &traj, |settings, _| {
            settings.hard_estimator = HardEstimator::QuasiHarmonicMie2;
        });
        let full = library(&top, &traj, |_, _| {});
        assert_eq!(
            value(&lines, "Hard entropy").parse::<f64>().unwrap(),
            expected.hard
        );
        assert_eq!(
            lines.last().unwrap(),
            &format!("Total entropy = {}", expected.total)
        );
        // Only the hard term changes.
        assert_eq!(
            (expected.soft, expected.coupling),
            (full.soft, full.coupling)
        );
        assert_ne!(expected.hard, full.hard);
    }

    #[test]
    fn per_state_output_lists_the_states() {
        let (top, traj) = chain("cli_states");
        let lines = lines(&run(
            &top,
            &traj,
            &[
                "--estimator",
                "split",
                "--coupling",
                "states",
                "--min-state-frames",
                "10",
            ],
        ));
        let expected = library(&top, &traj, |settings, _| {
            settings.coupling = Coupling::PerState(StateOptions {
                min_frames: 10,
                ..Default::default()
            });
        });
        let states = expected.states.unwrap();
        assert!(lines.contains(&"state\tbasins\tframes\tpopulation\thard_entropy".to_string()));
        assert_eq!(
            value(&lines, "Dropped frames"),
            states.dropped_frames.to_string()
        );
        assert_eq!(
            value(&lines, "Total entropy").parse::<f64>().unwrap(),
            expected.total
        );
    }

    #[test]
    fn a_large_soft_set_is_warned_about() {
        // The bundled fixture has 29 coordinates, enough to make them all soft.
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let list = (1..=29)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let output = run(
            &fixtures.join("test.parm7"),
            &fixtures.join("test.nc"),
            &[
                "--estimator",
                "split",
                "--soft",
                &list,
                "--coupling",
                "none",
            ],
        );
        lines_ok(&output);
        const { assert!(29 > LARGE_SOFT_SET) };
        assert!(String::from_utf8_lossy(&output.stderr).contains("Warning: 29 soft coordinates"));
        let (top, traj) = chain("cli_small_soft");
        let small = run(&top, &traj, &["--estimator", "split"]);
        assert!(!String::from_utf8_lossy(&small.stderr).contains("soft coordinates"));
    }

    #[test]
    fn mist_and_harmonics_flags_match_the_library() {
        let (top, traj) = chain("cli_mist");
        let mist = lines(&run(
            &top,
            &traj,
            &[
                "--estimator",
                "split",
                "--soft-estimator",
                "mist",
                "--harmonics",
                "2",
            ],
        ));
        let expected = library(&top, &traj, |settings, _| {
            settings.soft_estimator = SoftEstimator::MutualInformationSpanningTree;
            settings.coupling = Coupling::Gaussian {
                features: SoftFeatures::Harmonics(2),
            };
        });
        assert_eq!(
            value(&mist, "Soft entropy").parse::<f64>().unwrap(),
            expected.soft
        );
        assert_eq!(
            value(&mist, "Total entropy").parse::<f64>().unwrap(),
            expected.total
        );
        let network = lines(&run(
            &top,
            &traj,
            &[
                "--estimator",
                "split",
                "--soft-estimator",
                "network",
                "--network-parents",
                "3",
            ],
        ));
        let expected = library(&top, &traj, |settings, _| {
            settings.soft_estimator = SoftEstimator::BayesianNetwork { parents: 3 };
        });
        assert_eq!(
            value(&network, "Total entropy").parse::<f64>().unwrap(),
            expected.total
        );
        let cos_sin = lines(&run(
            &top,
            &traj,
            &["--estimator", "split", "--soft-features", "cossin"],
        ));
        let expected = library(&top, &traj, |settings, _| {
            settings.coupling = Coupling::Gaussian {
                features: SoftFeatures::CosSin,
            };
        });
        assert_eq!(
            value(&cos_sin, "Coupling term").parse::<f64>().unwrap(),
            expected.coupling
        );
    }

    #[test]
    fn per_coordinate_split_output_matches_the_library() {
        // The fixture's torsions only jitter about one value, so their higher
        // harmonics are nearly collinear and the shares' eigendecomposition
        // refuses them; one harmonic is well conditioned.
        let (top, traj) = chain("cli_per_coordinate");
        let output = run(
            &top,
            &traj,
            &[
                "--estimator",
                "split",
                "--per-coordinate",
                "--soft-features",
                "cossin",
            ],
        );
        let lines = lines(&output);
        let header = lines
            .iter()
            .position(|l| l == "coordinate\ttype\tatoms\tentropy\tset\town\tcoupling")
            .expect("per-coordinate header");
        let mut internal = InternalCoordinates::new(&top).unwrap();
        internal
            .calculate_internal_coords(&traj, usize::MAX, false)
            .unwrap();
        let data = internal.columns(0, 1).unwrap();
        let metrics = internal.coordinate_metrics();
        let soft = internal.soft_torsions();
        let mut settings = SplitOptions::new(SoftSelection::Indices(&soft));
        settings.coupling = Coupling::Gaussian {
            features: SoftFeatures::CosSin,
        };
        let expected = estimate_coordinate_split_entropy(
            &data,
            data[0].len(),
            &settings,
            &EntropyOptions {
                metrics: Some(&metrics),
                ..Default::default()
            },
        )
        .unwrap();
        for (index, line) in lines[header + 1..header + 1 + data.len()]
            .iter()
            .enumerate()
        {
            let cells: Vec<&str> = line.split('\t').collect();
            assert_eq!(cells[0], (index + 1).to_string());
            let set = if soft.contains(&index) {
                "soft"
            } else {
                "hard"
            };
            assert_eq!(cells[4], set);
            assert_eq!(cells[3].parse::<f64>().unwrap(), expected.total[index]);
            assert_eq!(cells[5].parse::<f64>().unwrap(), expected.own[index]);
            assert_eq!(cells[6].parse::<f64>().unwrap(), expected.coupling[index]);
        }
        assert_eq!(
            lines.last().unwrap(),
            &format!("Total entropy = {}", expected.split.total)
        );
    }

    #[test]
    fn soft_and_state_lists_count_from_one() {
        let (top, traj) = chain("cli_lists");
        let lines = lines(&run(
            &top,
            &traj,
            &[
                "--estimator",
                "split",
                "--soft",
                "8,10",
                "--coupling",
                "states",
                "--state-torsions",
                "8",
                "--min-state-frames",
                "10",
            ],
        ));
        assert_eq!(value(&lines, "Soft coordinates"), "8,10");
        let expected = library(&top, &traj, |settings, _| {
            settings.soft = SoftSelection::Indices(&[7, 9]);
            settings.coupling = Coupling::PerState(StateOptions {
                definition: StateDefinition::Detected {
                    defining: Some(&[7]),
                    exclude: &[],
                },
                min_frames: 10,
                ..Default::default()
            });
        });
        assert_eq!(
            value(&lines, "Total entropy").parse::<f64>().unwrap(),
            expected.total
        );
        // Detection runs on every torsion.
        lines_ok(&run(
            &top,
            &traj,
            &["--estimator", "split", "--soft", "detect"],
        ));
    }

    fn lines_ok(output: &Output) {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn split_flags_are_checked() {
        let (top, traj) = chain("cli_rejected");
        for (args, message) in [
            (
                &["--coupling", "none"][..],
                "--coupling applies only to --estimator split",
            ),
            (
                &["--estimator", "split", "--soft", "99"][..],
                "--soft coordinate 99 is beyond",
            ),
            (
                &["--estimator", "split", "--soft", "0"][..],
                "invalid --soft value",
            ),
            (
                &[
                    "--estimator",
                    "split",
                    "--coupling",
                    "states",
                    "--soft-features",
                    "cossin",
                ][..],
                "--soft-features and --harmonics apply only",
            ),
            (
                &[
                    "--estimator",
                    "split",
                    "--soft-features",
                    "cossin",
                    "--harmonics",
                    "2",
                ][..],
                "--harmonics applies only to --soft-features harmonics",
            ),
            (
                &["--estimator", "split", "--harmonics", "0"][..],
                "--harmonics must be at least 1",
            ),
            (
                &[
                    "--estimator",
                    "split",
                    "--soft-estimator",
                    "mist",
                    "--mie-order",
                    "3",
                ][..],
                "--mie-order applies only",
            ),
            (
                &[
                    "--estimator",
                    "split",
                    "--soft-estimator",
                    "mist",
                    "--network-parents",
                    "3",
                ][..],
                "--network-parents applies only",
            ),
            (
                &["--estimator", "split", "--state-torsions", "auto"][..],
                "apply only to --coupling states",
            ),
            (
                &["--estimator", "split", "--mutual-information"][..],
                "--mutual-information applies only to --estimator nn",
            ),
            (
                &[
                    "--estimator",
                    "split",
                    "--soft-estimator",
                    "joint",
                    "--per-coordinate",
                ][..],
                "the joint soft estimate has no per-coordinate split",
            ),
            (
                &["--estimator", "split", "--no-phase-angles"][..],
                "--soft topology needs phase angles",
            ),
            (
                &[
                    "--estimator",
                    "split",
                    "--soft-estimator",
                    "joint",
                    "--mie-order",
                    "3",
                ][..],
                "--mie-order applies only",
            ),
            (
                &["--estimator", "split", "--histogram-bins", "20"][..],
                "--histogram-bins applies only",
            ),
            (
                &["--estimator", "split", "--coupling", "maybe"][..],
                "--coupling must be one of gaussian, states, nn, none",
            ),
        ] {
            let output = run(&top, &traj, args);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(output.status.code(), Some(1), "{args:?}: {stderr}");
            assert!(stderr.contains(message), "{args:?}: {stderr}");
        }
    }
}
