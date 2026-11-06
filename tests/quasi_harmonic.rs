//! Quasi-harmonic (QH) and Boltzmann quasi-harmonic (BQH) entropy.
//!
//! Expected values come from closed forms (normal and uniform entropies), from
//! sample covariances computed independently here, or from the
//! nearest-neighbor estimator the BQH first-order term is defined to equal.

mod common;

use std::f64::consts::{E, PI, TAU};

use common::{Rng, bivariate_normal_entropy, normal_entropy};
use molentropy::quasi_harmonic::{
    CoordinateQuasiHarmonicEntropy, MarginalEstimator, PeriodicCorrelation, QuasiHarmonicEntropy,
    QuasiHarmonicOptions, estimate_coordinate_quasi_harmonic_entropy,
    estimate_quasi_harmonic_entropy,
};
use molentropy::{CoordinateMetric, EntropyOptions, estimate_coordinate_entropy};

fn estimate(data: &[Vec<f64>], marginals: MarginalEstimator) -> QuasiHarmonicEntropy {
    let options = QuasiHarmonicOptions {
        marginals,
        ..Default::default()
    };
    estimate_quasi_harmonic_entropy(data, data[0].len(), &options, &EntropyOptions::default())
        .unwrap()
}

fn estimate_with_metrics(data: &[Vec<f64>], metrics: &[CoordinateMetric]) -> QuasiHarmonicEntropy {
    estimate_periodic(data, metrics, PeriodicCorrelation::Complex)
}

fn estimate_periodic(
    data: &[Vec<f64>],
    metrics: &[CoordinateMetric],
    periodic: PeriodicCorrelation,
) -> QuasiHarmonicEntropy {
    let options = EntropyOptions {
        metrics: Some(metrics),
        ..Default::default()
    };
    let quasi_harmonic = QuasiHarmonicOptions {
        periodic,
        ..Default::default()
    };
    estimate_quasi_harmonic_entropy(data, data[0].len(), &quasi_harmonic, &options).unwrap()
}

/// Population covariance, two-pass.
fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / n
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: {actual} vs {expected} (tolerance {tolerance})"
    );
}

// ---------------------------------------------------------------------------
// Quasi-harmonic term against the sample covariance
// ---------------------------------------------------------------------------

#[test]
fn bivariate_qh_is_the_gaussian_entropy_of_the_sample_covariance() {
    let mut rng = Rng::new(11);
    let (x, y) = rng.correlated_normal_pair(5_000, 0.7);
    let x: Vec<f64> = x.iter().map(|v| 2.0 * v + 3.0).collect();
    let result = estimate(&[x.clone(), y.clone()], MarginalEstimator::NearestNeighbor);

    let (sxx, syy, sxy) = (covariance(&x, &x), covariance(&y, &y), covariance(&x, &y));
    let determinant = sxx * syy - sxy * sxy;
    let r = sxy / (sxx * syy).sqrt();
    assert_close(
        result.quasi_harmonic,
        (TAU * E).ln() + 0.5 * determinant.ln(),
        1e-10,
        "QH",
    );
    assert_close(
        result.correlation,
        0.5 * (1.0 - r * r).ln(),
        1e-10,
        "1/2 ln|C|",
    );
    assert_close(
        result.first_order_gaussian,
        0.5 * (TAU * E * sxx).ln() + 0.5 * (TAU * E * syy).ln(),
        1e-10,
        "Gaussian first order",
    );
}

#[test]
fn trivariate_qh_uses_the_full_determinant() {
    let mut rng = Rng::new(12);
    let n = 4_000;
    let (a, b, c) = (
        rng.normal_vec(n, 0.0, 1.0),
        rng.normal_vec(n, 0.0, 1.0),
        rng.normal_vec(n, 0.0, 1.0),
    );
    let x: Vec<f64> = (0..n).map(|k| a[k]).collect();
    let y: Vec<f64> = (0..n).map(|k| 0.5 * a[k] + b[k]).collect();
    let z: Vec<f64> = (0..n)
        .map(|k| -0.3 * a[k] + 0.8 * b[k] + 0.4 * c[k])
        .collect();
    let result = estimate(
        &[x.clone(), y.clone(), z.clone()],
        MarginalEstimator::NearestNeighbor,
    );

    let s = |p: &[f64], q: &[f64]| covariance(p, q);
    let m = [
        [s(&x, &x), s(&x, &y), s(&x, &z)],
        [s(&y, &x), s(&y, &y), s(&y, &z)],
        [s(&z, &x), s(&z, &y), s(&z, &z)],
    ];
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    assert_close(
        result.quasi_harmonic,
        1.5 * (TAU * E).ln() + 0.5 * determinant.ln(),
        1e-10,
        "QH",
    );
}

#[test]
fn qh_converges_to_the_bivariate_normal_entropy() {
    let mut rng = Rng::new(13);
    let (x, y) = rng.correlated_normal_pair(200_000, 0.8);
    let result = estimate(&[x, y], MarginalEstimator::NearestNeighbor);
    assert_close(
        result.quasi_harmonic,
        bivariate_normal_entropy(0.8),
        0.01,
        "QH",
    );
    assert_close(
        result.boltzmann_quasi_harmonic,
        bivariate_normal_entropy(0.8),
        0.02,
        "BQH",
    );
}

#[test]
fn independent_coordinates_have_almost_no_correlation_term() {
    let mut rng = Rng::new(14);
    let data: Vec<Vec<f64>> = (0..4).map(|_| rng.normal_vec(50_000, 0.0, 1.0)).collect();
    let result = estimate(&data, MarginalEstimator::NearestNeighbor);
    assert!(result.correlation <= 0.0);
    assert!(result.correlation > -0.01, "{}", result.correlation);
}

#[test]
fn scaling_a_coordinate_shifts_qh_by_the_log_of_the_factor() {
    let mut rng = Rng::new(15);
    let (x, y) = rng.correlated_normal_pair(3_000, 0.4);
    let base = estimate(&[x.clone(), y.clone()], MarginalEstimator::NearestNeighbor);
    let scaled_x: Vec<f64> = x.iter().map(|v| -3.0 * v).collect();
    let scaled = estimate(&[scaled_x, y], MarginalEstimator::NearestNeighbor);
    assert_close(
        scaled.quasi_harmonic,
        base.quasi_harmonic + 3.0_f64.ln(),
        1e-10,
        "QH",
    );
    assert_close(scaled.correlation, base.correlation, 1e-12, "correlation");
}

// ---------------------------------------------------------------------------
// Boltzmann first-order term
// ---------------------------------------------------------------------------

#[test]
fn nearest_neighbor_marginals_are_the_first_order_nn_entropies() {
    let mut rng = Rng::new(16);
    let (x, y) = rng.correlated_normal_pair(2_000, 0.5);
    let z = rng.exponential_vec(2_000, 2.0);
    let data = vec![x, y, z];
    let result = estimate(&data, MarginalEstimator::NearestNeighbor);
    let marginals: f64 = estimate_coordinate_entropy(&data, 2_000, 1, &EntropyOptions::default())
        .unwrap()
        .iter()
        .sum();
    assert_eq!(result.first_order_boltzmann, marginals);
    assert_eq!(
        result.boltzmann_quasi_harmonic,
        result.first_order_boltzmann + result.correlation
    );
}

#[test]
fn bqh_corrects_qh_for_a_bimodal_coordinate() {
    // Two narrow, well separated wells: the true entropy is
    // ln 2 + H(normal, 0.1), while a Gaussian fit sees one wide well.
    let mut rng = Rng::new(17);
    let x: Vec<f64> = (0..40_000)
        .map(|k| if k % 2 == 0 { -2.0 } else { 2.0 } + 0.1 * rng.normal())
        .collect();
    let y = rng.normal_vec(40_000, 0.0, 1.0);
    let truth = 2.0_f64.ln() + normal_entropy(0.1) + normal_entropy(1.0);
    for marginals in [
        MarginalEstimator::NearestNeighbor,
        MarginalEstimator::Histogram { bins: 400 },
    ] {
        let result = estimate(&[x.clone(), y.clone()], marginals);
        assert_close(result.boltzmann_quasi_harmonic, truth, 0.03, "BQH");
        assert!(
            result.quasi_harmonic > truth + 1.0,
            "QH should overestimate"
        );
    }
}

#[test]
fn histogram_entropy_matches_hand_counted_bins() {
    // Range [0, 3] in 3 bins of width 1: the maximum falls in the last bin, so
    // the counts are 3, 1 and 4 of 8.
    let x = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let result = estimate(&[x], MarginalEstimator::Histogram { bins: 3 });
    let expected: f64 = [3.0, 1.0, 4.0]
        .iter()
        .map(|c| -(c / 8.0) * (c / 8.0_f64).ln())
        .sum();
    assert_close(result.first_order_boltzmann, expected, 1e-15, "histogram");
}

#[test]
fn periodic_histograms_span_one_period() {
    // -0.1 wraps to 2 pi - 0.1, in the last of four bins of width pi / 2.
    let x = vec![0.1, 0.2, 1.7, 3.3, -0.1, -0.2];
    let metrics = [CoordinateMetric::Periodic { period: TAU }];
    let options = EntropyOptions {
        metrics: Some(&metrics),
        ..Default::default()
    };
    let result = estimate_quasi_harmonic_entropy(
        &[x],
        6,
        &QuasiHarmonicOptions {
            marginals: MarginalEstimator::Histogram { bins: 4 },
            ..Default::default()
        },
        &options,
    )
    .unwrap();
    // Counts 2, 1, 1, 2.
    let width = PI / 2.0;
    let expected: f64 = [2.0, 1.0, 1.0, 2.0]
        .iter()
        .map(|c| -(c / 6.0) * ((c / 6.0) / width).ln())
        .sum();
    assert_close(
        result.first_order_boltzmann,
        expected,
        1e-15,
        "periodic histogram",
    );
}

#[test]
fn histogram_of_a_uniform_sample_converges() {
    let mut rng = Rng::new(18);
    let x = rng.uniform_vec(200_000, 0.0, 3.0);
    let result = estimate(&[x], MarginalEstimator::Histogram { bins: 100 });
    assert_close(result.first_order_boltzmann, 3.0_f64.ln(), 0.01, "uniform");
}

// ---------------------------------------------------------------------------
// Periodic coordinates
// ---------------------------------------------------------------------------

/// A torsion whose single well straddles the wrap point at +-pi.
fn torsion_at_pi(rng: &mut Rng, n: usize, sigma: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let phi = PI + sigma * rng.normal();
            phi - TAU * ((phi + PI) / TAU).floor()
        })
        .collect()
}

#[test]
fn a_well_across_the_wrap_point_is_centered() {
    let mut rng = Rng::new(19);
    let phi = torsion_at_pi(&mut rng, 50_000, 0.3);
    let periodic = estimate_with_metrics(
        std::slice::from_ref(&phi),
        &[CoordinateMetric::Periodic { period: TAU }],
    );
    let linear = estimate_with_metrics(&[phi], &[CoordinateMetric::Linear]);
    assert_close(
        periodic.quasi_harmonic,
        normal_entropy(0.3),
        0.01,
        "periodic QH",
    );
    assert!(linear.quasi_harmonic > periodic.quasi_harmonic + 1.0);
}

#[test]
fn rotating_a_periodic_coordinate_changes_nothing() {
    let mut rng = Rng::new(20);
    let phi = torsion_at_pi(&mut rng, 5_000, 0.4);
    let x: Vec<f64> = phi
        .iter()
        .map(|p| 0.5 * p.sin() + 0.2 * rng.normal())
        .collect();
    let metrics = [
        CoordinateMetric::Periodic { period: TAU },
        CoordinateMetric::Linear,
    ];
    let base = estimate_with_metrics(&[phi.clone(), x.clone()], &metrics);
    let rotated: Vec<f64> = phi.iter().map(|p| p + 1.234).collect();
    let moved = estimate_with_metrics(&[rotated, x], &metrics);
    assert_close(moved.quasi_harmonic, base.quasi_harmonic, 1e-9, "QH");
    assert_close(moved.correlation, base.correlation, 1e-9, "correlation");
}

// ---------------------------------------------------------------------------
// Complex periodic correlations (Harpole and Sharp's z = e^(i phi))
// ---------------------------------------------------------------------------

type Complex = (f64, f64);

fn mul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn conj(a: Complex) -> Complex {
    (a.0, -a.1)
}

fn norm_sqr(a: Complex) -> f64 {
    a.0 * a.0 + a.1 * a.1
}

/// Deviations from the mean: e^(i phi) for a torsion, the value itself for a
/// linear coordinate.
fn deviations(values: &[f64], periodic: bool) -> Vec<Complex> {
    let z: Vec<Complex> = values
        .iter()
        .map(|&v| {
            if periodic {
                (v.cos(), v.sin())
            } else {
                (v, 0.0)
            }
        })
        .collect();
    let n = z.len() as f64;
    let mean = z
        .iter()
        .fold((0.0, 0.0), |m, v| (m.0 + v.0 / n, m.1 + v.1 / n));
    z.iter().map(|v| (v.0 - mean.0, v.1 - mean.1)).collect()
}

/// Hermitian correlation coefficient <a conj(b)> / sqrt(<|a|^2> <|b|^2>).
fn coefficient(a: &[Complex], b: &[Complex]) -> Complex {
    let n = a.len() as f64;
    let cross = a.iter().zip(b).fold((0.0, 0.0), |s, (x, y)| {
        let p = mul(*x, conj(*y));
        (s.0 + p.0 / n, s.1 + p.1 / n)
    });
    let scale = (a.iter().map(|v| norm_sqr(*v)).sum::<f64>() / n
        * (b.iter().map(|v| norm_sqr(*v)).sum::<f64>() / n))
        .sqrt();
    (cross.0 / scale, cross.1 / scale)
}

fn wrap(angle: f64) -> f64 {
    angle - TAU * ((angle + PI) / TAU).floor()
}

const TORSION: CoordinateMetric = CoordinateMetric::Periodic { period: TAU };

#[test]
fn a_rotor_pair_correlates_through_the_hermitian_coefficient() {
    // Two hydrogens of a freely rotating methyl group: both torsions sweep the
    // circle, 120 degrees apart up to vibration.
    let mut rng = Rng::new(31);
    let phi1 = rng.uniform_vec(20_000, -PI, PI);
    let phi2: Vec<f64> = phi1
        .iter()
        .map(|p| wrap(p + TAU / 3.0 + 0.2 * rng.normal()))
        .collect();
    let data = [phi1.clone(), phi2.clone()];
    let complex = estimate_periodic(&data, &[TORSION, TORSION], PeriodicCorrelation::Complex);

    let rho = coefficient(&deviations(&phi1, true), &deviations(&phi2, true));
    assert!(norm_sqr(rho) > 0.9, "|rho|^2 = {}", norm_sqr(rho));
    assert_close(
        complex.correlation,
        0.5 * (1.0 - norm_sqr(rho)).ln(),
        1e-12,
        "1/2 ln|C|",
    );

    // Centering sees a sawtooth and misses most of the correlation.
    let centered = estimate_periodic(&data, &[TORSION, TORSION], PeriodicCorrelation::Centered);
    assert!(centered.correlation > complex.correlation + 0.5);
}

#[test]
fn a_linear_coordinate_correlates_with_a_torsion_through_cos_and_sin() {
    let mut rng = Rng::new(32);
    let phi = rng.uniform_vec(20_000, -PI, PI);
    let x: Vec<f64> = phi
        .iter()
        .map(|p| 0.8 * p.cos() - 0.5 * p.sin() + 0.3 * rng.normal())
        .collect();
    let result = estimate_periodic(
        &[x.clone(), phi.clone()],
        &[CoordinateMetric::Linear, TORSION],
        PeriodicCorrelation::Complex,
    );
    let rho = coefficient(&deviations(&x, false), &deviations(&phi, true));
    assert_close(
        result.correlation,
        0.5 * (1.0 - norm_sqr(rho)).ln(),
        1e-12,
        "1/2 ln|C|",
    );
}

#[test]
fn three_coordinates_match_the_hermitian_determinant() {
    let mut rng = Rng::new(33);
    let n = 10_000;
    let phi1 = rng.uniform_vec(n, -PI, PI);
    let phi2: Vec<f64> = phi1.iter().map(|p| wrap(-p + 0.4 * rng.normal())).collect();
    let x: Vec<f64> = phi1.iter().map(|p| p.sin() + rng.normal()).collect();
    let result = estimate_periodic(
        &[phi1.clone(), x.clone(), phi2.clone()],
        &[TORSION, CoordinateMetric::Linear, TORSION],
        PeriodicCorrelation::Complex,
    );
    let (a, b, c) = (
        deviations(&phi1, true),
        deviations(&x, false),
        deviations(&phi2, true),
    );
    let (c12, c13, c23) = (
        coefficient(&a, &b),
        coefficient(&a, &c),
        coefficient(&b, &c),
    );
    // det of [[1, c12, c13], [c12*, 1, c23], [c13*, c23*, 1]]
    let determinant =
        1.0 + 2.0 * mul(mul(c12, c23), conj(c13)).0 - norm_sqr(c12) - norm_sqr(c13) - norm_sqr(c23);
    assert_close(
        result.correlation,
        0.5 * determinant.ln(),
        1e-12,
        "1/2 ln|C|",
    );
}

#[test]
fn linear_coordinates_are_unaffected_by_the_periodic_treatment() {
    let mut rng = Rng::new(34);
    let (x, y) = rng.correlated_normal_pair(3_000, 0.6);
    let metrics = [CoordinateMetric::Linear, CoordinateMetric::Linear];
    let data = [x, y];
    assert_eq!(
        estimate_periodic(&data, &metrics, PeriodicCorrelation::Complex),
        estimate_periodic(&data, &metrics, PeriodicCorrelation::Centered)
    );
}

#[test]
fn narrow_torsions_correlate_alike_either_way() {
    // In one narrow well e^(i phi) is linear in phi to first order, so both
    // treatments see the same correlation.
    let mut rng = Rng::new(35);
    let (u, v) = rng.correlated_normal_pair(50_000, 0.7);
    let phi: Vec<f64> = u.iter().map(|u| wrap(3.0 + 0.05 * u)).collect();
    let psi: Vec<f64> = v.iter().map(|v| wrap(-1.0 + 0.05 * v)).collect();
    let data = [phi, psi];
    let complex = estimate_periodic(&data, &[TORSION, TORSION], PeriodicCorrelation::Complex);
    let centered = estimate_periodic(&data, &[TORSION, TORSION], PeriodicCorrelation::Centered);
    assert_close(
        complex.correlation,
        centered.correlation,
        1e-3,
        "correlation",
    );
    assert_close(
        complex.correlation,
        0.5 * (1.0 - 0.49_f64).ln(),
        0.01,
        "closed form",
    );
}

#[test]
fn rotating_each_torsion_leaves_the_complex_correlation_unchanged() {
    let mut rng = Rng::new(36);
    let phi1 = rng.uniform_vec(5_000, -PI, PI);
    let phi2: Vec<f64> = phi1
        .iter()
        .map(|p| wrap(p + 1.0 + 0.3 * rng.normal()))
        .collect();
    let base = estimate_with_metrics(&[phi1.clone(), phi2.clone()], &[TORSION, TORSION]);
    let moved = estimate_with_metrics(
        &[
            phi1.iter().map(|p| p + 0.7).collect(),
            phi2.iter().map(|p| p - 2.1).collect(),
        ],
        &[TORSION, TORSION],
    );
    assert_close(moved.correlation, base.correlation, 1e-10, "correlation");
}

// ---------------------------------------------------------------------------
// Per-coordinate shares
// ---------------------------------------------------------------------------

fn per_coordinate(
    data: &[Vec<f64>],
    metrics: &[CoordinateMetric],
    quasi_harmonic: QuasiHarmonicOptions,
) -> (CoordinateQuasiHarmonicEntropy, QuasiHarmonicEntropy) {
    let options = EntropyOptions {
        metrics: Some(metrics),
        ..Default::default()
    };
    let n = data[0].len();
    (
        estimate_coordinate_quasi_harmonic_entropy(data, n, &quasi_harmonic, &options).unwrap(),
        estimate_quasi_harmonic_entropy(data, n, &quasi_harmonic, &options).unwrap(),
    )
}

/// Two correlated normals, a rotor pair of torsions, and an exponential.
fn mixed_system(rng: &mut Rng, n: usize) -> (Vec<Vec<f64>>, Vec<CoordinateMetric>) {
    let (x, y) = rng.correlated_normal_pair(n, 0.6);
    let phi = rng.uniform_vec(n, -PI, PI);
    let psi: Vec<f64> = phi
        .iter()
        .zip(&x)
        .map(|(p, x)| wrap(p + TAU / 3.0 + 0.2 * rng.normal() + 0.1 * x))
        .collect();
    let z = rng.exponential_vec(n, 2.0);
    let metrics = vec![
        CoordinateMetric::Linear,
        CoordinateMetric::Linear,
        TORSION,
        TORSION,
        CoordinateMetric::Linear,
    ];
    (vec![x, y, phi, psi, z], metrics)
}

#[test]
fn per_coordinate_shares_sum_to_the_totals() {
    let mut rng = Rng::new(41);
    let (data, metrics) = mixed_system(&mut rng, 8_000);
    for quasi_harmonic in [
        QuasiHarmonicOptions::default(),
        QuasiHarmonicOptions {
            marginals: MarginalEstimator::Histogram { bins: 50 },
            periodic: PeriodicCorrelation::Centered,
            ..Default::default()
        },
    ] {
        let (shares, totals) = per_coordinate(&data, &metrics, quasi_harmonic);
        let sum = |values: &[f64]| values.iter().sum::<f64>();
        assert_close(
            sum(&shares.quasi_harmonic),
            totals.quasi_harmonic,
            1e-9,
            "QH",
        );
        assert_close(
            sum(&shares.boltzmann_quasi_harmonic),
            totals.boltzmann_quasi_harmonic,
            1e-9,
            "BQH",
        );
        assert_close(
            sum(&shares.correlation),
            totals.correlation,
            1e-9,
            "correlation",
        );
        assert_eq!(
            sum(&shares.first_order_gaussian),
            totals.first_order_gaussian
        );
        assert_eq!(
            sum(&shares.first_order_boltzmann),
            totals.first_order_boltzmann
        );
        assert!(shares.correlation.iter().all(|&share| share <= 1e-12));
    }
}

#[test]
fn two_coordinates_split_the_correlation_evenly() {
    // C = [[1, rho], [rho*, 1]] has eigenvalues 1 +- |rho| with eigenvectors of
    // equal weight, so (log C)_ii = 1/2 ln(1 - |rho|^2) for both coordinates.
    let mut rng = Rng::new(42);
    let (x, y) = rng.correlated_normal_pair(5_000, 0.7);
    let linear = [CoordinateMetric::Linear, CoordinateMetric::Linear];
    let (shares, _) = per_coordinate(&[x.clone(), y.clone()], &linear, Default::default());
    let r = covariance(&x, &y) / (covariance(&x, &x) * covariance(&y, &y)).sqrt();
    for share in &shares.correlation {
        assert_close(*share, 0.25 * (1.0 - r * r).ln(), 1e-12, "real pair");
    }

    let phi1 = rng.uniform_vec(5_000, -PI, PI);
    let phi2: Vec<f64> = phi1
        .iter()
        .map(|p| wrap(p + 2.0 + 0.3 * rng.normal()))
        .collect();
    let (shares, _) = per_coordinate(
        &[phi1.clone(), phi2.clone()],
        &[TORSION, TORSION],
        Default::default(),
    );
    let rho = coefficient(&deviations(&phi1, true), &deviations(&phi2, true));
    for share in &shares.correlation {
        assert_close(
            *share,
            0.25 * (1.0 - norm_sqr(rho)).ln(),
            1e-12,
            "rotor pair",
        );
    }
}

#[test]
fn shares_follow_their_coordinates_when_relabelled() {
    let mut rng = Rng::new(43);
    let (data, metrics) = mixed_system(&mut rng, 4_000);
    let (shares, _) = per_coordinate(&data, &metrics, Default::default());
    let order = [3, 0, 4, 2, 1];
    let permuted: Vec<Vec<f64>> = order.iter().map(|&i| data[i].clone()).collect();
    let permuted_metrics: Vec<CoordinateMetric> = order.iter().map(|&i| metrics[i]).collect();
    let (moved, _) = per_coordinate(&permuted, &permuted_metrics, Default::default());
    for (position, &original) in order.iter().enumerate() {
        assert_close(
            moved.correlation[position],
            shares.correlation[original],
            1e-10,
            "correlation share",
        );
    }
}

#[test]
fn an_independent_coordinate_takes_no_correlation_share() {
    let mut rng = Rng::new(44);
    let (x, y) = rng.correlated_normal_pair(50_000, 0.8);
    let z = rng.normal_vec(50_000, 0.0, 1.0);
    let linear = [CoordinateMetric::Linear; 3];
    let (shares, _) = per_coordinate(&[x, y, z], &linear, Default::default());
    // Each of x and y takes about 1/4 ln(1 - 0.64) = -0.255.
    assert!(shares.correlation[0] < -0.2 && shares.correlation[1] < -0.2);
    assert!(shares.correlation[2] > -1e-3, "{}", shares.correlation[2]);
}

#[test]
fn first_order_shares_are_the_single_coordinate_estimates() {
    let mut rng = Rng::new(45);
    let (data, metrics) = mixed_system(&mut rng, 3_000);
    let (shares, _) = per_coordinate(&data, &metrics, Default::default());
    let options = EntropyOptions {
        metrics: Some(&metrics),
        ..Default::default()
    };
    let marginals = estimate_coordinate_entropy(&data, 3_000, 1, &options).unwrap();
    assert_eq!(shares.first_order_boltzmann, marginals);
    for i in [0, 1, 4] {
        let variance = covariance(&data[i], &data[i]);
        assert_close(
            shares.first_order_gaussian[i],
            0.5 * (TAU * E * variance).ln(),
            1e-12,
            "Gaussian first order",
        );
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
fn too_few_frames_are_rejected() {
    let data = vec![
        vec![0.0, 1.0, 2.0],
        vec![1.0, 0.0, 2.0],
        vec![2.0, 2.0, 0.0],
    ];
    let err = estimate_quasi_harmonic_entropy(
        &data,
        3,
        &QuasiHarmonicOptions {
            marginals: MarginalEstimator::NearestNeighbor,
            ..Default::default()
        },
        &EntropyOptions::default(),
    )
    .unwrap_err();
    assert!(err.contains("needs more than 3 frames"), "{err}");
}

#[test]
fn a_linearly_dependent_coordinate_is_rejected() {
    // Rounding leaves the zero eigenvalue of an exactly dependent set slightly
    // positive or negative; either way it must count as singular.
    let mut rng = Rng::new(21);
    let x = rng.normal_vec(100, 0.0, 1.0);
    let z = rng.normal_vec(100, 0.0, 1.0);
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    let err = estimate_quasi_harmonic_entropy(
        &[x.clone(), z, y.clone()],
        100,
        &QuasiHarmonicOptions::default(),
        &EntropyOptions::default(),
    )
    .unwrap_err();
    assert!(err.contains("singular"), "{err}");
    let err = estimate_quasi_harmonic_entropy(
        &[x, y],
        100,
        &QuasiHarmonicOptions {
            marginals: MarginalEstimator::NearestNeighbor,
            ..Default::default()
        },
        &EntropyOptions::default(),
    )
    .unwrap_err();
    assert!(err.contains("singular"), "{err}");
}

#[test]
fn a_linearly_dependent_coordinate_is_rejected_per_coordinate_too() {
    let mut rng = Rng::new(21);
    let x = rng.normal_vec(100, 0.0, 1.0);
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    let z = rng.normal_vec(100, 0.0, 1.0);
    let err = estimate_coordinate_quasi_harmonic_entropy(
        &[x, z, y],
        100,
        &QuasiHarmonicOptions::default(),
        &EntropyOptions::default(),
    )
    .unwrap_err();
    assert!(err.contains("singular"), "{err}");
}

#[test]
fn a_histogram_needs_bins() {
    let err = estimate_quasi_harmonic_entropy(
        &[vec![0.0, 1.0, 2.0]],
        3,
        &QuasiHarmonicOptions {
            marginals: MarginalEstimator::Histogram { bins: 0 },
            ..Default::default()
        },
        &EntropyOptions::default(),
    )
    .unwrap_err();
    assert!(err.contains("at least one bin"), "{err}");
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
    use molentropy::quasi_harmonic::{
        MarginalEstimator, PeriodicCorrelation, QuasiHarmonicOptions,
        estimate_quasi_harmonic_entropy,
    };

    /// A chain molecule with more frames than coordinates.
    fn chain_system(name: &str) -> (PathBuf, PathBuf) {
        let mut topology = Topology::default();
        topology.push_molecule("BUT", &CHAIN_ATOMS, &CHAIN_BONDS);
        let top = scratch_path(&format!("qh_{name}.parm7"));
        let traj = scratch_path(&format!("qh_{name}.nc"));
        write_prmtop(&top, &topology);
        let mut rng = Rng::new(0x9a11);
        let frames = chain_frames(&mut rng, 60, [0.0, 0.0, 0.0]);
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

    fn entropy(output: &Output) -> f64 {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        // The total is always the last line.
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .lines()
            .last()
            .and_then(|line| line.strip_prefix("Total entropy = "))
            .expect("missing entropy output")
            .parse()
            .unwrap()
    }

    #[test]
    fn estimator_flag_selects_qh_or_bqh() {
        let (top, traj) = chain_system("cli_estimators");
        let mut internal = InternalCoordinates::new(&top).unwrap();
        internal
            .calculate_internal_coords(&traj, usize::MAX, false)
            .unwrap();
        let metrics = internal.coordinate_metrics();
        let data: Vec<Vec<f64>> = (0..internal.dim)
            .map(|j| internal.int_coords.iter().map(|frame| frame[j]).collect())
            .collect();
        let options = EntropyOptions {
            metrics: Some(&metrics),
            ..Default::default()
        };
        let expected = |marginals| {
            let quasi_harmonic = QuasiHarmonicOptions {
                marginals,
                ..Default::default()
            };
            estimate_quasi_harmonic_entropy(&data, 60, &quasi_harmonic, &options).unwrap()
        };

        let nn = expected(MarginalEstimator::NearestNeighbor);
        let histogram = expected(MarginalEstimator::Histogram { bins: 20 });
        assert_eq!(
            entropy(&run(&top, &traj, &["--estimator", "qh"])),
            nn.quasi_harmonic
        );
        assert_eq!(
            entropy(&run(&top, &traj, &["--estimator", "bqh"])),
            nn.boltzmann_quasi_harmonic
        );
        assert_eq!(
            entropy(&run(
                &top,
                &traj,
                &["--estimator", "bqh", "--histogram-bins", "20"]
            )),
            histogram.boltzmann_quasi_harmonic
        );
        assert_eq!(
            entropy(&run(&top, &traj, &["--estimator", "nn"])),
            entropy(&run(&top, &traj, &[]))
        );
    }

    #[test]
    fn inapplicable_estimator_options_are_rejected() {
        let (top, traj) = chain_system("cli_rejected");
        for (args, message) in [
            (
                &["--estimator", "qh", "--mie-order", "3"][..],
                "--mie-order applies only",
            ),
            (
                &["--histogram-bins", "10"][..],
                "--histogram-bins applies only",
            ),
            (
                &["--estimator", "qh", "--histogram-bins", "10"][..],
                "--histogram-bins applies only",
            ),
            (
                &["--estimator", "bqh", "--histogram-bins", "0"][..],
                "at least 1",
            ),
            (
                &["--estimator", "qh", "--legacy"][..],
                "--estimator other than nn",
            ),
            (&["--estimator", "kde"][..], "must be nn, qh, bqh or split"),
            (
                &["--periodic-correlation", "complex"][..],
                "--periodic-correlation applies only",
            ),
            (
                &["--estimator", "qh", "--periodic-correlation", "polar"][..],
                "must be complex or centered",
            ),
        ] {
            let output = run(&top, &traj, args);
            assert!(!output.status.success(), "{args:?} should fail");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains(message), "{args:?}: {stderr}");
        }
    }

    #[test]
    fn periodic_correlation_flag_reaches_the_estimator() {
        let (top, traj) = chain_system("cli_periodic");
        let mut internal = InternalCoordinates::new(&top).unwrap();
        internal
            .calculate_internal_coords(&traj, usize::MAX, false)
            .unwrap();
        let metrics = internal.coordinate_metrics();
        let data: Vec<Vec<f64>> = (0..internal.dim)
            .map(|j| internal.int_coords.iter().map(|frame| frame[j]).collect())
            .collect();
        let options = EntropyOptions {
            metrics: Some(&metrics),
            ..Default::default()
        };
        for (flag, periodic) in [
            ("complex", PeriodicCorrelation::Complex),
            ("centered", PeriodicCorrelation::Centered),
        ] {
            let quasi_harmonic = QuasiHarmonicOptions {
                periodic,
                ..Default::default()
            };
            let expected =
                estimate_quasi_harmonic_entropy(&data, 60, &quasi_harmonic, &options).unwrap();
            let output = run(
                &top,
                &traj,
                &["--estimator", "qh", "--periodic-correlation", flag],
            );
            assert_eq!(entropy(&output), expected.quasi_harmonic, "{flag}");
        }
    }

    /// The rows of `--per-coordinate` output: (type, atoms, entropy).
    fn table(output: &Output) -> Vec<(String, String, f64)> {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines = stdout.lines();
        // QH and BQH tables add first_order and correlation columns.
        let header = lines.next().unwrap();
        assert!(
            header.starts_with("coordinate\ttype\tatoms\tentropy"),
            "{header}"
        );
        lines
            .take_while(|line| !line.contains('='))
            .enumerate()
            .map(|(index, line)| {
                let fields: Vec<&str> = line.split('\t').collect();
                assert_eq!(fields[0], (index + 1).to_string());
                (
                    fields[1].to_string(),
                    fields[2].to_string(),
                    fields[3].parse().unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn per_coordinate_tables_sum_to_the_total() {
        let (top, traj) = chain_system("cli_per_coordinate");
        let internal = {
            let mut internal = InternalCoordinates::new(&top).unwrap();
            internal
                .calculate_internal_coords(&traj, usize::MAX, false)
                .unwrap();
            internal
        };
        for args in [
            &["--estimator", "bqh", "--per-coordinate"][..],
            &["--estimator", "qh", "--per-coordinate"][..],
            &["--per-coordinate"][..],
            &["--per-coordinate", "--mie-order", "1"][..],
        ] {
            let output = run(&top, &traj, args);
            let rows = table(&output);
            assert_eq!(rows.len(), internal.dim, "{args:?}");
            let sum: f64 = rows.iter().map(|row| row.2).sum();
            assert!(
                (sum - entropy(&output)).abs() < 1e-9,
                "{args:?}: {sum} vs {}",
                entropy(&output)
            );
        }

        // Types and 1-based atoms follow the coordinates.
        let rows = table(&run(&top, &traj, &["--per-coordinate"]));
        let references = internal.phase_references();
        for (index, (atoms, row)) in internal.coordinate_atoms().iter().zip(&rows).enumerate() {
            let kind = match atoms.len() {
                2 => "bond",
                3 => "angle",
                _ if references[index].is_some() => "phase angle",
                _ => "torsion",
            };
            assert_eq!(row.0, kind);
            let numbered: Vec<String> = atoms.iter().map(|a| (a + 1).to_string()).collect();
            assert_eq!(row.1, numbered.join("-"));
        }
    }

    #[test]
    fn per_coordinate_rejects_legacy_mode() {
        // Every other estimator but the joint soft estimate has shares; see
        // tests/per_coordinate.rs and the split CLI tests.
        let (top, traj) = chain_system("cli_per_coordinate_rejected");
        let output = run(&top, &traj, &["--per-coordinate", "--legacy"]);
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--per-coordinate cannot be combined"),
            "{stderr}"
        );
    }

    /// The value of the `name = value` line of the output.
    fn line_value(output: &Output, name: &str) -> f64 {
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name} = ")))
            .unwrap_or_else(|| panic!("no {name} line"))
            .parse()
            .unwrap()
    }

    #[test]
    fn quasi_harmonic_terms_add_up_to_the_total() {
        let (top, traj) = chain_system("cli_terms");
        for estimator in ["qh", "bqh"] {
            let output = run(&top, &traj, &["--estimator", estimator]);
            let total = entropy(&output);
            let first_order = line_value(&output, "First-order entropy");
            let correlation = line_value(&output, "Correlation term");
            assert_eq!(first_order + correlation, total, "{estimator}");
            assert!(correlation <= 0.0);
        }
        // The NN MIE has no such terms.
        let stdout = String::from_utf8_lossy(&run(&top, &traj, &[]).stdout).to_string();
        assert!(!stdout.contains("First-order"), "{stdout}");
    }

    #[test]
    fn quasi_harmonic_tables_split_each_entropy_into_its_terms() {
        let (top, traj) = chain_system("cli_term_columns");
        let output = run(&top, &traj, &["--estimator", "bqh", "--per-coordinate"]);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines = stdout.lines();
        assert_eq!(
            lines.next(),
            Some("coordinate\ttype\tatoms\tentropy\tfirst_order\tcorrelation")
        );
        let mut first_order_sum = 0.0;
        for line in lines.take_while(|line| !line.contains('=')) {
            let fields: Vec<f64> = line
                .split('\t')
                .skip(3)
                .map(|f| f.parse().unwrap())
                .collect();
            assert!(
                (fields[0] - (fields[1] + fields[2])).abs() < 1e-12,
                "{line}"
            );
            first_order_sum += fields[1];
        }
        assert!((first_order_sum - line_value(&output, "First-order entropy")).abs() < 1e-9);
    }
}
