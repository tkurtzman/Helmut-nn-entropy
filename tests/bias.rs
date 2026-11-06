//! The Wishart finite-sample correction for the Gaussian terms.
//!
//! Three things are checked: that the default is unchanged, that the
//! correction has the size it claims to have on data whose entropy is known in
//! closed form, and that the S1/S2 form and the full Gaussian agree once
//! corrected, which they must for two coordinates.

use molentropy::bias::{
    BiasCorrection, correlation_correction, digamma, marginal_correction, pair_residual,
    wishart_log_det_bias,
};
use molentropy::quasi_harmonic::{QuasiHarmonicOptions, estimate_quasi_harmonic_entropy};
use molentropy::{CoordinateMetric, EntropyOptions};

/// A self-contained generator, so the test does not depend on a crate.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Box-Muller.
    fn normal(&mut self) -> f64 {
        let u = self.uniform().max(1e-300);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn independent_normals(dimensions: usize, frames: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng(seed);
    (0..dimensions)
        .map(|_| (0..frames).map(|_| rng.normal()).collect())
        .collect()
}

fn linear(dimensions: usize) -> Vec<CoordinateMetric> {
    vec![CoordinateMetric::Linear; dimensions]
}

fn quasi_harmonic(data: &[Vec<f64>], bias: BiasCorrection) -> f64 {
    let metrics = linear(data.len());
    let options = EntropyOptions {
        metrics: Some(&metrics),
        ..Default::default()
    };
    estimate_quasi_harmonic_entropy(
        data,
        data[0].len(),
        &QuasiHarmonicOptions {
            bias,
            ..Default::default()
        },
        &options,
    )
    .unwrap()
    .quasi_harmonic
}

#[test]
fn default_is_unchanged() {
    let data = independent_normals(6, 4000, 0x5eed);
    assert_eq!(
        quasi_harmonic(&data, BiasCorrection::None),
        quasi_harmonic(&data, Default::default())
    );
}

#[test]
fn the_correction_removes_the_known_bias() {
    // d independent standard normals have entropy d/2 ln(2 pi e) exactly.
    let dimensions = 20;
    let truth = 0.5 * dimensions as f64 * (std::f64::consts::TAU * std::f64::consts::E).ln();
    for &frames in &[60usize, 120, 400] {
        let nu = frames as f64 - 1.0;
        let expected_bias = 0.5 * wishart_log_det_bias(nu, dimensions).unwrap();
        // average over seeds so the comparison is with the expectation
        let (mut plain, mut corrected) = (0.0, 0.0);
        let repeats = 40;
        for seed in 0..repeats {
            let data = independent_normals(dimensions, frames, 0x1234 + seed * 2 + 1);
            plain += quasi_harmonic(&data, BiasCorrection::None);
            corrected += quasi_harmonic(&data, BiasCorrection::Wishart);
        }
        plain /= repeats as f64;
        corrected /= repeats as f64;
        // the uncorrected estimate sits a known distance below the truth
        assert!(
            (plain - (truth + expected_bias)).abs() < 0.25,
            "frames {frames}: plain {plain}, expected {}",
            truth + expected_bias
        );
        // and the corrected one sits on it
        assert!(
            (corrected - truth).abs() < 0.25,
            "frames {frames}: corrected {corrected}, truth {truth}"
        );
        // the correction is not cosmetic at these sample sizes
        assert!((plain - corrected).abs() > 0.1, "frames {frames}");
    }
}

#[test]
fn correction_shrinks_with_sample_size() {
    // beta(nu, 1) ~ -1/nu, so the marginal correction is negative and is
    // *subtracted*, raising the entropy by about 1/(2 nu). Both the sign and
    // the size matter: a sign slip here would push every Gaussian term the
    // wrong way.
    for &nu in &[99.0, 999.0, 9999.0] {
        let correction = marginal_correction(nu).unwrap();
        assert!(correction < 0.0, "nu {nu}: {correction}");
        assert!(
            (correction * -2.0 * nu - 1.0).abs() < 0.05,
            "nu {nu}: {correction} is not about -1/(2 nu)"
        );
    }
    assert!(marginal_correction(99.0).unwrap().abs() > marginal_correction(9999.0).unwrap().abs());
    // and the d-dimensional remainder vanishes as the sample grows
    assert!(correlation_correction(9999.0, 10).unwrap().abs() < 0.01);
    assert!(
        correlation_correction(99.0, 10).unwrap().abs()
            > correlation_correction(9999.0, 10).unwrap().abs()
    );
}

#[test]
fn pair_and_full_corrections_agree_for_two_coordinates() {
    // S1(i) + S1(j) - I2(i,j) must equal the two-dimensional Gaussian entropy,
    // corrections included. That forces the pair residual to be exactly minus
    // the two-dimensional correlation remainder.
    for &nu in &[50.0, 500.0, 50_000.0] {
        let remainder = correlation_correction(nu, 2).unwrap();
        let residual = pair_residual(nu).unwrap();
        assert!((remainder + residual).abs() < 1e-13, "nu {nu}");
    }
}

#[test]
fn digamma_is_accurate() {
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    assert!((digamma(1.0) + EULER_GAMMA).abs() < 1e-13);
    assert!((digamma(0.5) - (-EULER_GAMMA - 2.0 * 2.0_f64.ln())).abs() < 1e-13);
}

#[test]
fn effective_frames_overrides_the_frame_count() {
    let data = independent_normals(5, 2000, 0xabc);
    let metrics = linear(5);
    let options = EntropyOptions {
        metrics: Some(&metrics),
        ..Default::default()
    };
    let with = |effective_frames| {
        estimate_quasi_harmonic_entropy(
            &data,
            2000,
            &QuasiHarmonicOptions {
                bias: BiasCorrection::Wishart,
                effective_frames,
                ..Default::default()
            },
            &options,
        )
        .unwrap()
        .quasi_harmonic
    };
    // Claiming ten times fewer independent samples makes the correction
    // ten times larger, so the corrected entropy rises.
    assert!(with(Some(200.0)) > with(None));
}
