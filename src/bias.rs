//! Finite-sample bias correction for the Gaussian (quasi-harmonic) terms.
//!
//! A Gaussian entropy evaluated from a sample covariance is biased low,
//! because ln|S| underestimates ln|Sigma| at finite sample size. For n
//! independent samples the sample covariance is Wishart distributed with
//! nu = n - 1 degrees of freedom, and the expectation of its log-determinant
//! is known in closed form:
//!
//! ```text
//! <ln|S|> = ln|Sigma| + beta(nu, d),
//! beta(nu, d) = sum_{i=1..d} psi((nu - i + 1) / 2) + d ln(2 / nu)
//! ```
//!
//! so subtracting beta removes the bias exactly, with no resampling and at no
//! computational cost. Three places need it, and they are not the same amount:
//!
//! - a one-variable Gaussian entropy gets `-1/2 beta(nu, 1)`;
//! - a d-dimensional one gets `-1/2 beta(nu, d)`, which is what the
//!   correlation term 1/2 ln|C| carries once the marginals have taken their
//!   own share;
//! - a *pairwise mutual information* keeps almost none of it, because the
//!   corrections of S1(i), S1(j) and S2(i,j) very nearly cancel. What is left
//!   is [`pair_residual`], about 1/(2 nu) per pair and independent of rho.
//!
//! The last one is the one that matters at scale. It is negligible for a small
//! molecule and is not negligible for a binding site: with 10^5 coordinate
//! pairs and nu = 10^4 the uncorrected sum of pairwise mutual information is
//! biased by several kcal/mol, and the bias does not cancel between two
//! ensembles unless they have the same number of *independent* samples.
//!
//! Because beta depends on the number of independent samples rather than the
//! number of frames, [`QuasiHarmonicOptions::effective_frames`] lets the
//! caller pass n_eff. [`effective_frames_from_null`] estimates it from the
//! data: circularly shifting one copy of a column set against the other
//! destroys the cross-correlations while preserving every marginal and
//! autocorrelation, and the resulting null correlations have spread
//! 1/sqrt(nu).
//!
//! [`QuasiHarmonicOptions::effective_frames`]: crate::quasi_harmonic::QuasiHarmonicOptions::effective_frames

/// Whether the Gaussian terms carry a finite-sample bias correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BiasCorrection {
    /// Plug-in estimates, biased low at finite sample size. The historical
    /// behavior, kept as the default so existing results are unchanged.
    #[default]
    None,
    /// Subtract the Wishart expectation of the log-determinant.
    Wishart,
}

/// The digamma function, psi(x) = d/dx ln Gamma(x), for x > 0.
///
/// Recurrence up to x >= 10 followed by the standard asymptotic series, which
/// is accurate to near machine precision there. Returns NaN for x <= 0, where
/// the corrections below are meaningless anyway.
pub fn digamma(x: f64) -> f64 {
    if !(x > 0.0) {
        return f64::NAN;
    }
    let mut x = x;
    let mut result = 0.0;
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inverse = 1.0 / x;
    let t = inverse * inverse;
    result + x.ln() - 0.5 * inverse - t / 12.0 + t * t / 120.0 - t * t * t / 252.0
        + t * t * t * t / 240.0
        - t * t * t * t * t / 132.0
}

/// `beta(nu, d)`, the Wishart bias of `ln|S|` for a `d`-dimensional sample
/// covariance with `nu` degrees of freedom.
///
/// Returns `None` when `nu <= d`, where the sample covariance is singular and
/// the correction does not exist.
pub fn wishart_log_det_bias(nu: f64, dimensions: usize) -> Option<f64> {
    if !(nu > dimensions as f64) {
        return None;
    }
    let sum: f64 = (1..=dimensions)
        .map(|i| digamma((nu - i as f64 + 1.0) / 2.0))
        .sum();
    Some(sum + dimensions as f64 * (2.0 / nu).ln())
}

/// What to subtract from a one-variable Gaussian entropy: `1/2 beta(nu, 1)`.
pub fn marginal_correction(nu: f64) -> Option<f64> {
    wishart_log_det_bias(nu, 1).map(|beta| 0.5 * beta)
}

/// What to subtract from `1/2 ln|C|` once every marginal has already taken
/// [`marginal_correction`], so that the d-dimensional Gaussian entropy as a
/// whole carries exactly `-1/2 beta(nu, d)`.
pub fn correlation_correction(nu: f64, dimensions: usize) -> Option<f64> {
    let full = wishart_log_det_bias(nu, dimensions)?;
    let marginals = wishart_log_det_bias(nu, 1)? * dimensions as f64;
    Some(0.5 * (full - marginals))
}

/// What survives of the correction in a pairwise Gaussian mutual information:
/// `1/2 [psi(nu/2) - psi((nu-1)/2)]`, approximately `1/(2 nu)`.
///
/// `I2` corrected is `I2` plug-in minus this, so an entropy built as
/// `S1 - sum I2` gains it back once per pair.
pub fn pair_residual(nu: f64) -> Option<f64> {
    if !(nu > 1.0) {
        return None;
    }
    Some(0.5 * (digamma(nu / 2.0) - digamma((nu - 1.0) / 2.0)))
}

/// Estimate the number of independent samples from the spread of null
/// correlations, as described in the module documentation.
///
/// `columns` is `[coordinate][frame]`; `shifts` is how many circular shifts to
/// average over. Each shift contributes both lag directions as separate null
/// draws, since averaging them would halve the apparent spread and overstate
/// `n_eff` by a factor of two.
pub fn effective_frames_from_null(columns: &[Vec<f64>], frames_end: usize, shifts: usize) -> f64 {
    let dimensions = columns.len();
    if dimensions < 2 || frames_end < 4 || shifts == 0 {
        return frames_end as f64;
    }
    let standardized: Vec<Vec<f64>> = columns
        .iter()
        .map(|values| {
            let values = &values[..frames_end];
            let mean = values.iter().sum::<f64>() / frames_end as f64;
            let variance =
                values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / frames_end as f64;
            let scale = if variance > 0.0 {
                1.0 / variance.sqrt()
            } else {
                0.0
            };
            values.iter().map(|v| (v - mean) * scale).collect()
        })
        .collect();
    let mut sum_squares = 0.0;
    let mut count = 0usize;
    for shift in 1..=shifts {
        let lag = (shift * frames_end) / (shifts + 1);
        let lag = lag.max(1) % frames_end;
        for i in 0..dimensions {
            for j in 0..dimensions {
                if i == j {
                    continue;
                }
                let mut accumulator = 0.0;
                for frame in 0..frames_end {
                    accumulator +=
                        standardized[i][frame] * standardized[j][(frame + lag) % frames_end];
                }
                let correlation = accumulator / frames_end as f64;
                sum_squares += correlation * correlation;
                count += 1;
            }
        }
    }
    if count == 0 || sum_squares <= 0.0 {
        return frames_end as f64;
    }
    let variance = sum_squares / count as f64;
    (1.0 / variance).min(frames_end as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

    #[test]
    fn digamma_matches_known_values() {
        assert!((digamma(1.0) + EULER_GAMMA).abs() < 1e-12);
        // psi(1/2) = -gamma - 2 ln 2
        assert!((digamma(0.5) - (-EULER_GAMMA - 2.0 * 2.0_f64.ln())).abs() < 1e-12);
        // psi(n) = -gamma + H_{n-1}
        for n in 2..40 {
            let harmonic: f64 = (1..n).map(|i| 1.0 / i as f64).sum();
            assert!((digamma(n as f64) - (-EULER_GAMMA + harmonic)).abs() < 1e-11);
        }
        // recurrence psi(x+1) = psi(x) + 1/x
        for k in 1..60 {
            let x = k as f64 * 0.37;
            assert!((digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-11);
        }
    }

    #[test]
    fn pair_residual_is_about_one_over_two_nu() {
        for &nu in &[1.0e3, 1.0e4, 1.0e5] {
            let residual = pair_residual(nu).unwrap();
            assert!((residual * 2.0 * nu - 1.0).abs() < 1e-2, "nu {nu}");
        }
    }

    #[test]
    fn corrections_are_consistent() {
        // d marginals plus the correlation remainder must equal the full
        // d-dimensional correction.
        let nu = 5000.0;
        for d in 1..12 {
            let full = 0.5 * wishart_log_det_bias(nu, d).unwrap();
            let parts =
                d as f64 * marginal_correction(nu).unwrap() + correlation_correction(nu, d).unwrap();
            assert!((full - parts).abs() < 1e-12, "d {d}");
        }
    }

    #[test]
    fn pair_residual_equals_the_two_dimensional_remainder() {
        // For d = 2 the S1/S2 form and the full Gaussian must agree, which
        // forces beta(nu,2)/2 - beta(nu,1) to be minus the pair residual.
        let nu = 12_345.0;
        let remainder = correlation_correction(nu, 2).unwrap();
        assert!((remainder + pair_residual(nu).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn singular_cases_return_none() {
        assert!(wishart_log_det_bias(3.0, 4).is_none());
        assert!(wishart_log_det_bias(4.0, 4).is_none());
        assert!(wishart_log_det_bias(5.0, 4).is_some());
    }
}
