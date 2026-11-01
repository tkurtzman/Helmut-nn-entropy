//! Quasi-harmonic (QH) and Boltzmann quasi-harmonic (BQH) entropy.
//!
//! Both treat correlations through the correlation matrix C of the
//! coordinates, C_ij = sigma_ij / sqrt(sigma_ii sigma_jj), which contributes
//! 1/2 ln|C| <= 0. They differ in the first-order term:
//!
//! - QH (Karplus and Kushick, 1981) fits a Gaussian to each coordinate, which
//!   contributes 1/2 ln(2 pi e sigma_ii).
//! - BQH (Harpole and Sharp, J. Phys. Chem. B 2011) replaces the Gaussian
//!   marginals by the entropy of each coordinate's actual distribution,
//!   -integral p ln p, which matters for multimodal coordinates such as
//!   torsions with several rotamers.
//!
//! All values are differential entropies in units of k_B, like the
//! nearest-neighbor estimators, so they can be compared directly.
//!
//! Periodic coordinates enter C as z = e^(i 2 pi q / P), following Harpole
//! and Sharp, which makes C Hermitian; see [`PeriodicCorrelation`]. For the
//! first-order Gaussian term they are centered on their circular mean and
//! wrapped to (-P/2, P/2], so sigma_ii is in the coordinate's own units.

use rayon::prelude::*;

use crate::bias::{self, BiasCorrection};
use crate::{CoordinateMetric, EntropyOptions, coordinate_entropy_impl, resolve_input};

/// How the first-order entropy -integral p ln p of each coordinate is estimated
/// for BQH.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarginalEstimator {
    /// The one-dimensional nearest-neighbor estimate, the same first-order term
    /// as the MIE. Uses `EntropyOptions::constant` and `theiler_window`.
    #[default]
    NearestNeighbor,
    /// A histogram of `bins` equal bins, -sum p_k ln(p_k / width), Harpole and
    /// Sharp's eq 9. Linear coordinates are binned over their observed range,
    /// periodic ones over a full period. Some later BQH work reports the
    /// discrete entropy -sum p_k ln p_k, which is this value minus ln(width).
    Histogram {
        /// Number of equal-width bins per coordinate.
        bins: usize,
    },
}

/// How periodic coordinates enter the correlation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PeriodicCorrelation {
    /// As the complex number z = e^(i 2 pi q / P), so C is Hermitian (Harpole
    /// and Sharp). Correlations on the circle, such as the fixed 120 degree
    /// offsets between the hydrogens of a rotating methyl group, are captured
    /// whatever the angles' spread.
    #[default]
    Complex,
    /// Centered on the circular mean and wrapped to (-P/2, P/2], then treated
    /// as a real coordinate. Adequate for coordinates confined to one narrow
    /// well, but a correlation between coordinates that sweep the circle looks
    /// like a weak sawtooth and is largely missed.
    Centered,
}

/// Settings specific to the quasi-harmonic estimators.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QuasiHarmonicOptions {
    /// How BQH estimates each coordinate's first-order entropy.
    pub marginals: MarginalEstimator,
    /// How periodic coordinates enter the correlation matrix.
    pub periodic: PeriodicCorrelation,
    /// Whether the Gaussian terms carry the Wishart finite-sample bias
    /// correction. Defaults to [`BiasCorrection::None`], the historical
    /// behavior.
    pub bias: BiasCorrection,
    /// The number of *independent* samples, if it differs from the number of
    /// frames. The bias correction is a function of this, not of the frame
    /// count; see [`crate::bias::effective_frames_from_null`].
    pub effective_frames: Option<f64>,
}

/// The degrees of freedom the bias correction is evaluated at.
fn degrees_of_freedom(quasi_harmonic: &QuasiHarmonicOptions, frames_end: usize) -> f64 {
    quasi_harmonic.effective_frames.unwrap_or(frames_end as f64) - 1.0
}

/// QH and BQH totals with the terms they are built from, all in units of k_B.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuasiHarmonicEntropy {
    /// `first_order_gaussian + correlation`. Without periodic coordinates this
    /// is 1/2 ln((2 pi e)^n |sigma|).
    pub quasi_harmonic: f64,
    /// `first_order_boltzmann + correlation`.
    pub boltzmann_quasi_harmonic: f64,
    /// Sum over coordinates of 1/2 ln(2 pi e sigma_ii).
    pub first_order_gaussian: f64,
    /// Sum over coordinates of -integral p ln p, from the marginal estimator.
    pub first_order_boltzmann: f64,
    /// 1/2 ln|C|, never positive.
    pub correlation: f64,
}

/// Each coordinate's share of the QH and BQH entropy, in units of k_B. Every
/// vector has one entry per coordinate and sums to the matching field of
/// [`QuasiHarmonicEntropy`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateQuasiHarmonicEntropy {
    /// `first_order_gaussian + correlation`, per coordinate.
    pub quasi_harmonic: Vec<f64>,
    /// `first_order_boltzmann + correlation`, per coordinate.
    pub boltzmann_quasi_harmonic: Vec<f64>,
    /// 1/2 ln(2 pi e sigma_ii).
    pub first_order_gaussian: Vec<f64>,
    /// -integral p ln p of the coordinate, from the marginal estimator.
    pub first_order_boltzmann: Vec<f64>,
    /// 1/2 (log C)_ii, never positive. See [`estimate_coordinate_quasi_harmonic_entropy`].
    pub correlation: Vec<f64>,
}

/// QH and BQH entropy of the coordinates in `one_d_data`.
///
/// `options.metrics` marks the periodic coordinates. The correlation matrix
/// must be nonsingular, which needs more frames than coordinates and no
/// coordinate that is a linear combination of others.
pub fn estimate_quasi_harmonic_entropy(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    quasi_harmonic: &QuasiHarmonicOptions,
    options: &EntropyOptions,
) -> Result<QuasiHarmonicEntropy, String> {
    quasi_harmonic_terms(one_d_data, frames_end, quasi_harmonic, options, true)
}

/// [`estimate_quasi_harmonic_entropy`], optionally without the first-order
/// Boltzmann term, whose fields are then NaN.
pub(crate) fn quasi_harmonic_terms(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    quasi_harmonic: &QuasiHarmonicOptions,
    options: &EntropyOptions,
    need_boltzmann: bool,
) -> Result<QuasiHarmonicEntropy, String> {
    let terms = terms(
        one_d_data,
        frames_end,
        quasi_harmonic,
        options,
        need_boltzmann,
    )?;
    let first_order_gaussian = terms.first_order_gaussian.iter().sum();
    let first_order_boltzmann = terms.first_order_boltzmann.iter().sum();
    let mut correlation = 0.5 * log_determinant(&terms.correlation)?;
    if quasi_harmonic.bias == BiasCorrection::Wishart {
        let nu = degrees_of_freedom(quasi_harmonic, frames_end);
        let dimensions = terms.first_order_gaussian.len();
        correlation -= bias::correlation_correction(nu, dimensions).ok_or_else(|| {
            format!(
                "the Wishart correction needs more than {dimensions} independent samples, got {nu} degrees of freedom"
            )
        })?;
    }
    Ok(QuasiHarmonicEntropy {
        quasi_harmonic: first_order_gaussian + correlation,
        boltzmann_quasi_harmonic: first_order_boltzmann + correlation,
        first_order_gaussian,
        first_order_boltzmann,
        correlation,
    })
}

/// Each coordinate's share of the QH and BQH entropy of `one_d_data`.
///
/// The first-order terms belong to single coordinates already. The
/// correlation term 1/2 ln|C| belongs to the whole set; it is shared out using
/// ln|C| = tr(log C), so coordinate i receives 1/2 (log C)_ii. The shares sum
/// to the correlation term exactly, do not depend on the order of the
/// coordinates, and are each at most 0 because C has a unit diagonal. A
/// coordinate correlated with many others carries a large negative share; one
/// independent of the rest carries 0.
///
/// The totals match [`estimate_quasi_harmonic_entropy`] up to rounding, and the
/// same conditions apply.
pub fn estimate_coordinate_quasi_harmonic_entropy(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    quasi_harmonic: &QuasiHarmonicOptions,
    options: &EntropyOptions,
) -> Result<CoordinateQuasiHarmonicEntropy, String> {
    coordinate_quasi_harmonic_terms(one_d_data, frames_end, quasi_harmonic, options, true)
}

/// [`estimate_coordinate_quasi_harmonic_entropy`], optionally without the
/// first-order Boltzmann terms, whose fields are then NaN.
pub(crate) fn coordinate_quasi_harmonic_terms(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    quasi_harmonic: &QuasiHarmonicOptions,
    options: &EntropyOptions,
    need_boltzmann: bool,
) -> Result<CoordinateQuasiHarmonicEntropy, String> {
    let terms = terms(
        one_d_data,
        frames_end,
        quasi_harmonic,
        options,
        need_boltzmann,
    )?;
    let correlation: Vec<f64> = log_diagonal(&terms.correlation)?
        .into_iter()
        .map(|value| 0.5 * value)
        .collect();
    let add = |first_order: &[f64]| -> Vec<f64> {
        first_order
            .iter()
            .zip(&correlation)
            .map(|(a, b)| a + b)
            .collect()
    };
    Ok(CoordinateQuasiHarmonicEntropy {
        quasi_harmonic: add(&terms.first_order_gaussian),
        boltzmann_quasi_harmonic: add(&terms.first_order_boltzmann),
        first_order_gaussian: terms.first_order_gaussian,
        first_order_boltzmann: terms.first_order_boltzmann,
        correlation,
    })
}

/// QH truncated at first or second order in the mutual information expansion,
/// and each coordinate's share of it.
///
/// The first-order term is QH's, sum over i of 1/2 ln(2 pi e sigma_ii): the
/// hard coordinates as independent Gaussians. At first order the correlation
/// term 1/2 ln|C| is dropped. At second order it is replaced by its pairwise
/// expansion, 1/2 sum over i < j of ln(1 - |C_ij|^2), minus the Gaussian
/// mutual information of each pair. For a Gaussian the full term is exact and
/// this is an approximation to it, which misses the dependence shared among
/// three or more coordinates: it overcounts redundant dependences (three
/// coordinates all correlated with one another) and undercounts synergistic
/// ones (two independent coordinates that together determine a third). Each
/// pair's term is shared equally between its two coordinates, as in the MIE.
///
/// The same conditions apply as for [`estimate_quasi_harmonic_entropy`], except
/// that only each pair, not the whole correlation matrix, must be nonsingular.
pub(crate) fn truncated_quasi_harmonic_terms(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    quasi_harmonic: &QuasiHarmonicOptions,
    options: &EntropyOptions,
    order: usize,
) -> Result<(f64, Vec<f64>), String> {
    debug_assert!(order == 1 || order == 2, "order {order}");
    let terms = terms(one_d_data, frames_end, quasi_harmonic, options, false)?;
    let mut shares = terms.first_order_gaussian;
    if order == 2 {
        let tolerance = singular_tolerance(2);
        // S = S1 - sum I2, and the corrected I2 is the plug-in value minus the
        // residual, so each pair hands the residual back to the entropy, split
        // evenly between its two coordinates.
        let half_residual = match quasi_harmonic.bias {
            BiasCorrection::None => 0.0,
            BiasCorrection::Wishart => {
                let nu = degrees_of_freedom(quasi_harmonic, frames_end);
                0.5 * bias::pair_residual(nu).ok_or_else(|| {
                    format!("the Wishart correction needs more than 2 independent samples, got {nu} degrees of freedom")
                })?
            }
        };
        for (i, row) in terms.correlation.iter().enumerate() {
            for (j, &(real, imaginary)) in row.iter().enumerate().skip(i + 1) {
                let unexplained = 1.0 - (real * real + imaginary * imaginary);
                if !unexplained.is_finite() || unexplained <= tolerance {
                    return Err(format!(
                        "coordinates {i} and {j} are perfectly correlated, so their pairwise mutual information is infinite"
                    ));
                }
                let half_pair = 0.25 * unexplained.ln() + half_residual;
                shares[i] += half_pair;
                shares[j] += half_pair;
            }
        }
    }
    Ok((shares.iter().sum(), shares))
}

/// The per-coordinate first-order terms and the correlation matrix.
struct Terms {
    first_order_gaussian: Vec<f64>,
    first_order_boltzmann: Vec<f64>,
    correlation: Vec<Vec<Complex>>,
}

/// `need_boltzmann = false` skips the first-order Boltzmann terms, which QH
/// does not use, and fills them with NaN.
fn terms(
    one_d_data: &[Vec<f64>],
    frames_end: usize,
    quasi_harmonic: &QuasiHarmonicOptions,
    options: &EntropyOptions,
    need_boltzmann: bool,
) -> Result<Terms, String> {
    let (one_d_data, metrics) = resolve_input(one_d_data, frames_end, options)?;
    let dimensions = one_d_data.len();
    if frames_end <= dimensions {
        return Err(format!(
            "the correlation matrix of {dimensions} coordinates needs more than {dimensions} frames, got {frames_end}"
        ));
    }

    let centered: Vec<Vec<f64>> = one_d_data
        .par_iter()
        .zip(metrics.par_iter())
        .map(|(values, metric)| center(values, *metric))
        .collect();
    let marginal_bias = match quasi_harmonic.bias {
        BiasCorrection::None => 0.0,
        BiasCorrection::Wishart => {
            let nu = degrees_of_freedom(quasi_harmonic, frames_end);
            bias::marginal_correction(nu).ok_or_else(|| {
                format!("the Wishart correction needs more than 1 independent sample, got {nu} degrees of freedom")
            })?
        }
    };
    let first_order_gaussian = centered
        .iter()
        .map(|values| {
            let variance = mean_square(values);
            0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * variance).ln() - marginal_bias
        })
        .collect();

    let deviations: Vec<Deviations> = centered
        .into_par_iter()
        .zip(one_d_data.par_iter().zip(metrics.par_iter()))
        .map(
            |(centered, (values, metric))| match (metric, quasi_harmonic.periodic) {
                (CoordinateMetric::Periodic { period }, PeriodicCorrelation::Complex) => {
                    complex_deviations(values, *period)
                }
                _ => Deviations {
                    real: centered,
                    imaginary: None,
                },
            },
        )
        .map(Deviations::standardized)
        .collect();
    let correlation = correlation_matrix(&deviations);

    if !need_boltzmann {
        return Ok(Terms {
            first_order_boltzmann: vec![f64::NAN; dimensions],
            first_order_gaussian,
            correlation,
        });
    }
    let first_order_boltzmann = match quasi_harmonic.marginals {
        MarginalEstimator::NearestNeighbor => coordinate_entropy_impl(
            &one_d_data,
            &metrics,
            options.constant,
            options.theiler_window,
        )?,
        MarginalEstimator::Histogram { bins } => {
            if bins == 0 {
                return Err("histogram needs at least one bin".to_string());
            }
            one_d_data
                .iter()
                .zip(&metrics)
                .map(|(values, metric)| histogram_entropy(values, *metric, bins))
                .collect()
        }
    };

    Ok(Terms {
        first_order_gaussian,
        first_order_boltzmann,
        correlation,
    })
}

/// How one coordinate enters a correlation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Representation {
    /// As its value; for a periodic coordinate, after centering on its
    /// circular mean and wrapping (see [`center`]).
    Real,
    /// A periodic coordinate as e^(i 2 pi q / P), one complex entry.
    Complex,
    /// A periodic coordinate as cos and sin of 2 pi k q / P for k = 1 to the
    /// given number of harmonics, two real entries per harmonic.
    Harmonics(usize),
}

impl Representation {
    /// How [`PeriodicCorrelation`] represents a coordinate with `metric`.
    pub(crate) fn of(metric: CoordinateMetric, periodic: PeriodicCorrelation) -> Self {
        match (metric, periodic) {
            (CoordinateMetric::Periodic { .. }, PeriodicCorrelation::Complex) => Self::Complex,
            _ => Self::Real,
        }
    }
}

/// ln|C| of the correlation matrix of `columns`, each entering as
/// `representations` says (a linear coordinate is always real). The
/// first-order terms are not computed.
pub(crate) fn correlation_log_det(
    columns: &[&[f64]],
    metrics: &[CoordinateMetric],
    representations: &[Representation],
) -> Result<f64, String> {
    let (deviations, _) = represented_deviations(columns, metrics, representations)?;
    log_determinant(&correlation_matrix(&deviations))
}

/// Each column's share of [`correlation_log_det`]: the sum of (log C)_ii over
/// the entries of C that the column contributes (one, or two per harmonic).
/// The shares sum to ln|C|.
pub(crate) fn correlation_log_diagonal(
    columns: &[&[f64]],
    metrics: &[CoordinateMetric],
    representations: &[Representation],
) -> Result<Vec<f64>, String> {
    let (deviations, owners) = represented_deviations(columns, metrics, representations)?;
    let mut shares = vec![0.0; columns.len()];
    for (owner, value) in owners
        .into_iter()
        .zip(log_diagonal(&correlation_matrix(&deviations))?)
    {
        shares[owner] += value;
    }
    Ok(shares)
}

/// The standardized deviations of `columns` as `representations` says, and
/// the column each came from.
fn represented_deviations(
    columns: &[&[f64]],
    metrics: &[CoordinateMetric],
    representations: &[Representation],
) -> Result<(Vec<Deviations>, Vec<usize>), String> {
    let (deviations, owners): (Vec<Deviations>, Vec<usize>) = columns
        .par_iter()
        .zip(metrics.par_iter().zip(representations.par_iter()))
        .enumerate()
        .flat_map_iter(|(owner, (values, (metric, representation)))| {
            let entries = match (*metric, *representation) {
                (CoordinateMetric::Periodic { period }, Representation::Complex) => {
                    vec![complex_deviations(values, period)]
                }
                (CoordinateMetric::Periodic { period }, Representation::Harmonics(count)) => (1
                    ..=count)
                    .flat_map(|harmonic| {
                        let complex = harmonic_deviations(values, period, harmonic);
                        [
                            Deviations {
                                real: complex.real,
                                imaginary: None,
                            },
                            Deviations {
                                real: complex.imaginary.expect("complex deviations"),
                                imaginary: None,
                            },
                        ]
                    })
                    .collect(),
                (metric, _) => vec![Deviations {
                    real: center(values, metric),
                    imaginary: None,
                }],
            };
            entries
                .into_iter()
                .map(move |entry| (entry.standardized(), owner))
        })
        .unzip();
    let frames = columns.first().map_or(0, |column| column.len());
    if frames <= deviations.len() {
        return Err(format!(
            "the correlation matrix of {} entries needs more than {} frames, got {frames}",
            deviations.len(),
            deviations.len()
        ));
    }
    Ok((deviations, owners))
}

fn mean_square(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64
}

/// Deviations of one coordinate from its mean, complex for a periodic
/// coordinate under [`PeriodicCorrelation::Complex`].
struct Deviations {
    real: Vec<f64>,
    imaginary: Option<Vec<f64>>,
}

impl Deviations {
    /// Scaled to unit mean |deviation|^2, so the correlation matrix has a unit
    /// diagonal.
    fn standardized(mut self) -> Self {
        let variance = mean_square(&self.real) + self.imaginary.as_deref().map_or(0.0, mean_square);
        let scale = variance.sqrt().recip();
        self.real.iter_mut().for_each(|v| *v *= scale);
        if let Some(imaginary) = self.imaginary.as_mut() {
            imaginary.iter_mut().for_each(|v| *v *= scale);
        }
        self
    }
}

/// z - <z> for z = e^(i 2 pi q / P).
fn complex_deviations(values: &[f64], period: f64) -> Deviations {
    harmonic_deviations(values, period, 1)
}

/// z - <z> for z = e^(i 2 pi k q / P), the k-th harmonic.
fn harmonic_deviations(values: &[f64], period: f64, harmonic: usize) -> Deviations {
    let to_radians = std::f64::consts::TAU / period;
    let k = harmonic as f64;
    let (real, imaginary): (Vec<f64>, Vec<f64>) = values
        .iter()
        .map(|v| (k * (v * to_radians)).cos())
        .zip(values.iter().map(|v| (k * (v * to_radians)).sin()))
        .unzip();
    let center = |series: Vec<f64>| {
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        series.into_iter().map(|v| v - mean).collect::<Vec<f64>>()
    };
    Deviations {
        real: center(real),
        imaginary: Some(center(imaginary)),
    }
}

/// Deviations from the mean. A periodic coordinate is first rotated so its
/// circular mean sits at 0 and wrapped to (-P/2, P/2].
fn center(values: &[f64], metric: CoordinateMetric) -> Vec<f64> {
    let shifted: Vec<f64> = match metric {
        CoordinateMetric::Linear => values.to_vec(),
        CoordinateMetric::Periodic { period } => {
            let to_radians = std::f64::consts::TAU / period;
            let (sin, cos) = values.iter().fold((0.0, 0.0), |(s, c), v| {
                let angle = v * to_radians;
                (s + angle.sin(), c + angle.cos())
            });
            let circular_mean = sin.atan2(cos) / to_radians;
            let half = 0.5 * period;
            values
                .iter()
                .map(|v| half - (half - (v - circular_mean)).rem_euclid(period))
                .collect()
        }
    };
    let mean = shifted.iter().sum::<f64>() / shifted.len() as f64;
    shifted.iter().map(|v| v - mean).collect()
}

/// A complex number as (real, imaginary).
type Complex = (f64, f64);

/// C_ij = mean over frames of a_i conj(a_j) for standardized deviations a.
/// Hermitian with a unit diagonal; real when no deviation is complex.
fn correlation_matrix(deviations: &[Deviations]) -> Vec<Vec<Complex>> {
    let dimensions = deviations.len();
    let frames = deviations[0].real.len() as f64;
    let dot = |left: &[f64], right: &[f64]| left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let upper: Vec<Vec<Complex>> = (0..dimensions)
        .into_par_iter()
        .map(|i| {
            (i..dimensions)
                .map(|j| {
                    if i == j {
                        return (1.0, 0.0);
                    }
                    let (a, b) = (&deviations[i], &deviations[j]);
                    // (ar + i ai)(br - i bi) = ar br + ai bi + i (ai br - ar bi)
                    let mut real = dot(&a.real, &b.real);
                    let mut imaginary = 0.0;
                    if let Some(ai) = &a.imaginary {
                        imaginary += dot(ai, &b.real);
                    }
                    if let Some(bi) = &b.imaginary {
                        imaginary -= dot(&a.real, bi);
                    }
                    if let (Some(ai), Some(bi)) = (&a.imaginary, &b.imaginary) {
                        real += dot(ai, bi);
                    }
                    (real / frames, imaginary / frames)
                })
                .collect()
        })
        .collect();
    let mut matrix = vec![vec![(0.0, 0.0); dimensions]; dimensions];
    for (i, row) in upper.iter().enumerate() {
        for (offset, &(real, imaginary)) in row.iter().enumerate() {
            matrix[i][i + offset] = (real, imaginary);
            matrix[i + offset][i] = (real, -imaginary);
        }
    }
    matrix
}

/// ln|A| of a Hermitian positive-definite matrix, from its Cholesky factor
/// A = L L^H, whose diagonal is real and positive.
/// Below this, a Cholesky pivot or an eigenvalue of an n x n correlation
/// matrix cannot be told apart from zero. C has a unit diagonal, so both
/// measure a coordinate's variance left unexplained, relative to 1, and the
/// rounding error in C grows with n.
fn singular_tolerance(n: usize) -> f64 {
    64.0 * f64::EPSILON * n as f64
}

fn log_determinant(matrix: &[Vec<Complex>]) -> Result<f64, String> {
    let n = matrix.len();
    let tolerance = singular_tolerance(n);
    let mut lower = vec![vec![(0.0, 0.0); n]; n];
    let mut log_det = 0.0;
    for j in 0..n {
        let diagonal = matrix[j][j].0
            - lower[j][..j]
                .iter()
                .map(|(re, im)| re * re + im * im)
                .sum::<f64>();
        // A NaN or vanishing pivot means C is singular.
        if !diagonal.is_finite() || diagonal <= tolerance {
            return Err(format!(
                "the correlation matrix is singular: coordinate {j} is a linear combination of earlier coordinates"
            ));
        }
        let pivot = diagonal.sqrt();
        lower[j][j] = (pivot, 0.0);
        log_det += 2.0 * pivot.ln();
        for i in (j + 1)..n {
            // sum over k of L_ik conj(L_jk)
            let (mut re, mut im) = (0.0, 0.0);
            for (a, b) in lower[i][..j].iter().zip(&lower[j][..j]) {
                re += a.0 * b.0 + a.1 * b.1;
                im += a.1 * b.0 - a.0 * b.1;
            }
            lower[i][j] = ((matrix[i][j].0 - re) / pivot, (matrix[i][j].1 - im) / pivot);
        }
    }
    Ok(log_det)
}

/// The diagonal of log A for a Hermitian positive-definite matrix A, from its
/// eigendecomposition A = V diag(lambda) V^H: (log A)_ii = sum over k of
/// |V_ik|^2 ln lambda_k.
fn log_diagonal(matrix: &[Vec<Complex>]) -> Result<Vec<f64>, String> {
    let n = matrix.len();
    let tolerance = singular_tolerance(n);
    let singular = || {
        "the correlation matrix is singular: some coordinate is a linear combination of others"
            .to_string()
    };
    let log_eigenvalues = |eigenvalues: &[f64]| -> Result<Vec<f64>, String> {
        eigenvalues
            .iter()
            .map(|&lambda| {
                if lambda.is_finite() && lambda > tolerance {
                    Ok(lambda.ln())
                } else {
                    Err(singular())
                }
            })
            .collect()
    };
    if matrix
        .iter()
        .flatten()
        .all(|&(_, imaginary)| imaginary == 0.0)
    {
        let a = nalgebra::DMatrix::from_fn(n, n, |i, j| matrix[i][j].0);
        let eigen = a.symmetric_eigen();
        let logs = log_eigenvalues(eigen.eigenvalues.as_slice())?;
        Ok((0..n)
            .map(|i| {
                (0..n)
                    .map(|k| eigen.eigenvectors[(i, k)].powi(2) * logs[k])
                    .sum()
            })
            .collect())
    } else {
        let a = nalgebra::DMatrix::from_fn(n, n, |i, j| {
            nalgebra::Complex::new(matrix[i][j].0, matrix[i][j].1)
        });
        let eigen = a.symmetric_eigen();
        let logs = log_eigenvalues(eigen.eigenvalues.as_slice())?;
        Ok((0..n)
            .map(|i| {
                (0..n)
                    .map(|k| eigen.eigenvectors[(i, k)].norm_sqr() * logs[k])
                    .sum()
            })
            .collect())
    }
}

/// -sum p_k ln(p_k / width) over `bins` equal bins.
fn histogram_entropy(values: &[f64], metric: CoordinateMetric, bins: usize) -> f64 {
    let (low, width) = match metric {
        CoordinateMetric::Periodic { period } => (0.0, period / bins as f64),
        CoordinateMetric::Linear => {
            let low = values.iter().copied().fold(f64::INFINITY, f64::min);
            let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (low, (high - low) / bins as f64)
        }
    };
    let mut counts = vec![0usize; bins];
    for &value in values {
        let value = metric.canonicalize(value);
        // The maximum of a linear range lands exactly on the upper edge.
        let bin = (((value - low) / width) as usize).min(bins - 1);
        counts[bin] += 1;
    }
    let total = values.len() as f64;
    counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * (p / width).ln()
        })
        .sum()
}
